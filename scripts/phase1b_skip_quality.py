#!/usr/bin/env python3
"""
Phase 1b: Actual Skip + Video Quality Comparison
=================================================
Implements 6 skip strategies, generates actual videos, computes SSIM/PSNR.

Strategies:
  S0: No skip (baseline)
  S1: Skip first 25% steps (step 0-12) every other step
  S2: Skip first 50% steps (step 0-24) every other step
  S3: Skip all steps every other step
  S4: Skip first 25% steps entirely (extreme)
  S5: Adaptive threshold (skip when change < τ, uses 1a data if available)

Usage:
  python scripts/phase1b_skip_quality.py --output_dir results/phase1b
  python scripts/phase1b_skip_quality.py --num_prompts 3 --strategies S0,S1,S2
"""

import torch
import numpy as np
import json
import argparse
import gc
import time
from pathlib import Path
from datetime import datetime

from phase1_utils import (
    PROMPTS, load_wan_pipeline, install_skippable_processors,
    set_all_steps, clear_all_caches, generate_video,
    compute_ssim, compute_psnr, save_video_frames,
    get_env_info, get_latent_shape, json_convert,
)


# ─── Skip Strategies ─────────────────────────────────────────────────────────

def make_skip_controller(strategy, num_steps, skip_layers=None):
    """
    Returns a skip controller function: (step, layer_idx) -> bool.

    Args:
        strategy: str, one of S0-S5
        num_steps: total denoising steps
        skip_layers: optional list of layers to skip (None = all layers)
    """
    def layer_ok(layer_idx):
        if skip_layers is None:
            return True
        return layer_idx in skip_layers

    if strategy == 'S0':
        return lambda step, layer: False

    elif strategy == 'S1':
        # Skip first 25% every other step
        boundary = num_steps // 4
        return lambda step, layer: (step < boundary and step % 2 == 1 and layer_ok(layer))

    elif strategy == 'S2':
        # Skip first 50% every other step
        boundary = num_steps // 2
        return lambda step, layer: (step < boundary and step % 2 == 1 and layer_ok(layer))

    elif strategy == 'S3':
        # Skip all steps every other step
        return lambda step, layer: (step % 2 == 1 and layer_ok(layer))

    elif strategy == 'S4':
        # Skip first 25% entirely (no compute at all)
        boundary = num_steps // 4
        return lambda step, layer: (step > 0 and step < boundary and layer_ok(layer))

    elif strategy == 'S5':
        # Adaptive: skip odd steps in first 60%, every 3rd step in 60-85%
        boundary1 = int(num_steps * 0.6)
        boundary2 = int(num_steps * 0.85)
        return lambda step, layer: (
            (step < boundary1 and step % 2 == 1 and layer_ok(layer)) or
            (boundary1 <= step < boundary2 and step % 3 != 0 and layer_ok(layer))
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def count_skips(strategy, num_steps, num_layers):
    """Count how many (step, layer) pairs are skipped."""
    ctrl = make_skip_controller(strategy, num_steps)
    total = 0
    skipped = 0
    for s in range(num_steps):
        for l in range(num_layers):
            total += 1
            if s > 0 and ctrl(s, l):  # step 0 always computes
                skipped += 1
    return skipped, total


def run_experiment(args):
    print("=" * 60)
    print("Phase 1b: Skip + Quality Comparison")
    print("=" * 60)

    device = args.device
    env_info = get_env_info(device)
    print(f"Environment: {json.dumps(env_info, indent=2)}")

    strategies = args.strategies.split(',')
    if 'S0' not in strategies:
        strategies.insert(0, 'S0')  # Always need baseline
    print(f"Strategies: {strategies}")

    # ── Load model ──
    pipe = load_wan_pipeline(args.model, device)
    processors = install_skippable_processors(pipe.transformer)

    self_procs = {k: v for k, v in processors.items() if k.startswith('self_')}
    num_layers = len(self_procs)

    prompts = PROMPTS[:args.num_prompts]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Show skip counts ──
    print(f"\nSkip counts (num_layers={num_layers}):")
    for strat in strategies:
        skipped, total = count_skips(strat, args.num_steps, num_layers)
        print(f"  {strat}: {skipped}/{total} = {skipped/total*100:.1f}% skipped")

    # ── Run each strategy ──
    all_results = {}
    baseline_videos = {}  # prompt_idx -> frames tensor

    start_time = time.time()

    for strat in strategies:
        print(f"\n{'─' * 40}")
        print(f"Strategy: {strat}")
        print(f"{'─' * 40}")

        skip_ctrl = make_skip_controller(strat, args.num_steps)
        for proc in self_procs.values():
            proc.skip_controller = skip_ctrl

        strat_results = {
            'ssim': [],
            'psnr': [],
            'gen_times': [],
        }

        for pi, prompt in enumerate(prompts):
            t0 = time.time()
            print(f"  [{pi+1}/{len(prompts)}] \"{prompt[:50]}\"", end="", flush=True)

            frames = generate_video(
                pipe, prompt, processors,
                height=args.height, width=args.width,
                num_frames=args.num_frames, num_steps=args.num_steps,
                seed=args.seed, device=device,
            )

            gen_time = time.time() - t0
            strat_results['gen_times'].append(gen_time)

            if strat == 'S0':
                baseline_videos[pi] = frames
                strat_results['ssim'].append(1.0)
                strat_results['psnr'].append(100.0)
                print(f"  {gen_time:.1f}s (baseline)")
            else:
                ssim = compute_ssim(frames, baseline_videos[pi])
                psnr = compute_psnr(frames, baseline_videos[pi])
                strat_results['ssim'].append(ssim)
                strat_results['psnr'].append(psnr)
                print(f"  {gen_time:.1f}s  SSIM={ssim:.4f}  PSNR={psnr:.1f}dB")

            # Save sample frames for first prompt
            if pi == 0 and args.save_frames:
                save_video_frames(
                    frames,
                    output_dir / f"frames_{strat}",
                    prefix=f"{strat}"
                )

            gc.collect()
            torch.cuda.empty_cache()

        # Aggregate
        strat_results['ssim_mean'] = float(np.mean(strat_results['ssim']))
        strat_results['ssim_std'] = float(np.std(strat_results['ssim']))
        strat_results['psnr_mean'] = float(np.mean(strat_results['psnr']))
        strat_results['psnr_std'] = float(np.std(strat_results['psnr']))
        strat_results['gen_time_mean'] = float(np.mean(strat_results['gen_times']))

        skipped, total = count_skips(strat, args.num_steps, num_layers)
        strat_results['skip_ratio'] = skipped / total
        strat_results['speedup_theoretical'] = total / max(total - skipped, 1)

        all_results[strat] = strat_results

        print(f"\n  Mean SSIM: {strat_results['ssim_mean']:.4f} ± {strat_results['ssim_std']:.4f}")
        print(f"  Mean PSNR: {strat_results['psnr_mean']:.1f} ± {strat_results['psnr_std']:.1f} dB")
        print(f"  Mean gen time: {strat_results['gen_time_mean']:.1f}s")

    total_time = time.time() - start_time

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':>8} | {'SSIM':>10} | {'PSNR':>10} | {'Skip%':>6} | {'Time':>6}")
    print("-" * 55)
    for strat in strategies:
        r = all_results[strat]
        print(f"{strat:>8} | {r['ssim_mean']:.4f}±{r['ssim_std']:.3f} | "
              f"{r['psnr_mean']:5.1f}±{r['psnr_std']:.1f} | "
              f"{r['skip_ratio']*100:5.1f}% | {r['gen_time_mean']:5.1f}s")

    # ── Go/No-Go ──
    best_strat = None
    for strat in ['S3', 'S2', 'S1']:
        if strat in all_results and all_results[strat]['ssim_mean'] > 0.95:
            best_strat = strat
            break

    if best_strat:
        print(f"\n✅ GO: {best_strat} achieves SSIM > 0.95 ({all_results[best_strat]['ssim_mean']:.4f})")
    else:
        max_ssim_strat = max(
            [s for s in strategies if s != 'S0'],
            key=lambda s: all_results[s]['ssim_mean']
        )
        r = all_results[max_ssim_strat]
        if r['ssim_mean'] > 0.90:
            print(f"\n⚠️ CONDITIONAL: Best is {max_ssim_strat} with SSIM {r['ssim_mean']:.4f}")
        else:
            print(f"\n❌ NO GO: Best SSIM is {r['ssim_mean']:.4f} ({max_ssim_strat})")

    # ── Save ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_data = {
        'config': {
            'model': args.model,
            'height': args.height,
            'width': args.width,
            'num_frames': args.num_frames,
            'num_steps': args.num_steps,
            'num_prompts': args.num_prompts,
            'seed': args.seed,
            'strategies': strategies,
            'prompts': prompts,
        },
        'env': env_info,
        'total_time_seconds': round(total_time, 1),
        'results': all_results,
    }

    results_file = output_dir / f"skip_quality_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(save_data, f, default=json_convert)

    report_file = output_dir / f"skip_quality_{timestamp}.md"
    generate_report(save_data, report_file, args)

    print(f"\nResults: {results_file}")
    print(f"Report: {report_file}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")


def generate_report(data, report_file, args):
    """Generate markdown report."""
    env = data['env']
    cfg = data['config']
    results = data['results']

    lines = [
        f"# 实验：Phase 1b 注意力跳过质量对比",
        f"> 日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 环境",
        f"- Commit: `{env.get('git_commit', 'unknown')}`",
        f"- GPU: {env.get('gpu_name', 'unknown')}",
        f"- Python: {env.get('python_version', 'unknown')}",
        f"- PyTorch: {env.get('torch_version', 'unknown')}",
        f"- CUDA: {env.get('cuda_version', 'unknown')}",
        f"- Diffusers: {env.get('diffusers_version', 'unknown')}",
        "",
        "## 运行命令",
        "```bash",
        f"python scripts/phase1b_skip_quality.py --num_prompts {cfg['num_prompts']} "
        f"--strategies {','.join(cfg['strategies'])} --output_dir {args.output_dir}",
        "```",
        "",
        "## 策略说明",
        "",
        "| 策略 | 描述 | 跳过比例 |",
        "|------|------|---------|",
        "| S0 | 基准 (不跳过) | 0% |",
        "| S1 | 前25%步隔步跳 | ~12.5% |",
        "| S2 | 前50%步隔步跳 | ~25% |",
        "| S3 | 全程隔步跳 | ~50% |",
        "| S4 | 前25%步全跳 (极端) | ~25% |",
        "| S5 | 自适应 (前60%隔步+60-85%每3步) | ~35% |",
        "",
        "## 结果",
        "",
        "| 策略 | SSIM | PSNR (dB) | 跳过比例 | 平均耗时 |",
        "|------|------|-----------|---------|---------|",
    ]

    for strat in cfg['strategies']:
        r = results[strat]
        lines.append(
            f"| {strat} | {r['ssim_mean']:.4f}±{r['ssim_std']:.3f} | "
            f"{r['psnr_mean']:.1f}±{r['psnr_std']:.1f} | "
            f"{r['skip_ratio']*100:.1f}% | {r['gen_time_mean']:.1f}s |"
        )

    # Per-prompt details
    lines.extend([
        "",
        "## 逐 Prompt 详情",
        "",
        "| Prompt | " + " | ".join(cfg['strategies']) + " |",
        "|--------|" + "|".join(["------"] * len(cfg['strategies'])) + "|",
    ])

    for pi in range(cfg['num_prompts']):
        row = f"| {cfg['prompts'][pi][:30]}... |"
        for strat in cfg['strategies']:
            r = results[strat]
            row += f" {r['ssim'][pi]:.4f} |"
        lines.append(row)

    lines.extend([
        "",
        f"## 总耗时",
        f"{data['total_time_seconds']:.1f}s ({data['total_time_seconds']/60:.1f}min)",
    ])

    with open(report_file, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1b: Skip + Quality")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategies", default="S0,S1,S2,S3,S4,S5",
                        help="Comma-separated strategies to test")
    parser.add_argument("--save_frames", action="store_true",
                        help="Save video frames as PNGs")
    parser.add_argument("--output_dir", default="results/phase1b")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_experiment(args)
