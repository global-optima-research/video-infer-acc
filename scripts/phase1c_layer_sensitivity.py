#!/usr/bin/env python3
"""
Phase 1c: Per-Layer Sensitivity Ranking
========================================
Skip one layer at a time (reuse previous step output), measure SSIM/PSNR.
Produces a 30-layer sensitivity ranking.

Usage:
  python scripts/phase1c_layer_sensitivity.py --output_dir results/phase1c
  python scripts/phase1c_layer_sensitivity.py --num_prompts 3 --layers 0,1,2,15,28,29
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
    compute_ssim, compute_psnr,
    get_env_info, get_latent_shape, json_convert,
)


def run_experiment(args):
    print("=" * 60)
    print("Phase 1c: Per-Layer Sensitivity Ranking")
    print("=" * 60)

    device = args.device
    env_info = get_env_info(device)
    print(f"Environment: {json.dumps(env_info, indent=2)}")

    # ── Load model ──
    pipe = load_wan_pipeline(args.model, device)
    processors = install_skippable_processors(pipe.transformer)

    self_procs = {k: v for k, v in processors.items() if k.startswith('self_')}
    num_layers = len(self_procs)

    # Which layers to test
    if args.layers:
        test_layers = [int(x) for x in args.layers.split(',')]
    else:
        test_layers = list(range(num_layers))
    print(f"Testing {len(test_layers)} layers: {test_layers}")

    prompts = PROMPTS[:args.num_prompts]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # ── Generate baseline videos ──
    print("\n── Generating baseline videos (no skip) ──")
    baseline_videos = {}
    baseline_times = []
    for proc in self_procs.values():
        proc.skip_controller = None

    for pi, prompt in enumerate(prompts):
        t0 = time.time()
        print(f"  [{pi+1}/{len(prompts)}] \"{prompt[:50]}\"", end="", flush=True)

        frames = generate_video(
            pipe, prompt, processors,
            height=args.height, width=args.width,
            num_frames=args.num_frames, num_steps=args.num_steps,
            seed=args.seed, device=device,
        )
        baseline_videos[pi] = frames
        elapsed = time.time() - t0
        baseline_times.append(elapsed)
        print(f"  {elapsed:.1f}s")

        gc.collect()
        torch.cuda.empty_cache()

    # ── Test each layer ──
    layer_results = {}

    for li, target_layer in enumerate(test_layers):
        print(f"\n── Layer {target_layer} ({li+1}/{len(test_layers)}) ──")

        # Skip controller: always skip this layer (every step), all others normal
        def make_ctrl(tl):
            return lambda step, layer: (layer == tl)
        for proc in self_procs.values():
            proc.skip_controller = make_ctrl(target_layer)

        ssim_list = []
        psnr_list = []
        gen_times = []

        for pi, prompt in enumerate(prompts):
            t0 = time.time()
            print(f"  [{pi+1}/{len(prompts)}] \"{prompt[:30]}...\"", end="", flush=True)

            frames = generate_video(
                pipe, prompt, processors,
                height=args.height, width=args.width,
                num_frames=args.num_frames, num_steps=args.num_steps,
                seed=args.seed, device=device,
            )

            ssim = compute_ssim(frames, baseline_videos[pi])
            psnr = compute_psnr(frames, baseline_videos[pi])
            elapsed = time.time() - t0

            ssim_list.append(ssim)
            psnr_list.append(psnr)
            gen_times.append(elapsed)
            print(f"  SSIM={ssim:.4f} PSNR={psnr:.1f}dB {elapsed:.1f}s")

            gc.collect()
            torch.cuda.empty_cache()

        layer_results[target_layer] = {
            'ssim': ssim_list,
            'psnr': psnr_list,
            'ssim_mean': float(np.mean(ssim_list)),
            'ssim_std': float(np.std(ssim_list)),
            'psnr_mean': float(np.mean(psnr_list)),
            'psnr_std': float(np.std(psnr_list)),
            'gen_time_mean': float(np.mean(gen_times)),
        }

        print(f"  → Mean SSIM: {layer_results[target_layer]['ssim_mean']:.4f}")

    total_time = time.time() - start_time

    # ── Ranking ──
    print("\n" + "=" * 60)
    print("LAYER SENSITIVITY RANKING (most sensitive first)")
    print("=" * 60)

    ranked = sorted(layer_results.keys(), key=lambda l: layer_results[l]['ssim_mean'])
    print(f"{'Rank':>4} | {'Layer':>5} | {'SSIM':>10} | {'PSNR':>10} | Sensitivity")
    print("-" * 60)
    for rank, layer in enumerate(ranked):
        r = layer_results[layer]
        sensitivity = 1.0 - r['ssim_mean']
        bar = "█" * int(sensitivity * 500)
        print(f"{rank+1:4d} | {layer:5d} | {r['ssim_mean']:.4f}±{r['ssim_std']:.3f} | "
              f"{r['psnr_mean']:5.1f}±{r['psnr_std']:.1f} | {bar}")

    # Classify layers
    critical = [l for l in ranked if layer_results[l]['ssim_mean'] < 0.95]
    safe = [l for l in ranked if layer_results[l]['ssim_mean'] >= 0.99]
    moderate = [l for l in ranked if 0.95 <= layer_results[l]['ssim_mean'] < 0.99]

    print(f"\nCritical layers (SSIM < 0.95): {critical}")
    print(f"Moderate layers (0.95-0.99):   {moderate}")
    print(f"Safe layers (SSIM ≥ 0.99):     {safe}")
    print(f"Safe layer count: {len(safe)}/{len(test_layers)} = {len(safe)/len(test_layers)*100:.0f}%")

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
            'test_layers': test_layers,
            'prompts': prompts,
        },
        'env': env_info,
        'total_time_seconds': round(total_time, 1),
        'layer_results': {str(k): v for k, v in layer_results.items()},
        'ranking': ranked,
        'critical_layers': critical,
        'moderate_layers': moderate,
        'safe_layers': safe,
        'baseline_gen_time_mean': float(np.mean(baseline_times)),
    }

    results_file = output_dir / f"layer_sensitivity_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(save_data, f, default=json_convert)

    report_file = output_dir / f"layer_sensitivity_{timestamp}.md"
    generate_report(save_data, report_file, args)

    print(f"\nResults: {results_file}")
    print(f"Report: {report_file}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")


def generate_report(data, report_file, args):
    env = data['env']
    cfg = data['config']
    layer_results = data['layer_results']

    lines = [
        f"# 实验：Phase 1c 逐层敏感度排名",
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
        f"python scripts/phase1c_layer_sensitivity.py --num_prompts {cfg['num_prompts']} "
        f"--output_dir {args.output_dir}",
        "```",
        "",
        "## 方法",
        "对每个层 l，跳过该层所有步的注意力计算（复用上一步输出），其他层正常。",
        "生成视频，计算 SSIM/PSNR 与基准的差异。",
        "",
        "## 敏感度排名 (从最敏感到最安全)",
        "",
        "| 排名 | 层 | SSIM | PSNR (dB) | 敏感度 |",
        "|------|-----|------|-----------|--------|",
    ]

    ranked = data['ranking']
    for rank, layer in enumerate(ranked):
        r = layer_results[str(layer)]
        sensitivity = 1.0 - r['ssim_mean']
        bar = "█" * int(sensitivity * 200)
        lines.append(
            f"| {rank+1} | {layer} | {r['ssim_mean']:.4f} | {r['psnr_mean']:.1f} | {bar} |"
        )

    lines.extend([
        "",
        "## 分类",
        f"- **关键层** (SSIM < 0.95): {data['critical_layers']}",
        f"- **中等层** (0.95-0.99): {data['moderate_layers']}",
        f"- **安全层** (SSIM ≥ 0.99): {data['safe_layers']}",
        f"- 安全层比例: {len(data['safe_layers'])}/{len(cfg['test_layers'])}",
        "",
        f"## 总耗时",
        f"{data['total_time_seconds']:.1f}s ({data['total_time_seconds']/60:.1f}min)",
    ])

    with open(report_file, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1c: Layer Sensitivity")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to test (default: all 30)")
    parser.add_argument("--output_dir", default="results/phase1c")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_experiment(args)
