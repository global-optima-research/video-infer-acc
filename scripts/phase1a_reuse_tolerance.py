#!/usr/bin/env python3
"""
Phase 1a: Step-wise Reuse Tolerance
====================================
Measures ||O_s - O_{s-1}|| / ||O_s|| for each (step, layer) pair.
Produces a (step × layer) heatmap showing which combinations can safely skip.

Usage:
  python scripts/phase1a_reuse_tolerance.py --output_dir results/phase1a
  python scripts/phase1a_reuse_tolerance.py --num_prompts 3 --output_dir results/phase1a_quick
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
    set_all_steps, clear_all_caches, get_env_info, get_latent_shape, json_convert,
)


def run_experiment(args):
    print("=" * 60)
    print("Phase 1a: Step-wise Reuse Tolerance")
    print("=" * 60)

    device = args.device
    env_info = get_env_info(device)
    print(f"Environment: {json.dumps(env_info, indent=2)}")

    # ── Load model ──
    pipe = load_wan_pipeline(args.model, device)
    processors = install_skippable_processors(pipe.transformer)

    latent_shape = get_latent_shape(args.height, args.width, args.num_frames)
    print(f"Latent shape: {latent_shape}, tokens: {np.prod(latent_shape)}")

    # Enable output capture on all self-attention processors
    self_procs = {k: v for k, v in processors.items() if k.startswith('self_')}
    for proc in self_procs.values():
        proc.capture_output = True

    num_layers = len(self_procs)
    prompts = PROMPTS[:args.num_prompts]

    # Storage: (prompt, step, layer) -> relative change
    all_changes = np.zeros((len(prompts), args.num_steps, num_layers))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for pi, prompt in enumerate(prompts):
        t0 = time.time()
        print(f"\n[{pi+1}/{len(prompts)}] \"{prompt[:60]}\"")

        clear_all_caches(processors)
        for proc in self_procs.values():
            proc.captured_outputs = {}

        generator = torch.Generator(device=device).manual_seed(args.seed)

        def step_callback(pipe_obj, step, timestep, kwargs):
            set_all_steps(processors, step + 1)
            return kwargs

        with torch.no_grad():
            _ = pipe(
                prompt=prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_steps,
                generator=generator,
                callback_on_step_end=step_callback,
                output_type="latent",
            )

        # Compute step-to-step change for each layer
        for layer_idx in range(num_layers):
            proc = self_procs[f'self_{layer_idx}']
            steps_captured = sorted(proc.captured_outputs.keys())

            for si in range(1, len(steps_captured)):
                s_curr = steps_captured[si]
                s_prev = steps_captured[si - 1]
                o_curr = proc.captured_outputs[s_curr].float()
                o_prev = proc.captured_outputs[s_prev].float()

                norm_curr = torch.norm(o_curr).item()
                if norm_curr > 1e-10:
                    rel_change = torch.norm(o_curr - o_prev).item() / norm_curr
                else:
                    rel_change = 0.0

                all_changes[pi, s_curr, layer_idx] = rel_change

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        # Free captured outputs
        for proc in self_procs.values():
            proc.captured_outputs = {}
        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time

    # ── Aggregate across prompts ──
    mean_changes = all_changes.mean(axis=0)  # (steps, layers)
    std_changes = all_changes.std(axis=0)

    # ── Analysis ──
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Step-level averages
    step_avg = mean_changes.mean(axis=1)  # (steps,)
    print("\nStep-level average change rate:")
    for s in range(0, args.num_steps, 5):
        print(f"  Step {s:3d}: {step_avg[s]:.4f}")

    # Identify safe skip zones
    threshold = args.threshold
    safe_mask = mean_changes < threshold
    safe_pct = safe_mask.sum() / safe_mask.size * 100
    print(f"\nSafe to skip (change < {threshold}): {safe_pct:.1f}% of (step, layer) pairs")

    # Per-phase analysis
    phase_boundaries = [
        (0, args.num_steps // 4, "Early (0-25%)"),
        (args.num_steps // 4, args.num_steps // 2, "Mid-early (25-50%)"),
        (args.num_steps // 2, 3 * args.num_steps // 4, "Mid-late (50-75%)"),
        (3 * args.num_steps // 4, args.num_steps, "Late (75-100%)"),
    ]
    print("\nPer-phase safe skip rates:")
    for s_start, s_end, name in phase_boundaries:
        phase_safe = safe_mask[s_start:s_end].sum() / safe_mask[s_start:s_end].size * 100
        phase_mean = mean_changes[s_start:s_end].mean()
        print(f"  {name}: {phase_safe:.1f}% safe, mean change {phase_mean:.4f}")

    # Layer-level averages
    layer_avg = mean_changes.mean(axis=0)  # (layers,)
    top5_stable = np.argsort(layer_avg)[:5]
    top5_volatile = np.argsort(layer_avg)[-5:][::-1]
    print(f"\nMost stable layers: {top5_stable.tolist()}")
    print(f"Most volatile layers: {top5_volatile.tolist()}")

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    results = {
        'config': {
            'model': args.model,
            'height': args.height,
            'width': args.width,
            'num_frames': args.num_frames,
            'num_steps': args.num_steps,
            'num_prompts': args.num_prompts,
            'seed': args.seed,
            'threshold': threshold,
            'prompts': prompts,
        },
        'env': env_info,
        'latent_shape': list(latent_shape),
        'total_time_seconds': round(total_time, 1),
        'mean_changes': mean_changes.tolist(),  # (steps, layers)
        'std_changes': std_changes.tolist(),
        'step_avg': step_avg.tolist(),
        'layer_avg': layer_avg.tolist(),
        'safe_pct': safe_pct,
        'top5_stable_layers': top5_stable.tolist(),
        'top5_volatile_layers': top5_volatile.tolist(),
    }

    results_file = output_dir / f"reuse_tolerance_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, default=json_convert)
    print(f"\nResults saved to {results_file}")

    # ── Generate heatmap as ASCII (for report) ──
    report_file = output_dir / f"reuse_tolerance_{timestamp}.md"
    generate_report(results, report_file, args)
    print(f"Report saved to {report_file}")

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")


def generate_report(results, report_file, args):
    """Generate experiment report markdown."""
    env = results['env']
    cfg = results['config']
    mean_changes = np.array(results['mean_changes'])

    lines = [
        f"# 实验：Phase 1a 逐步复用容忍度",
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
        f"python scripts/phase1a_reuse_tolerance.py --num_prompts {cfg['num_prompts']} "
        f"--num_steps {cfg['num_steps']} --threshold {cfg['threshold']} "
        f"--output_dir {args.output_dir}",
        "```",
        "",
        "## 参数",
        f"| 参数 | 值 |",
        f"|------|-----|",
        f"| 模型 | {cfg['model']} |",
        f"| 分辨率 | {cfg['height']}×{cfg['width']} |",
        f"| 帧数 | {cfg['num_frames']} |",
        f"| 去噪步数 | {cfg['num_steps']} |",
        f"| Prompt 数 | {cfg['num_prompts']} |",
        f"| 阈值 | {cfg['threshold']} |",
        f"| Seed | {cfg['seed']} |",
        "",
        "## 结果",
        "",
        "### 整体统计",
        f"- 安全跳过比例 (change < {cfg['threshold']}): **{results['safe_pct']:.1f}%**",
        f"- 最稳定层: {results['top5_stable_layers']}",
        f"- 最不稳定层: {results['top5_volatile_layers']}",
        "",
        "### 按阶段统计",
        "",
        "| 阶段 | 步范围 | 平均变化率 | 安全跳过比例 |",
        "|------|--------|-----------|------------|",
    ]

    ns = cfg['num_steps']
    phases = [
        (0, ns // 4, "Early (0-25%)"),
        (ns // 4, ns // 2, "Mid-early (25-50%)"),
        (ns // 2, 3 * ns // 4, "Mid-late (50-75%)"),
        (3 * ns // 4, ns, "Late (75-100%)"),
    ]
    threshold = cfg['threshold']
    for s_start, s_end, name in phases:
        phase_data = mean_changes[s_start:s_end]
        phase_mean = phase_data.mean()
        phase_safe = (phase_data < threshold).sum() / phase_data.size * 100
        lines.append(f"| {name} | {s_start}-{s_end} | {phase_mean:.4f} | {phase_safe:.1f}% |")

    lines.extend([
        "",
        "### 逐步变化率 (每层平均)",
        "",
        "| Step | 变化率 |",
        "|------|--------|",
    ])
    step_avg = results['step_avg']
    for s in range(len(step_avg)):
        bar = "█" * int(step_avg[s] * 200)
        lines.append(f"| {s:3d} | {step_avg[s]:.4f} {bar} |")

    lines.extend([
        "",
        "### 逐层变化率 (每步平均)",
        "",
        "| Layer | 变化率 |",
        "|-------|--------|",
    ])
    layer_avg = results['layer_avg']
    for l in range(len(layer_avg)):
        bar = "█" * int(layer_avg[l] * 200)
        lines.append(f"| {l:3d} | {layer_avg[l]:.4f} {bar} |")

    lines.extend([
        "",
        f"## 总耗时",
        f"{results['total_time_seconds']:.1f}s ({results['total_time_seconds']/60:.1f}min)",
    ])

    with open(report_file, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1a: Reuse Tolerance Heatmap")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Relative change threshold for 'safe to skip'")
    parser.add_argument("--output_dir", default="results/phase1a")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_experiment(args)
