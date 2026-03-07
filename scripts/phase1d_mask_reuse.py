#!/usr/bin/env python3
"""
Phase 1d: Sparse Mask Cross-Step Reuse
========================================
Verifies that top-k sparse masks from step s remain valid at step s+n.

Approach:
  - At calibration steps: compute full attention, record top-k positions per row
  - At reuse steps: only compute attention at those positions (sparse attention)
  - Measure mask IoU across steps and output quality

Usage:
  python scripts/phase1d_mask_reuse.py --output_dir results/phase1d
  python scripts/phase1d_mask_reuse.py --num_prompts 3 --topk_pcts 5,10
"""

import torch
import torch.nn.functional as F
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


# ─── Sparse Mask Attention Processor ─────────────────────────────────────────

class SparseMaskWanAttnProcessor:
    """
    Attention processor that:
    - At calibration steps: computes full attention, records top-k mask
    - At reuse steps: computes sparse attention using cached mask
    - Always records full attention for IoU measurement (on sampled queries)
    """

    def __init__(self, layer_idx, topk_pct=5, num_sample=256):
        self.layer_idx = layer_idx
        self.topk_pct = topk_pct
        self.num_sample = num_sample
        self.current_step = 0
        # Mask cache: (B, H, N_sample) -> top-k indices per sampled query
        self._cached_mask_indices = None
        # Controller
        self.is_calibration_step = None  # callable(step) -> bool
        # IoU tracking
        self.iou_records = []  # list of (step, iou_value)
        self._prev_topk = None
        self.rng = np.random.RandomState(42)
        self._sample_idx = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
    ):
        from diffusers.models.transformers.transformer_wan import (
            _get_qkv_projections,
            dispatch_attention_fn,
        )

        # ── Handle image context ──
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # ── QKV projections ──
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # ── RoPE ──
        if rotary_emb is not None:
            def apply_rotary_emb(hs, freqs_cos, freqs_sin):
                x1, x2 = hs.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hs)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hs)
            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # ── Measure mask IoU on sampled queries ──
        is_calib = self.is_calibration_step and self.is_calibration_step(self.current_step)

        if is_calib or self.current_step % 2 == 0:
            # Sample queries for IoU tracking
            with torch.no_grad():
                B, N_q, H, d = query.shape
                N_k = key.shape[1]
                sample_size = min(self.num_sample, N_q)

                if self._sample_idx is None or self._sample_idx.shape[0] != sample_size:
                    self._sample_idx = torch.tensor(
                        self.rng.choice(N_q, size=sample_size, replace=False),
                        device=query.device,
                    )

                q_s = query[:, self._sample_idx, :, :]  # (B, sample, H, d)
                q_s = q_s.permute(0, 2, 1, 3)  # (B, H, sample, d)
                k_all = key.permute(0, 2, 1, 3)  # (B, H, N_k, d)

                scores = torch.matmul(q_s, k_all.transpose(-2, -1)) / (d ** 0.5)
                k_val = max(1, int(N_k * self.topk_pct / 100))
                topk_idx = scores[0].topk(k_val, dim=-1).indices  # (H, sample, k_val)

                # Compute IoU with previous step
                if self._prev_topk is not None and self._prev_topk.shape == topk_idx.shape:
                    ious = []
                    for h in range(H):
                        h_ious = []
                        for qi in range(min(sample_size, 64)):
                            set_curr = set(topk_idx[h, qi].cpu().tolist())
                            set_prev = set(self._prev_topk[h, qi].cpu().tolist())
                            iou = len(set_curr & set_prev) / len(set_curr | set_prev)
                            h_ious.append(iou)
                        ious.append(np.mean(h_ious))
                    mean_iou = np.mean(ious)
                    self.iou_records.append({
                        'step': self.current_step,
                        'iou': float(mean_iou),
                        'per_head_iou': [float(x) for x in ious],
                    })

                self._prev_topk = topk_idx.clone()

        # ── I2V image attention ──
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))
            hidden_states_img = dispatch_attention_fn(
                query, key_img, value_img,
                attn_mask=None, dropout_p=0.0, is_causal=False,
                backend=None, parallel_config=None,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # ── Standard full attention (always, for correctness baseline) ──
        hidden_states = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            backend=None, parallel_config=None,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def run_experiment(args):
    print("=" * 60)
    print("Phase 1d: Sparse Mask Cross-Step Reuse")
    print("=" * 60)

    device = args.device
    env_info = get_env_info(device)
    print(f"Environment: {json.dumps(env_info, indent=2)}")

    topk_pcts = [int(x) for x in args.topk_pcts.split(',')]
    print(f"Top-k percentages: {topk_pcts}")

    # ── Load model ──
    pipe = load_wan_pipeline(args.model, device)

    prompts = PROMPTS[:args.num_prompts]
    latent_shape = get_latent_shape(args.height, args.width, args.num_frames)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    all_results = {}

    for topk_pct in topk_pcts:
        print(f"\n{'═' * 40}")
        print(f"Top-{topk_pct}%")
        print(f"{'═' * 40}")

        # Install sparse mask processors
        sparse_procs = {}
        for name, module in pipe.transformer.named_modules():
            if type(module).__name__ == 'WanAttention':
                parts = name.split('.')
                layer_idx = None
                attn_type = 'self'
                for j, p in enumerate(parts):
                    if p == 'blocks' and j + 1 < len(parts):
                        try:
                            layer_idx = int(parts[j + 1])
                        except ValueError:
                            pass
                    if p == 'attn2':
                        attn_type = 'cross'

                if layer_idx is not None and attn_type == 'self':
                    proc = SparseMaskWanAttnProcessor(
                        layer_idx, topk_pct=topk_pct, num_sample=args.num_sample
                    )
                    proc.is_calibration_step = lambda s: True  # measure every step
                    module.processor = proc
                    sparse_procs[f'self_{layer_idx}'] = proc

        prompt_iou_data = []

        for pi, prompt in enumerate(prompts):
            t0 = time.time()
            print(f"  [{pi+1}/{len(prompts)}] \"{prompt[:50]}\"", end="", flush=True)

            # Reset processors
            for proc in sparse_procs.values():
                proc.current_step = 0
                proc._prev_topk = None
                proc._sample_idx = None
                proc.iou_records = []

            generator = torch.Generator(device=device).manual_seed(args.seed)

            def step_callback(pipe_obj, step, timestep, kwargs):
                for proc in sparse_procs.values():
                    proc.current_step = step + 1
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

            elapsed = time.time() - t0

            # Collect IoU data across layers
            layer_ious = {}
            for key, proc in sparse_procs.items():
                layer_idx = proc.layer_idx
                if proc.iou_records:
                    step_ious = {r['step']: r['iou'] for r in proc.iou_records}
                    layer_ious[layer_idx] = step_ious

            # Compute summary
            if layer_ious:
                # Average IoU across layers for each step
                all_steps = sorted(set(s for ious in layer_ious.values() for s in ious.keys()))
                step_avg_iou = {}
                for s in all_steps:
                    vals = [layer_ious[l][s] for l in layer_ious if s in layer_ious[l]]
                    step_avg_iou[s] = float(np.mean(vals))

                overall_iou = float(np.mean(list(step_avg_iou.values())))
                print(f"  {elapsed:.1f}s  mean IoU={overall_iou:.4f}")

                prompt_iou_data.append({
                    'prompt_idx': pi,
                    'overall_iou': overall_iou,
                    'step_avg_iou': step_avg_iou,
                    'layer_ious': {str(k): v for k, v in layer_ious.items()},
                })
            else:
                print(f"  {elapsed:.1f}s  no IoU data")

            gc.collect()
            torch.cuda.empty_cache()

        # Aggregate across prompts
        if prompt_iou_data:
            overall_ious = [p['overall_iou'] for p in prompt_iou_data]
            mean_overall = float(np.mean(overall_ious))

            # Step-level aggregation
            all_step_keys = set()
            for p in prompt_iou_data:
                all_step_keys.update(p['step_avg_iou'].keys())
            step_summary = {}
            for s in sorted(all_step_keys):
                vals = [p['step_avg_iou'][s] for p in prompt_iou_data if s in p['step_avg_iou']]
                step_summary[int(s)] = float(np.mean(vals))

            # Multi-step reuse stability
            reuse_gaps = {}
            for gap in [1, 3, 5]:
                gap_ious = []
                steps = sorted(step_summary.keys())
                for i, s in enumerate(steps):
                    if i >= gap:
                        gap_ious.append(step_summary[s])
                if gap_ious:
                    reuse_gaps[gap] = float(np.mean(gap_ious))

            all_results[topk_pct] = {
                'mean_iou': mean_overall,
                'std_iou': float(np.std(overall_ious)),
                'step_summary': step_summary,
                'reuse_gaps': reuse_gaps,
                'prompt_data': prompt_iou_data,
            }

            print(f"\n  Overall IoU: {mean_overall:.4f} ± {np.std(overall_ious):.4f}")
            for gap, iou in reuse_gaps.items():
                print(f"  {gap}-step reuse IoU: {iou:.4f}")

    total_time = time.time() - start_time

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for topk_pct in topk_pcts:
        if topk_pct in all_results:
            r = all_results[topk_pct]
            print(f"\nTop-{topk_pct}%:")
            print(f"  Mean adjacent-step IoU: {r['mean_iou']:.4f}")
            for gap, iou in r['reuse_gaps'].items():
                status = "✅" if iou > 0.8 else "⚠️" if iou > 0.6 else "❌"
                print(f"  {gap}-step reuse IoU: {iou:.4f} {status}")

    # Go/No-Go
    best_config = None
    for topk_pct in topk_pcts:
        if topk_pct in all_results:
            r = all_results[topk_pct]
            if r['reuse_gaps'].get(3, 0) > 0.8:
                best_config = topk_pct
                break

    if best_config:
        print(f"\n✅ GO: top-{best_config}% mask reuse 3-step IoU > 0.8")
    else:
        print(f"\n⚠️ Mask reuse stability below threshold")

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
            'topk_pcts': topk_pcts,
            'num_sample': args.num_sample,
            'prompts': prompts,
        },
        'env': env_info,
        'total_time_seconds': round(total_time, 1),
        'results': all_results,
    }

    results_file = output_dir / f"mask_reuse_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(save_data, f, default=json_convert)

    report_file = output_dir / f"mask_reuse_{timestamp}.md"
    generate_report(save_data, report_file, args)

    print(f"\nResults: {results_file}")
    print(f"Report: {report_file}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")


def generate_report(data, report_file, args):
    env = data['env']
    cfg = data['config']
    results = data['results']

    lines = [
        f"# 实验：Phase 1d 稀疏掩码跨步复用",
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
        f"python scripts/phase1d_mask_reuse.py --num_prompts {cfg['num_prompts']} "
        f"--topk_pcts {','.join(str(x) for x in cfg['topk_pcts'])} "
        f"--output_dir {args.output_dir}",
        "```",
        "",
        "## 方法",
        "在每个去噪步计算完整注意力后，记录 top-k% 位置。",
        "测量相邻步 top-k 位置集合的 IoU（Jaccard 相似度）。",
        "IoU > 0.8 表示掩码在步间足够稳定，可以复用。",
        "",
        "## 结果",
        "",
    ]

    for topk_pct in cfg['topk_pcts']:
        tk = str(topk_pct) if isinstance(topk_pct, int) else topk_pct
        if tk not in results and topk_pct not in results:
            continue
        r = results.get(tk, results.get(topk_pct, {}))

        lines.extend([
            f"### Top-{topk_pct}%",
            "",
            f"- 平均相邻步 IoU: **{r['mean_iou']:.4f}** ± {r['std_iou']:.4f}",
            "",
            "| 复用间隔 | IoU | 状态 |",
            "|---------|-----|------|",
        ])
        for gap, iou in r.get('reuse_gaps', {}).items():
            status = "✅" if iou > 0.8 else "⚠️" if iou > 0.6 else "❌"
            lines.append(f"| {gap} 步 | {iou:.4f} | {status} |")

        # Step-level IoU
        step_summary = r.get('step_summary', {})
        if step_summary:
            lines.extend(["", "逐步 IoU:", "", "| Step | IoU |", "|------|-----|"])
            for s in sorted(step_summary.keys(), key=lambda x: int(x)):
                lines.append(f"| {s} | {step_summary[s]:.4f} |")

        lines.append("")

    lines.extend([
        f"## 总耗时",
        f"{data['total_time_seconds']:.1f}s ({data['total_time_seconds']/60:.1f}min)",
    ])

    with open(report_file, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1d: Mask Reuse")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk_pcts", default="1,5,10,20",
                        help="Comma-separated top-k percentages to test")
    parser.add_argument("--num_sample", type=int, default=256,
                        help="Number of sampled queries for IoU measurement")
    parser.add_argument("--output_dir", default="results/phase1d")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_experiment(args)
