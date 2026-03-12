#!/usr/bin/env python3
"""
AutoAccel Phase 0.75: Sparse Attention Baseline + Sparse×Cache Interaction
===========================================================================
Implements frame-local sparse attention for Wan 2.1 and tests:
  1. Sparse attention quality at different window sizes
  2. Sparse attention speedup
  3. Sparse × FBC interaction (the key unknown for AutoAccel)

Based on Phase 0 findings:
  - Block diagonal ratio = 9x (same-frame >> cross-frame)
  - Temporal locality: 78.5% attention on same frame
  - Top-1% covers 57.8% attention mass

Configurations:
  C0: Baseline (full 3D attention)
  C1: Window=0 (same-frame only, ~5x compute reduction)
  C2: Window=1 (±1 frame, covers ~95% attention)
  C3: Window=0 + FBC t=0.03 (sparse + cache combo)
  C4: Window=1 + FBC t=0.03 (sparse + cache combo)

Usage:
  python scripts/autoaccel/phase075_sparse.py --output_dir results/autoaccel_phase075
  python scripts/autoaccel/phase075_sparse.py --num_prompts 3 --configs C0,C1
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
import gc
import time
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phase1_utils import (
    PROMPTS, compute_ssim, compute_psnr,
    save_video_frames, get_env_info, json_convert,
)


# ─── Frame-Local Sparse Attention Processor ─────────────────────────────────

class SparseWanAttnProcessor:
    """
    Wan attention processor with frame-local sparse attention.

    Instead of full 3D attention over all video tokens,
    each query frame only attends to frames within a window.

    window=0: same frame only (intra-frame attention)
    window=1: ±1 neighboring frames
    window=None: full attention (baseline)
    """

    def __init__(self, layer_idx, attn_type='self', window=None,
                 num_frames_latent=5, spatial_tokens=6240, step_tracker=None):
        self.layer_idx = layer_idx
        self.attn_type = attn_type
        self.window = window  # None = full attention
        self.num_frames_latent = num_frames_latent
        self.spatial_tokens = spatial_tokens
        self.step_tracker = step_tracker  # shared dict: {"step": int, "start": int}

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

        # ── Handle image context (I2V) ──
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # ── QKV projections ──
        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        # ── Norm ──
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # ── Reshape to multi-head ──
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

        # ── I2V image attention ──
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            from diffusers.models.transformers.transformer_wan import _get_added_kv_projections
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

        # ── Attention (full or sparse) ──
        # Step-adaptive: only apply sparse after start step
        step_ok = True
        if self.step_tracker is not None:
            step_ok = self.step_tracker["step"] >= self.step_tracker["start"]
        if (self.attn_type == 'self' and self.window is not None and step_ok
                and query.shape[1] == self.num_frames_latent * self.spatial_tokens):
            hidden_states = self._frame_local_attention(query, key, value)
        else:
            # Full attention (cross-attention or baseline)
            hidden_states = dispatch_attention_fn(
                query, key, value,
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                backend=None, parallel_config=None,
            )
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        # ── Output projection ──
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def _frame_local_attention(self, query, key, value):
        """
        Frame-local attention: each query frame attends only to nearby frames.

        query/key/value: (B, N, H, D) where N = T * S
        Returns: (B, N, H*D)
        """
        B, N, H, D = query.shape
        T = self.num_frames_latent
        S = self.spatial_tokens
        W = self.window

        # Reshape to (B, T, S, H, D)
        q = query.reshape(B, T, S, H, D)
        k = key.reshape(B, T, S, H, D)
        v = value.reshape(B, T, S, H, D)

        outputs = []
        for t in range(T):
            # Determine which frames to attend to
            t_start = max(0, t - W)
            t_end = min(T, t + W + 1)

            # Query for this frame: (B, S, H, D)
            q_t = q[:, t]

            # Key/Value for window: (B, window_frames * S, H, D)
            k_window = k[:, t_start:t_end].reshape(B, -1, H, D)
            v_window = v[:, t_start:t_end].reshape(B, -1, H, D)

            # Compute attention for this frame
            # SDPA expects (B, H, S, D) format
            q_t_sdpa = q_t.transpose(1, 2)  # (B, H, S, D)
            k_w_sdpa = k_window.transpose(1, 2)  # (B, H, W*S, D)
            v_w_sdpa = v_window.transpose(1, 2)  # (B, H, W*S, D)

            out_t = F.scaled_dot_product_attention(
                q_t_sdpa, k_w_sdpa, v_w_sdpa,
                dropout_p=0.0, is_causal=False,
            )  # (B, H, S, D)

            out_t = out_t.transpose(1, 2)  # (B, S, H, D)
            outputs.append(out_t)

        # Reassemble: (B, T, S, H, D) -> (B, N, H*D)
        result = torch.stack(outputs, dim=1)  # (B, T, S, H, D)
        result = result.reshape(B, N, H * D)
        result = result.type_as(query)
        return result


# ─── Model Setup ─────────────────────────────────────────────────────────────

def load_wan_pipeline(model_name, device):
    """Load Wan pipeline."""
    from diffusers import AutoencoderKLWan, WanPipeline
    vae = AutoencoderKLWan.from_pretrained(
        model_name, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(
        model_name, vae=vae, torch_dtype=torch.bfloat16
    )
    pipe.to(device)
    return pipe


def install_sparse_processors(transformer, window=None, num_frames_latent=5,
                              height=480, width=832, sparse_layers=None,
                              step_tracker=None):
    """Replace self-attention processors with sparse versions.

    Args:
        sparse_layers: If None, apply to all layers. If a set/list of ints,
                       only apply sparse attention to those layer indices.
        step_tracker: If provided, a shared dict {"step": int, "start": int}.
                      Sparse attention only activates when step >= start.
    """
    # Wan VAE downsamples spatially by 16x (not 8x)
    spatial_tokens = (height // 16) * (width // 16)  # 30 * 52 = 1560
    processors = {}

    for name, module in transformer.named_modules():
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

            if layer_idx is not None:
                # Determine window for this layer
                if attn_type == 'self' and (sparse_layers is None or layer_idx in sparse_layers):
                    w = window
                else:
                    w = None
                proc = SparseWanAttnProcessor(
                    layer_idx, attn_type, window=w,
                    num_frames_latent=num_frames_latent,
                    spatial_tokens=spatial_tokens,
                    step_tracker=step_tracker,
                )
                module.processor = proc
                key = f"{attn_type}_{layer_idx}"
                processors[key] = proc

    sparse_count = sum(1 for k, p in processors.items()
                       if k.startswith('self') and p.window is not None)
    total_self = sum(1 for k in processors if k.startswith('self'))
    cross_count = sum(1 for k in processors if k.startswith('cross'))
    print(f"Installed {sparse_count}/{total_self} sparse self-attn + {cross_count} cross-attn "
          f"processors (window={window}, sparse_layers={sparse_layers})")
    return processors


def reset_processors(transformer):
    """Reset all processors to default."""
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor2_0
    for name, module in transformer.named_modules():
        if type(module).__name__ == 'WanAttention':
            module.processor = WanAttnProcessor2_0()


# ─── FBC helpers ─────────────────────────────────────────────────────────────

def enable_fbc(pipe, threshold):
    from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
    apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold=threshold))


def disable_fbc(pipe):
    _BLOCK_NAMES = ["blocks", "transformer_blocks", "single_transformer_blocks",
                    "temporal_transformer_blocks"]
    for attr_name in _BLOCK_NAMES:
        blocks = getattr(pipe.transformer, attr_name, None)
        if blocks is None or not isinstance(blocks, torch.nn.ModuleList):
            continue
        for block in blocks:
            if hasattr(block, '_diffusers_hook'):
                hook_registry = block._diffusers_hook
                for name in list(hook_registry.hooks.keys()):
                    hook_registry.remove_hook(name)
                del block._diffusers_hook


# ─── Video Generation ────────────────────────────────────────────────────────

def generate_video(pipe, prompt, height=480, width=832,
                   num_frames=17, num_steps=50, seed=42, device="cuda:0",
                   step_tracker=None):
    generator = torch.Generator(device=device).manual_seed(seed)

    # Step callback to update tracker
    callback = None
    if step_tracker is not None:
        step_tracker["step"] = 0
        def _step_cb(pipe_obj, step_idx, timestep, cb_kwargs):
            step_tracker["step"] = step_idx
            return cb_kwargs
        callback = _step_cb

    with torch.no_grad():
        output = pipe(
            prompt=prompt, height=height, width=width,
            num_frames=num_frames, num_inference_steps=num_steps,
            generator=generator, output_type="np",
            callback_on_step_end=callback,
        )
    frames = output.frames[0]
    frames_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    return frames_tensor


# ─── Experiment Runner ───────────────────────────────────────────────────────

CONFIG_SPECS = {
    'C0': {'window': None, 'fbc': None,  'layers': None,  'desc': 'Baseline (full 3D attention)'},
    'C1': {'window': 0,    'fbc': None,  'layers': None,  'desc': 'Window=0 ALL layers'},
    'C2': {'window': 1,    'fbc': None,  'layers': None,  'desc': 'Window=1 ALL layers'},
    'C3': {'window': 0,    'fbc': 0.03,  'layers': None,  'desc': 'Window=0 ALL + FBC 0.03'},
    'C4': {'window': 1,    'fbc': 0.03,  'layers': None,  'desc': 'Window=1 ALL + FBC 0.03'},
    'C5': {'window': 0,    'fbc': 0.05,  'layers': None,  'desc': 'Window=0 ALL + FBC 0.05'},
    'C6': {'window': 1,    'fbc': 0.05,  'layers': None,  'desc': 'Window=1 ALL + FBC 0.05'},
    # Selective sparse: only apply to last N layers (30 total)
    'S1': {'window': 0,    'fbc': None,  'layers': set(range(15, 30)),  'desc': 'Window=0 last-15 layers'},
    'S2': {'window': 0,    'fbc': None,  'layers': set(range(20, 30)),  'desc': 'Window=0 last-10 layers'},
    'S3': {'window': 0,    'fbc': None,  'layers': set(range(25, 30)),  'desc': 'Window=0 last-5 layers'},
    'S4': {'window': 1,    'fbc': None,  'layers': set(range(15, 30)),  'desc': 'Window=1 last-15 layers'},
    'S5': {'window': 1,    'fbc': None,  'layers': set(range(20, 30)),  'desc': 'Window=1 last-10 layers'},
    # Selective sparse + FBC combos
    'S6': {'window': 0,    'fbc': 0.03,  'layers': set(range(15, 30)),  'desc': 'W0 last-15 + FBC 0.03'},
    'S7': {'window': 0,    'fbc': 0.03,  'layers': set(range(20, 30)),  'desc': 'W0 last-10 + FBC 0.03'},
    # Step-adaptive sparse: only apply sparse after step N (out of 50)
    'T1': {'window': 0,  'fbc': None, 'layers': None, 'step_start': 25, 'desc': 'W=0 ALL, steps 25-49'},
    'T2': {'window': 0,  'fbc': None, 'layers': None, 'step_start': 30, 'desc': 'W=0 ALL, steps 30-49'},
    'T3': {'window': 0,  'fbc': None, 'layers': None, 'step_start': 35, 'desc': 'W=0 ALL, steps 35-49'},
    'T4': {'window': 0,  'fbc': None, 'layers': None, 'step_start': 40, 'desc': 'W=0 ALL, steps 40-49'},
    'T5': {'window': 0,  'fbc': None, 'layers': None, 'step_start': 45, 'desc': 'W=0 ALL, steps 45-49'},
    # Step-adaptive + layer-selective combos
    'T6': {'window': 0,  'fbc': None, 'layers': set(range(10, 30)), 'step_start': 25, 'desc': 'W=0 L10-29, steps 25-49'},
    'T7': {'window': 0,  'fbc': None, 'layers': set(range(10, 30)), 'step_start': 35, 'desc': 'W=0 L10-29, steps 35-49'},
    # Best step-adaptive + FBC (will pick based on T1-T5 results)
    'T8': {'window': 0,  'fbc': 0.03, 'layers': None, 'step_start': 25, 'desc': 'W=0 steps 25-49 + FBC 0.03'},
    'T9': {'window': 0,  'fbc': 0.03, 'layers': None, 'step_start': 35, 'desc': 'W=0 steps 35-49 + FBC 0.03'},
}


def compute_sparsity(window, num_frames):
    """Compute theoretical attention sparsity for frame-local attention."""
    if window is None:
        return 0.0
    total_pairs = num_frames * num_frames
    active_pairs = 0
    for t in range(num_frames):
        t_start = max(0, t - window)
        t_end = min(num_frames, t + window + 1)
        active_pairs += (t_end - t_start)
    return 1.0 - active_pairs / total_pairs


def run_experiment(args):
    print("=" * 60)
    print("AutoAccel Phase 0.75: Sparse Attention + Interaction Test")
    print("=" * 60)

    device = args.device
    env_info = get_env_info(device)
    print(f"Environment: {json.dumps(env_info, indent=2)}")

    configs = args.configs.split(',')
    if 'C0' not in configs:
        configs.insert(0, 'C0')

    prompts = PROMPTS[:args.num_prompts]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute latent dimensions
    num_frames_latent = (args.num_frames - 1) // 4 + 1  # 17 -> 5
    spatial_tokens = (args.height // 16) * (args.width // 16)  # 30*52=1560
    total_tokens = num_frames_latent * spatial_tokens

    print(f"\nLatent: {num_frames_latent} frames × {spatial_tokens} spatial = {total_tokens} tokens")

    for cfg_name in configs:
        spec = CONFIG_SPECS[cfg_name]
        sparsity = compute_sparsity(spec['window'], num_frames_latent)
        print(f"  {cfg_name}: {spec['desc']} (sparsity={sparsity:.1%})")

    # Load model
    print(f"\nLoading model: {args.model}")
    pipe = load_wan_pipeline(args.model, device)

    all_results = {}
    start_time = time.time()

    for cfg_name in configs:
        spec = CONFIG_SPECS[cfg_name]
        print(f"\n{'─' * 50}")
        print(f"Config: {cfg_name} — {spec['desc']}")
        print(f"{'─' * 50}")

        # Setup step tracker for step-adaptive sparse
        step_start = spec.get('step_start')
        step_tracker = None
        if step_start is not None:
            step_tracker = {"step": 0, "start": step_start}
            print(f"  Step-adaptive: sparse only at steps >= {step_start}")

        # Setup sparse attention
        reset_processors(pipe.transformer)
        if spec['window'] is not None:
            install_sparse_processors(
                pipe.transformer, window=spec['window'],
                num_frames_latent=num_frames_latent,
                height=args.height, width=args.width,
                sparse_layers=spec.get('layers'),
                step_tracker=step_tracker,
            )

        # Setup FBC
        if spec['fbc'] is not None:
            enable_fbc(pipe, spec['fbc'])
            print(f"  FBC: enabled (threshold={spec['fbc']})")

        # Warmup
        if args.warmup:
            print(f"  Warmup...", end="", flush=True)
            _ = generate_video(pipe, "warmup test", height=args.height,
                             width=args.width, num_frames=args.num_frames,
                             num_steps=args.num_steps, seed=args.seed, device=device,
                             step_tracker=step_tracker)
            torch.cuda.synchronize()
            gc.collect(); torch.cuda.empty_cache()
            print(" done")

        results = {'times': [], 'frames': []}

        for pi, prompt in enumerate(prompts):
            print(f"  [{pi+1}/{len(prompts)}] \"{prompt[:50]}\"", end="", flush=True)
            torch.cuda.synchronize()
            t0 = time.time()

            frames = generate_video(pipe, prompt, height=args.height,
                                  width=args.width, num_frames=args.num_frames,
                                  num_steps=args.num_steps, seed=args.seed, device=device,
                                  step_tracker=step_tracker)

            torch.cuda.synchronize()
            gen_time = time.time() - t0
            results['times'].append(gen_time)
            results['frames'].append(frames)
            print(f"  {gen_time:.1f}s")

            gc.collect(); torch.cuda.empty_cache()

        # Teardown FBC
        if spec['fbc'] is not None:
            disable_fbc(pipe)

        results['time_mean'] = float(np.mean(results['times']))
        results['time_std'] = float(np.std(results['times']))
        results['description'] = spec['desc']
        results['window'] = spec['window']
        results['fbc_threshold'] = spec['fbc']
        results['sparse_layers'] = sorted(spec.get('layers')) if spec.get('layers') else None
        results['num_sparse_layers'] = len(spec['layers']) if spec.get('layers') else (30 if spec['window'] is not None else 0)
        results['step_start'] = spec.get('step_start')
        results['sparsity'] = compute_sparsity(spec['window'], num_frames_latent)
        all_results[cfg_name] = results

    total_time = time.time() - start_time

    # Reset processors
    reset_processors(pipe.transformer)

    # ── Quality metrics ──
    baseline_frames = all_results['C0']['frames']
    for key, results in all_results.items():
        if key == 'C0':
            results['ssim'] = [1.0] * len(prompts)
            results['psnr'] = [100.0] * len(prompts)
            results['ssim_mean'] = 1.0
            results['ssim_std'] = 0.0
            results['psnr_mean'] = 100.0
            results['psnr_std'] = 0.0
        else:
            ssims, psnrs = [], []
            for pi in range(len(prompts)):
                ssims.append(compute_ssim(results['frames'][pi], baseline_frames[pi]))
                psnrs.append(compute_psnr(results['frames'][pi], baseline_frames[pi]))
            results['ssim'] = ssims
            results['psnr'] = psnrs
            results['ssim_mean'] = float(np.mean(ssims))
            results['ssim_std'] = float(np.std(ssims))
            results['psnr_mean'] = float(np.mean(psnrs))
            results['psnr_std'] = float(np.std(psnrs))

    # ── Speedups ──
    baseline_time = all_results['C0']['time_mean']
    for key, results in all_results.items():
        results['speedup'] = baseline_time / results['time_mean']

    # ── Summary ──
    print(f"\n{'=' * 75}")
    print("SUMMARY")
    print(f"{'=' * 75}")
    print(f"{'Config':<8} | {'Desc':<30} | {'Time':>7} | {'Speed':>6} | "
          f"{'SSIM':>10} | {'Sparsity':>8}")
    print("-" * 80)

    for key in all_results:
        r = all_results[key]
        print(f"{key:<8} | {r['description'][:30]:<30} | "
              f"{r['time_mean']:5.1f}±{r['time_std']:.1f} | "
              f"{r['speedup']:5.2f}x | "
              f"{r['ssim_mean']:.4f}±{r['ssim_std']:.3f} | "
              f"{r['sparsity']:6.1%}")

    # ── Interaction analysis ──
    print(f"\n{'=' * 75}")
    print("INTERACTION ANALYSIS (Sparse × Cache)")
    print(f"{'=' * 75}")

    # Check C1+FBC vs C1 alone and FBC alone
    for sparse_cfg, combo_cfg, fbc_cfg in [('C1', 'C3', 'B2_t0.03'), ('C2', 'C4', 'B2_t0.03'),
                                            ('C1', 'C5', 'B2_t0.05'), ('C2', 'C6', 'B2_t0.05')]:
        if sparse_cfg not in all_results or combo_cfg not in all_results:
            continue
        sparse_r = all_results[sparse_cfg]
        combo_r = all_results[combo_cfg]
        fbc_thresh = CONFIG_SPECS[combo_cfg]['fbc']

        # We need FBC-only data from Phase 0.5
        # Use t=0.03 → 1.26x, t=0.05 → 1.62x
        fbc_speedups = {0.03: 1.26, 0.05: 1.62}
        fbc_speedup = fbc_speedups.get(fbc_thresh, 1.0)

        expected = sparse_r['speedup'] * fbc_speedup
        actual = combo_r['speedup']
        ratio = actual / expected if expected > 0 else 0

        print(f"\n  {sparse_cfg} ({sparse_r['description'][:20]})")
        print(f"    Sparse alone:     {sparse_r['speedup']:.2f}x  (SSIM={sparse_r['ssim_mean']:.4f})")
        print(f"    FBC t={fbc_thresh} alone:  {fbc_speedup:.2f}x  (from Phase 0.5)")
        print(f"    Expected combo:   {expected:.2f}x")
        print(f"    Actual combo:     {actual:.2f}x  (SSIM={combo_r['ssim_mean']:.4f})")
        print(f"    Interaction:      {ratio:.2f}")

        if ratio > 0.9:
            print(f"    → ORTHOGONAL")
        elif ratio > 0.7:
            print(f"    → PARTIAL INTERFERENCE")
        else:
            print(f"    → SIGNIFICANT INTERFERENCE")

    # ── Save results ──
    save_results = {}
    for key, r in all_results.items():
        save_results[key] = {k: v for k, v in r.items() if k != 'frames'}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_data = {
        'config': {
            'model': args.model,
            'height': args.height,
            'width': args.width,
            'num_frames': args.num_frames,
            'num_frames_latent': num_frames_latent,
            'spatial_tokens': spatial_tokens,
            'total_tokens': total_tokens,
            'num_steps': args.num_steps,
            'num_prompts': args.num_prompts,
            'seed': args.seed,
            'configs_tested': configs,
            'prompts': prompts,
            'warmup': args.warmup,
        },
        'env': env_info,
        'total_time_seconds': round(total_time, 1),
        'results': save_results,
    }

    results_file = output_dir / f"phase075_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(save_data, f, default=json_convert, indent=2)

    print(f"\nResults: {results_file}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoAccel Phase 0.75: Sparse Attention + Interaction")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--num_prompts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", default="C0,C1,C2,C3,C4,C5,C6",
                        help="Configs to test")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--output_dir", default="results/autoaccel_phase075")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_experiment(args)
