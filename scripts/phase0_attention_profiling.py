#!/usr/bin/env python3
"""
Phase 0: Video DiT Attention Profiling
=======================================
Profiles attention patterns in Wan 2.1-T2V-1.3B to verify 7 structural properties.

Properties measured:
  ① Block diagonal dominance (same-frame vs cross-frame attention mass)
  ② Temporal locality decay (attention mass vs frame distance)
  ③ Cross-prompt invariance (top-k support set IoU across prompts)
  ④ Step-to-step stability (attention change across denoising steps)
  ⑤ Head specialization (spatial vs temporal vs global vs sink classification)
  ⑦ Entropy distribution (per-head attention entropy)

Usage:
  python scripts/phase0_attention_profiling.py --output_dir results/phase0
  python scripts/phase0_attention_profiling.py --discover_only  # print architecture only
"""

import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict
from copy import deepcopy
import gc
import time
import sys
import subprocess


# ─── Prompts ───────────────────────────────────────────────────────────────────

PROMPTS = [
    "A cat sitting on a windowsill watching birds outside",
    "Ocean waves crashing on a sandy beach at sunset",
    "A busy city street with cars and pedestrians",
    "A person walking through a snowy forest",
    "Colorful fireworks exploding in the night sky",
    "A butterfly landing on a flower in a garden",
    "Rain falling on a quiet lake surrounded by mountains",
    "A dog running across a green field chasing a ball",
    "Steam rising from a cup of hot coffee on a table",
    "Stars twinkling in a clear night sky over a desert",
]


# ─── Profiling Attention Processor ─────────────────────────────────────────────

class ProfilingWanAttnProcessor:
    """
    Wraps WanAttnProcessor to capture Q, K after norm+RoPE and compute
    sampled attention statistics without affecting model output.
    """

    def __init__(self, collector, layer_idx, attn_type):
        """
        Args:
            collector: AttentionStatsCollector instance
            layer_idx: int, which transformer block (0-29)
            attn_type: 'self' or 'cross'
        """
        self.collector = collector
        self.layer_idx = layer_idx
        self.attn_type = attn_type

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
        query = query.unflatten(2, (attn.heads, -1))  # (B, N, H, d)
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # ── RoPE ──
        if rotary_emb is not None:
            def apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # ━━━ PROFILING: Compute sampled attention stats ━━━
        if self.collector.should_profile(self.layer_idx, self.attn_type):
            with torch.no_grad():
                self.collector.compute_stats(
                    query, key, self.layer_idx, self.attn_type
                )

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

        # ── Standard attention (FlashAttention / SDPA) ──
        hidden_states = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            backend=None,
            parallel_config=None,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        # ── Output projection ──
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


# ─── Statistics Collector ──────────────────────────────────────────────────────

class AttentionStatsCollector:
    """Collects and aggregates attention statistics across steps and prompts."""

    def __init__(self, num_sample_queries=512, seed=42, latent_shape=None,
                 profile_layers=None, profile_steps=None):
        """
        Args:
            num_sample_queries: number of query tokens to sample per layer
            seed: random seed for reproducible sampling
            latent_shape: (T, H, W) of video latent
            profile_layers: list of layer indices to profile (None = all)
            profile_steps: list of step indices to profile (None = all)
        """
        self.num_queries = num_sample_queries
        self.rng = np.random.RandomState(seed)
        self.latent_shape = latent_shape
        self.profile_layers = profile_layers
        self.profile_steps = profile_steps

        # State
        self.current_step = 0
        self.current_prompt_idx = 0

        # Storage: prompt -> step -> layer -> stats
        self.all_stats = {}

        # For property ④: store previous step's attention hash
        self._prev_attn_hash = {}  # layer_idx -> hash tensor

    def should_profile(self, layer_idx, attn_type):
        """Decide whether to profile this layer at this step."""
        if attn_type != 'self':
            return False  # Only profile self-attention for now
        if self.profile_layers is not None and layer_idx not in self.profile_layers:
            return False
        if self.profile_steps is not None and self.current_step not in self.profile_steps:
            return False
        return True

    def compute_stats(self, query, key, layer_idx, attn_type):
        """
        Compute attention statistics for sampled queries.

        Args:
            query: (B, N_q, H, d_head) - after norm + RoPE
            key:   (B, N_k, H, d_head) - after norm + RoPE
            layer_idx: int
            attn_type: 'self' or 'cross'
        """
        B, N_q, H, d = query.shape
        N_k = key.shape[1]

        # Sample query indices
        sample_size = min(self.num_queries, N_q)
        sample_idx = torch.tensor(
            self.rng.choice(N_q, size=sample_size, replace=False),
            device=query.device,
        )

        # Permute to (B, H, N, d) for matmul
        q = query.permute(0, 2, 1, 3)     # (B, H, N_q, d)
        k = key.permute(0, 2, 1, 3)       # (B, H, N_k, d)

        q_s = q[:, :, sample_idx, :]      # (B, H, sample, d)

        # Compute attention weights for sampled queries
        scores = torch.matmul(q_s, k.transpose(-2, -1)) / (d ** 0.5)  # (B, H, sample, N_k)
        aw = F.softmax(scores.float(), dim=-1)  # float32 for precision

        # Use first batch element
        aw = aw[0]  # (H, sample, N_k)

        stats = {
            'num_tokens_q': N_q,
            'num_tokens_k': N_k,
            'num_heads': H,
            'd_head': d,
            'sample_size': sample_size,
        }

        # ── Property ⑦: Entropy ──
        entropy = -(aw * (aw + 1e-10).log()).sum(dim=-1).mean(dim=-1)  # (H,)
        max_entropy = float(np.log(N_k))
        stats['entropy_normalized'] = (entropy / max_entropy).cpu().tolist()
        stats['max_entropy'] = max_entropy

        # ── Property ①: Block diagonal dominance ──
        if self.latent_shape is not None and attn_type == 'self':
            T, Lh, Lw = self.latent_shape
            n_spatial = Lh * Lw

            query_frames = (sample_idx // n_spatial).cpu()   # (sample,)
            all_frames = torch.arange(N_k, device=aw.device) // n_spatial  # (N_k,)

            # Mean attention to same-frame vs different-frame tokens
            same_frame_mass = torch.zeros(H, device=aw.device)
            diff_frame_mass = torch.zeros(H, device=aw.device)
            for qi in range(sample_size):
                qf = query_frames[qi].item()
                same_mask = (all_frames == qf).float()
                diff_mask = 1.0 - same_mask
                # Normalize by number of tokens in each group
                n_same = same_mask.sum()
                n_diff = diff_mask.sum()
                same_frame_mass += (aw[:, qi, :] * same_mask).sum(dim=-1) / n_same if n_same > 0 else 0
                diff_frame_mass += (aw[:, qi, :] * diff_mask).sum(dim=-1) / n_diff if n_diff > 0 else 0

            same_frame_mass /= sample_size
            diff_frame_mass /= sample_size

            ratio = (same_frame_mass / (diff_frame_mass + 1e-10))
            stats['block_diag_ratio'] = ratio.cpu().tolist()  # per head
            stats['same_frame_mean_attn'] = same_frame_mass.cpu().tolist()
            stats['diff_frame_mean_attn'] = diff_frame_mass.cpu().tolist()

            # ── Property ②: Temporal decay ──
            temporal_decay = {}
            for dist in range(min(T, 8)):  # up to 8 frames distance
                dist_mask = torch.zeros(sample_size, N_k, device=aw.device)
                for qi in range(sample_size):
                    qf = query_frames[qi].item()
                    dist_mask[qi] = (torch.abs(all_frames - qf) == dist).float()

                mass = (aw * dist_mask.unsqueeze(0)).sum(dim=-1).mean(dim=-1)  # (H,)
                temporal_decay[str(dist)] = mass.cpu().tolist()
            stats['temporal_decay'] = temporal_decay

            # ── Property ⑤: Head specialization ──
            # Temporal specificity: attention to same spatial position across frames
            spatial_positions = torch.arange(N_k, device=aw.device) % n_spatial
            query_spatial = (sample_idx % n_spatial).to(aw.device)

            temporal_spec = torch.zeros(H, device=aw.device)
            count = 0
            for qi in range(min(sample_size, 200)):  # limit for speed
                qf = query_frames[qi].item()
                qs = query_spatial[qi].item()
                mask = ((spatial_positions == qs) & (all_frames != qf)).float()
                if mask.sum() > 0:
                    temporal_spec += (aw[:, qi, :] * mask).sum(dim=-1) / mask.sum()
                    count += 1

            if count > 0:
                temporal_spec /= count
            stats['temporal_specificity'] = temporal_spec.cpu().tolist()

        # ── Attention concentration (top-k mass) ──
        for pct in [1, 5, 10, 20]:
            k_val = max(1, int(N_k * pct / 100))
            topk_mass = aw.topk(k_val, dim=-1).values.sum(dim=-1).mean(dim=-1)  # (H,)
            stats[f'top{pct}pct_mass'] = topk_mass.cpu().tolist()

        # ── Property ③: Top-k indices for cross-prompt IoU ──
        k_iou = min(500, sample_size * 10)
        aw_flat = aw.reshape(H, -1)  # (H, sample*N_k)
        topk_idx = aw_flat.topk(k_iou, dim=-1).indices  # (H, k_iou)
        stats['topk_indices_for_iou'] = topk_idx.cpu().tolist()

        # ── Property ④: Step-to-step change ──
        attn_summary = aw.mean(dim=1)  # (H, N_k) - mean across sampled queries
        attn_hash = attn_summary.cpu()

        prev = self._prev_attn_hash.get(layer_idx)
        if prev is not None:
            # Cosine similarity between steps
            cos_sim = F.cosine_similarity(
                prev.flatten().unsqueeze(0).float(),
                attn_hash.flatten().unsqueeze(0).float(),
            ).item()
            # Relative L2 change
            rel_change = torch.norm(attn_hash - prev).item() / (torch.norm(prev).item() + 1e-10)
            stats['step_cosine_sim'] = cos_sim
            stats['step_rel_change'] = rel_change

        self._prev_attn_hash[layer_idx] = attn_hash

        # ── Store ──
        p_key = str(self.current_prompt_idx)
        s_key = str(self.current_step)
        l_key = str(layer_idx)

        if p_key not in self.all_stats:
            self.all_stats[p_key] = {}
        if s_key not in self.all_stats[p_key]:
            self.all_stats[p_key][s_key] = {}
        self.all_stats[p_key][s_key][l_key] = stats

    def on_step_end(self, step):
        """Called after each denoising step."""
        self.current_step = step + 1

    def on_prompt_start(self, prompt_idx):
        """Called before each prompt."""
        self.current_prompt_idx = prompt_idx
        self.current_step = 0
        self._prev_attn_hash = {}


# ─── Utilities ─────────────────────────────────────────────────────────────────

def get_latent_shape(height, width, num_frames):
    """Compute latent shape for Wan 2.1 (4x temporal, 8x spatial compression)."""
    t = (num_frames - 1) // 4 + 1
    h = height // 8
    w = width // 8
    return (t, h, w)


def get_env_info(device):
    """Collect environment information for reproducibility."""
    info = {
        'python_version': sys.version.split()[0],
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'N/A',
        'gpu_memory_gb': round(torch.cuda.get_device_properties(device).total_mem / 1e9, 1)
            if torch.cuda.is_available() else 0,
    }
    try:
        info['diffusers_version'] = __import__('diffusers').__version__
    except Exception:
        info['diffusers_version'] = 'unknown'
    try:
        info['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True
        ).strip()
    except Exception:
        info['git_commit'] = 'unknown'
    return info


def install_profiling_processors(transformer, collector):
    """Replace all WanAttnProcessors with ProfilingWanAttnProcessors."""
    count = 0
    for name, module in transformer.named_modules():
        if type(module).__name__ == 'WanAttention':
            # Determine layer index and attn type from name
            # Expected: blocks.{i}.attn1 or blocks.{i}.attn2
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
                module.processor = ProfilingWanAttnProcessor(
                    collector, layer_idx, attn_type
                )
                count += 1

    print(f"Installed {count} profiling processors")
    return count


def discover_architecture(transformer):
    """Print model architecture."""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    for name, module in transformer.named_modules():
        cls = type(module).__name__
        if any(k in cls.lower() for k in ['attention', 'attn', 'block']):
            extra = ""
            if hasattr(module, 'heads'):
                extra += f" heads={module.heads}"
            if hasattr(module, 'processor'):
                extra += f" proc={type(module.processor).__name__}"
            print(f"  {name}: {cls}{extra}")
    print("=" * 60 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 0: Attention Profiling for Wan 2.1-T2V-1.3B"
    )
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--sample_queries", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/phase0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--discover_only", action="store_true",
                        help="Only print model architecture, don't profile")
    parser.add_argument("--profile_layers", type=str, default=None,
                        help="Comma-separated layer indices to profile (default: all)")
    parser.add_argument("--profile_steps", type=str, default=None,
                        help="Comma-separated step indices to profile (default: all)")
    args = parser.parse_args()

    # Parse layer/step filters
    profile_layers = None
    if args.profile_layers:
        profile_layers = [int(x) for x in args.profile_layers.split(',')]

    profile_steps = None
    if args.profile_steps:
        profile_steps = [int(x) for x in args.profile_steps.split(',')]

    # ── Load model ──
    print(f"Loading model: {args.model}")
    from diffusers import AutoencoderKLWan, WanPipeline

    vae = AutoencoderKLWan.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(
        args.model, vae=vae, torch_dtype=torch.bfloat16
    )
    pipe.to(args.device)

    transformer = pipe.transformer

    # ── Architecture discovery ──
    discover_architecture(transformer)

    if args.discover_only:
        return

    # ── Setup ──
    latent_shape = get_latent_shape(args.height, args.width, args.num_frames)
    n_tokens = latent_shape[0] * latent_shape[1] * latent_shape[2]
    print(f"Latent shape (T, H, W): {latent_shape}")
    print(f"Total video tokens: {n_tokens}")

    env_info = get_env_info(args.device)
    print(f"Environment: {json.dumps(env_info, indent=2)}")

    collector = AttentionStatsCollector(
        num_sample_queries=args.sample_queries,
        seed=args.seed,
        latent_shape=latent_shape,
        profile_layers=profile_layers,
        profile_steps=profile_steps,
    )

    install_profiling_processors(transformer, collector)

    # ── Run inference ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = PROMPTS[:args.num_prompts]
    start_time = time.time()

    for i, prompt in enumerate(prompts):
        t0 = time.time()
        print(f"\n[{i+1}/{len(prompts)}] \"{prompt[:60]}...\"")
        collector.on_prompt_start(i)

        generator = torch.Generator(device=args.device).manual_seed(args.seed)

        def step_callback(pipe, step, timestep, kwargs):
            collector.on_step_end(step)
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
        n_steps_profiled = len(collector.all_stats.get(str(i), {}))
        print(f"  Done in {elapsed:.1f}s, profiled {n_steps_profiled} steps")

        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    def json_convert(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    results = {
        'config': {
            **vars(args),
            'prompts': prompts,
        },
        'env': env_info,
        'latent_shape': list(latent_shape),
        'total_video_tokens': n_tokens,
        'total_time_seconds': round(total_time, 1),
        'stats': collector.all_stats,
    }

    stats_file = output_dir / f"attention_stats_{timestamp}.json"
    with open(stats_file, 'w') as f:
        json.dump(results, f, default=json_convert)

    print(f"\nResults saved to {stats_file}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"File size: {stats_file.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
