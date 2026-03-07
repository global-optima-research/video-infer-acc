#!/usr/bin/env python3
"""
Phase 1: Shared Utilities
=========================
Common attention processor, model loading, and video quality metrics
for all Phase 1 experiments (1a-1d).
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import sys
import subprocess
import gc
import time
from pathlib import Path
from datetime import datetime


# ─── Prompts (same as Phase 0 for consistency) ───────────────────────────────

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


# ─── Skippable Attention Processor ───────────────────────────────────────────

class SkippableWanAttnProcessor:
    """
    WanAttnProcessor that supports:
    1. Normal computation with output caching
    2. Skip mode: return cached output from previous step
    3. Output capture: record attention output for analysis

    The skip_controller function decides per (step, layer) whether to compute or skip.
    """

    def __init__(self, layer_idx, attn_type='self'):
        self.layer_idx = layer_idx
        self.attn_type = attn_type
        # Cache for skip mode
        self._cached_output = None
        # Controller: callable(step, layer_idx) -> bool (True = skip)
        self.skip_controller = None
        self.current_step = 0
        # Optional: capture output for analysis
        self.capture_output = False
        self.captured_outputs = {}  # step -> tensor (on CPU)

    def should_skip(self):
        """Check if this layer should be skipped at current step."""
        if self.skip_controller is None:
            return False
        if self._cached_output is None:
            return False  # Can't skip if no cache
        return self.skip_controller(self.current_step, self.layer_idx)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
    ):
        # Only skip self-attention
        if self.attn_type == 'self' and self.should_skip():
            return self._cached_output

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

        # ── Standard attention ──
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

        # Cache for potential reuse
        self._cached_output = hidden_states.detach()

        # Capture for analysis
        if self.capture_output and self.attn_type == 'self':
            self.captured_outputs[self.current_step] = hidden_states.detach().cpu()

        return hidden_states

    def clear_cache(self):
        self._cached_output = None
        self.captured_outputs = {}


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_wan_pipeline(model_name="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", device="cuda:0"):
    """Load Wan pipeline with float32 VAE for quality."""
    from diffusers import AutoencoderKLWan, WanPipeline

    vae = AutoencoderKLWan.from_pretrained(
        model_name, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(
        model_name, vae=vae, torch_dtype=torch.bfloat16
    )
    pipe.to(device)
    return pipe


def install_skippable_processors(transformer):
    """Replace attention processors with SkippableWanAttnProcessor. Returns dict of processors."""
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
                proc = SkippableWanAttnProcessor(layer_idx, attn_type)
                module.processor = proc
                key = f"{attn_type}_{layer_idx}"
                processors[key] = proc

    print(f"Installed {len(processors)} skippable processors")
    return processors


def set_all_steps(processors, step):
    """Update current_step on all processors."""
    for proc in processors.values():
        proc.current_step = step


def clear_all_caches(processors):
    """Clear caches on all processors."""
    for proc in processors.values():
        proc.clear_cache()


# ─── Video Generation ────────────────────────────────────────────────────────

def generate_video(pipe, prompt, processors, height=480, width=832,
                   num_frames=17, num_steps=50, seed=42, device="cuda:0",
                   step_callback=None):
    """
    Generate a single video, returning decoded frames as tensor.

    Args:
        step_callback: callable(step) called after each denoising step
    Returns:
        frames: tensor of shape (num_frames, 3, H, W) in [0, 1]
    """
    generator = torch.Generator(device=device).manual_seed(seed)

    clear_all_caches(processors)

    def _on_step_end(pipe_obj, step, timestep, kwargs):
        set_all_steps(processors, step + 1)
        if step_callback:
            step_callback(step)
        return kwargs

    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            generator=generator,
            callback_on_step_end=_on_step_end,
            output_type="np",
        )

    # output.frames is list of list of PIL or np array
    frames = output.frames[0]  # list of np arrays (H, W, 3) in [0, 1]
    frames_tensor = torch.from_numpy(np.stack(frames))  # (F, H, W, 3)
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)   # (F, 3, H, W)
    return frames_tensor


# ─── Quality Metrics ─────────────────────────────────────────────────────────

def compute_ssim(frames_a, frames_b):
    """
    Compute mean SSIM between two video tensors.
    frames_a, frames_b: (F, 3, H, W) in [0, 1]
    Uses a simple per-channel SSIM implementation.
    """
    from torch.nn.functional import avg_pool2d

    assert frames_a.shape == frames_b.shape

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    window_size = 11
    padding = window_size // 2

    ssim_vals = []
    for i in range(frames_a.shape[0]):
        for c in range(3):
            img1 = frames_a[i, c:c+1, :, :].unsqueeze(0).float()
            img2 = frames_b[i, c:c+1, :, :].unsqueeze(0).float()

            mu1 = avg_pool2d(img1, window_size, stride=1, padding=padding)
            mu2 = avg_pool2d(img2, window_size, stride=1, padding=padding)

            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = avg_pool2d(img1 * img1, window_size, stride=1, padding=padding) - mu1_sq
            sigma2_sq = avg_pool2d(img2 * img2, window_size, stride=1, padding=padding) - mu2_sq
            sigma12 = avg_pool2d(img1 * img2, window_size, stride=1, padding=padding) - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            ssim_vals.append(ssim_map.mean().item())

    return np.mean(ssim_vals)


def compute_psnr(frames_a, frames_b):
    """Compute mean PSNR between two video tensors (F, 3, H, W) in [0, 1]."""
    mse = ((frames_a.float() - frames_b.float()) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def save_video_frames(frames_tensor, output_dir, prefix="frame"):
    """Save video frames as PNG files. frames_tensor: (F, 3, H, W) in [0, 1]."""
    from PIL import Image
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(frames_tensor.shape[0]):
        img = (frames_tensor[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(output_dir / f"{prefix}_{i:04d}.png")


# ─── Environment Info ────────────────────────────────────────────────────────

def get_env_info(device="cuda:0"):
    """Collect environment information for reproducibility."""
    info = {
        'python_version': sys.version.split()[0],
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'N/A',
        'gpu_memory_gb': round(torch.cuda.get_device_properties(device).total_memory / 1e9, 1)
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


def get_latent_shape(height, width, num_frames):
    """Compute latent shape for Wan 2.1."""
    t = (num_frames - 1) // 4 + 1
    h = height // 8
    w = width // 8
    return (t, h, w)


def json_convert(obj):
    """JSON serializer for numpy/torch types."""
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")
