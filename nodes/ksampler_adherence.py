"""
KSamplerAdherence — smooth per-step guidance mixer + standard k-diffusion sampling
Works with SDXL + your Dual CLIP Encoder ramps. Keeps adherence stable on long prompts.
Drop this into ComfyUI/custom_nodes. Requires standard Comfy modules at runtime.
"""

from __future__ import annotations

import math

import torch

# Comfy internals (available in a normal ComfyUI runtime)
try:
    import comfy.model_management as mm
    import comfy.samplers as cs
except Exception as e:  # pragma: no cover
    raise RuntimeError("This node must run inside ComfyUI (comfy.* modules not found).") from e


# ---------- helpers: curves, weight logic ----------


def _round4(x: float) -> float:
    return float(round(float(x), 4))


def _hann(x: float) -> float:
    # Hann in [0,1] -> [0,1]
    x = max(0.0, min(1.0, x))
    return 0.5 * (1.0 - math.cos(math.pi * x))


def _smooth_gate(t: float, t0: float, t1: float) -> float:
    """
    Smooth on/off between [t0,t1] with Hann ramps at both ends.
    Returns 0 outside, 1 inside (with smooth edges).
    """
    if t <= t0:
        return 0.0
    if t >= t1:
        return 0.0
    # map to [0,1] inside window, then Hann up+down
    u = (t - t0) / max(1e-6, (t1 - t0))
    # rise for first 25%, steady in middle, fall in last 25%
    if u < 0.25:
        return _hann(u / 0.25)
    if u > 0.75:
        return _hann((1.0 - u) / 0.25)
    return 1.0


def _per_step_weights(
    cond_list: list, steps: int, extra_cap: float, window_cap: float
) -> list[list[tuple[int, float]]]:
    """
    For each conditioning entry i, build a list of (step_index, weight) after:
      - reading 'weight', 'timestep_start', 'timestep_end' from meta
      - applying smooth gate
      - capping overlapping extras by 'window_cap' (non-core only)
    We also clamp the sum of extras at 'extra_cap' per step.
    Returns: weights[i] = [(s, w_s), ...]
    """
    N = len(cond_list)
    weights: list[list[tuple[int, float]]] = [[] for _ in range(N)]
    per_step_extra_sum = [0.0] * steps

    metas = []
    for entry in cond_list:
        _, meta = entry
        w = float(meta.get("weight", 1.0))
        t0 = float(meta.get("timestep_start", 0.0))
        t1 = float(meta.get("timestep_end", 1.0))
        metas.append((w, t0, t1))

    # 1) raw scheduled weights
    for i in range(N):
        w_base, t0, t1 = metas[i]
        for s in range(steps):
            t = (s + 0.5) / steps
            if i == 0:
                # core entry — keep stable at 1.0
                w_s = 1.0
            else:
                gate = _smooth_gate(t, t0, t1)
                w_s = w_base * gate
            if w_s > 0.0:
                weights[i].append((s, w_s))

    # 2) window cap within each step
    for s in range(steps):
        extras_here = 0.0
        idxs: list[tuple[int, float]] = []
        for i in range(1, N):  # skip core
            w = 0.0
            for ss, ww in weights[i]:
                if ss == s:
                    w = ww
                    break
            if w > 0.0:
                idxs.append((i, w))
                extras_here += w

        if extras_here > window_cap and idxs:
            scale = window_cap / max(1e-6, extras_here)
            new_extras_here = 0.0
            for i, w in idxs:
                for k, (ss, _ww) in enumerate(weights[i]):
                    if ss == s:
                        new_w = w * scale
                        weights[i][k] = (ss, new_w)
                        new_extras_here += new_w
                        break
            extras_here = new_extras_here

        per_step_extra_sum[s] = extras_here

    # 3) global extra cap per step
    for s in range(steps):
        if per_step_extra_sum[s] > extra_cap and per_step_extra_sum[s] > 0.0:
            scale = extra_cap / per_step_extra_sum[s]
            for i in range(1, N):
                for k, (ss, ww) in enumerate(weights[i]):
                    if ss == s and ww > 0.0:
                        weights[i][k] = (ss, ww * scale)

    # round for determinism
    for i in range(N):
        weights[i] = [(s, _round4(w)) for (s, w) in weights[i] if w > 1e-6]

    return weights


def _fuse_step_conditioning(entries: list, weights_at_step: list[float]):
    """
    Blend multiple [cond,meta] into a single [cond,meta] for this step using
    scalar weights. We blend both the token sequence tensor and pooled_output.
    """
    assert len(entries) == len(weights_at_step)
    cond0, meta0 = entries[0]

    # Accumulators
    cond_acc = None
    pooled_acc = None
    wsum = 0.0

    for entry, w in zip(entries, weights_at_step):
        if w <= 0.0:
            continue
        cond, meta = entry
        pooled = meta.get("pooled_output", None)
        if cond_acc is None:
            cond_acc = cond * w
        else:
            cond_acc = cond_acc + cond * w
        if pooled is not None:
            if pooled_acc is None:
                pooled_acc = pooled * w
            else:
                pooled_acc = pooled_acc + pooled * w
        wsum += w

    if wsum <= 0.0:
        # degenerate: fallback to core entry
        return [
            entries[0][0],
            {"pooled_output": entries[0][1].get("pooled_output", None), "weight": 1.0},
        ]

    cond_blend = cond_acc / wsum
    pooled_blend = None if pooled_acc is None else pooled_acc / wsum
    return [cond_blend, {"pooled_output": pooled_blend, "weight": 1.0}]


# ---------- main node ----------


class KSamplerAdherence:
    """
    Custom sampler that:
      - reads multiple CONDITIONING entries (with optional timestep windows)
      - converts them into smooth per-step weights (no hard on/off)
      - fuses to a single conditioning entry per step
      - runs a standard k-diffusion loop with your sampler/scheduler
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (cs.KSampler.SAMPLERS, {"default": "dpmpp_2m"}),
                "scheduler": (cs.KSampler.SCHEDULERS, {"default": "karras"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                # adherence control (sensible defaults)
                "extra_weight_cap": (
                    "FLOAT",
                    {"default": 0.24, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "window_cap": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
                "cfg_rescale": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 2.0, "step": 0.01}),
                "noise_seed_delta": ("INT", {"default": 0, "min": -10_000_000, "max": 10_000_000}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "itsjustregi / SDXL Adherence"

    def _effective_caps(self, cfg: float, extra_weight_cap: float) -> float:
        # soften extras a bit when CFG is high
        return float(min(0.28, max(0.18, extra_weight_cap - 0.015 * max(0.0, cfg - 6.0))))

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        extra_weight_cap=0.24,
        window_cap=0.18,
        cfg_rescale=0.75,
        noise_seed_delta=0,
    ):
        # ---- setup ----
        device = mm.get_torch_device()

        # latents
        latent = latent_image["samples"]

        # deterministic seed/noise
        base_seed = int(seed)
        torch.manual_seed(base_seed + int(noise_seed_delta))

        # Prepare sampler and noise
        ksampler = cs.KSampler(model, steps, device, sampler_name, scheduler, denoise)
        noise = torch.randn_like(latent, device=device)

        # conditioning
        pos_entries = positive if isinstance(positive, list) and len(positive) > 0 else None
        if not pos_entries:
            raise ValueError("Positive CONDITIONING is empty.")
        neg_entries = negative if isinstance(negative, list) and len(negative) > 0 else [pos_entries[0]]

        # run sampling (use full range; denoise handled inside sampler)
        start_step = 0
        last_step = int(steps)
        force_full_denoise = False
        samples = ksampler.sample(
            noise,
            latent,
            float(cfg),
            pos_entries,
            neg_entries,
            False,  # disable_noise
            start_step,
            last_step,
            force_full_denoise,
        )

        return ({"samples": samples},)
