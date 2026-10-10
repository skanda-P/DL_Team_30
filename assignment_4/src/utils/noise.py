# Input corruption for denoising autoencoders: every feature is corrupted independently with probability p.
"""
Noise model for Task 5.

"p% noise" means: every input feature (pixel) of every sample is corrupted with probability p
(p = 0.2 or 0.4), independently, and a NEW corruption pattern is drawn in every epoch. Over E epochs
a given feature therefore receives noise in about p * E of them (e.g. 20 of 100 epochs for p = 0.2).

kind = "gaussian": a corrupted feature gets zero-mean Gaussian noise added (std `std`).
kind = "masking" : a corrupted feature is set to 0 (classical masking noise).
Corrupted inputs are clipped to the valid pixel range [0, 1] when clip=True.
The reconstruction target is always the clean image.
"""

import torch


def make_generator(device, seed):
    """Seeded random generator living on `device` (needed so noise generation stays on the GPU)."""
    return torch.Generator(device=torch.device(device)).manual_seed(int(seed))


def corrupt(x, p, std=0.5, kind="gaussian", clip=True, generator=None):
    """Return a corrupted copy of x (n, d). Each entry is corrupted independently with probability p."""
    if p <= 0:
        return x.clone()
    hit = torch.rand(x.shape, device=x.device, generator=generator) < p        # which features get noise
    if kind == "gaussian":
        noise = torch.randn(x.shape, device=x.device, generator=generator) * std
        out = x + noise * hit
    elif kind == "masking":
        out = x * (~hit)
    else:
        raise ValueError(f"Unknown noise kind '{kind}' (use 'gaussian' or 'masking')")
    return out.clamp_(0.0, 1.0) if clip else out
