"""Visualization and plotting helpers for Assignment 5."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_individual_filter(
    kernel: np.ndarray,
    path: str | Path,
    title: str = "Filter Kernel",
) -> None:
    """Save a single filter kernel as an atomic image with actual numerical values annotated in each cell."""
    h, w = kernel.shape
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(kernel, cmap="viridis")
    ax.set_title(title, fontsize=10, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = (kernel.min() + kernel.max()) / 2.0
    for r in range(h):
        for c in range(w):
            val = kernel[r, c]
            color = "white" if val < thresh else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8, fontweight="bold")

    ax.axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_atomic_feature_map(
    fmap: np.ndarray,
    path: str | Path,
    title: str = "Feature Map",
    cmap: str = "magma",
) -> None:
    """Save a single feature map channel as an atomic, standalone image with colorbar."""
    h, w = fmap.shape
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    im = ax.imshow(fmap, cmap=cmap)
    disp_title = title if f"({h}x{w})" in title else f"{title}\n({h}x{w})"
    ax.set_title(disp_title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_compact_feature_grid(
    maps: torch.Tensor,
    indices: Sequence[int],
    path: str | Path,
    title: str,
    cols: int = 3,
    stage_name: str = "Filter",
) -> None:
    """Save a compact, report-friendly grid (e.g. 2x3 or 2x5) of feature maps."""
    n_maps = len(indices)
    cols = min(cols, n_maps)
    rows = (n_maps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.8 * rows), squeeze=False)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < n_maps:
            ch = indices[i]
            fmap = maps[ch].numpy()
            im = ax.imshow(fmap, cmap="magma")
            ax.set_title(f"{stage_name} {ch + 1} ({fmap.shape[0]}x{fmap.shape[1]})", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_grid(
    images: torch.Tensor | np.ndarray,
    path: str | Path,
    title: str = "",
    cmap: str = "gray",
    cols: int = 5,
    titles: Sequence[str] | None = None,
) -> None:
    """Save an array of images in a clean grid layout."""
    if isinstance(images, torch.Tensor):
        imgs = images.detach().cpu().numpy()
    else:
        imgs = np.array(images)

    # Normalize channel layout
    if imgs.ndim == 4 and imgs.shape[1] in (1, 3):  # BCHW -> BHWC
        imgs = np.transpose(imgs, (0, 2, 3, 1))
    if imgs.ndim == 3 and imgs.shape[0] in (1, 3):  # CHW -> HWC
        imgs = np.transpose(imgs, (1, 2, 0))

    if imgs.ndim == 3 and imgs.shape[-1] not in (1, 3):
        n_images = imgs.shape[0]
    elif imgs.ndim == 4:
        n_images = imgs.shape[0]
    else:
        imgs = np.expand_dims(imgs, 0)
        n_images = 1

    cols = min(cols, n_images)
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.8 * rows), squeeze=False)

    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx < n_images:
            img = imgs[idx]
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)
            if img.dtype in (np.float32, np.float64):
                img_disp = np.clip((img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1)
            else:
                img_disp = img
            if img_disp.ndim == 2:
                ax.imshow(img_disp, cmap=cmap)
            else:
                ax.imshow(img_disp)
            if titles and idx < len(titles):
                ax.set_title(titles[idx], fontsize=9)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_filters(
    weights: torch.Tensor,
    path: str | Path,
    title: str = "Convolutional Filters with Weight Values",
    max_filters: int = 16,
) -> None:
    """Plot filter kernels as heatmaps with actual numerical values annotated in each cell."""
    weights = weights.detach().cpu()
    out_c, in_c, h, w = weights.shape
    num_to_plot = min(out_c, max_filters)

    cols = min(8, num_to_plot)
    rows = (num_to_plot + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < num_to_plot:
            kernel = weights[i, 0].numpy()  # plot first input channel slice
            im = ax.imshow(kernel, cmap="viridis")
            ax.set_title(f"Filter {i+1} ({h}x{w})", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Annotate actual numerical value in each cell
            thresh = (kernel.min() + kernel.max()) / 2.0
            for r in range(h):
                for c in range(w):
                    val = kernel[r, c]
                    color = "white" if val < thresh else "black"
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7, fontweight="bold")
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_feature_maps(
    feature_maps: torch.Tensor,
    path: str | Path,
    title: str = "Feature Maps",
    cols: int = 8,
) -> None:
    """Plot individual channels of a feature map tensor of shape (C, H, W) or (1, C, H, W)."""
    fmaps = feature_maps.detach().cpu()
    if fmaps.ndim == 4:
        fmaps = fmaps.squeeze(0)
    c, h, w = fmaps.shape
    cols = min(cols, c)
    rows = (c + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows), squeeze=False)

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < c:
            fmap = fmaps[i].numpy()
            ax.imshow(fmap, cmap="magma")
            ax.set_title(f"Ch {i+1}\n{h}x{w}", fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_filter_images(
    conv1_w: torch.Tensor,
    conv2_w: torch.Tensor,
    idx1: Sequence[int],
    idx2: Sequence[int],
    path: str | Path,
    title: str = "Convolutional Filters with Kernel Weight Values",
) -> None:
    """Draw selected kernels with their actual numerical values annotated in each cell."""
    conv1_w, conv2_w = conv1_w.detach().cpu(), conv2_w.detach().cpu()
    n = max(len(idx1), len(idx2))
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.2), squeeze=False)
    for row, (weights, indices, name) in enumerate(
        ((conv1_w, idx1, "Conv1"), (conv2_w, idx2, "Conv2"))
    ):
        for col in range(n):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(indices):
                kernel = weights[indices[col]].mean(dim=0).numpy()
                im = ax.imshow(kernel, cmap="viridis")
                suffix = "" if row == 0 else "\n(mean over in-ch)"
                ax.set_title(f"{name} Filter {indices[col] + 1}{suffix}", fontsize=9)

                # Annotate actual numerical value inside each filter cell
                thresh = (kernel.min() + kernel.max()) / 2.0
                for kr in range(kernel.shape[0]):
                    for kc in range(kernel.shape[1]):
                        val = kernel[kr, kc]
                        color = "white" if val < thresh else "black"
                        ax.text(kc, kr, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7, fontweight="bold")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_distinct_receptive_field_patch(
    image: np.ndarray | torch.Tensor,
    class_name: str,
    top_neurons: list[dict],
    path: str | Path,
    title: str = "Receptive Field Localization for Top Activated Neurons",
    is_rgb: bool = False,
) -> None:
    """Plot an individual distinct image with receptive field bounding boxes and cropped patches."""
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
    else:
        img = np.array(image)

    if not is_rgb and img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)

    n_patches = len(top_neurons)
    fig, axes = plt.subplots(1, 1 + n_patches, figsize=(3.2 * (1 + n_patches), 3.2), squeeze=False)

    # Panel 0: Full image with all bounding boxes overlaid
    ax0 = axes[0, 0]
    if is_rgb:
        ax0.imshow(np.clip(img, 0, 1))
    else:
        ax0.imshow(img, cmap="gray")

    colors = ["#ff2222", "#22ff22", "#3388ff", "#ffaa00", "#ff00ff"]
    for i, neuron in enumerate(top_neurons):
        ymin, xmin, ymax, xmax = neuron["bbox"]
        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax0.add_patch(rect)
        ax0.text(
            xmin, max(0, ymin - 1), f"N{i+1}",
            color="white", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.1", fc=color, ec="none", alpha=0.8),
        )

    ax0.set_title(f"Class: '{class_name}'", fontsize=10)
    ax0.axis("off")

    # Subsequent panels: Zoomed-in cropped patches
    for i, neuron in enumerate(top_neurons):
        ax = axes[0, 1 + i]
        ymin, xmin, ymax, xmax = neuron["bbox"]
        patch_img = img[ymin:ymax, xmin:xmax]
        if is_rgb:
            ax.imshow(np.clip(patch_img, 0, 1))
        else:
            ax.imshow(patch_img, cmap="gray")
        ch = neuron.get("channel", 0)
        val = neuron.get("activation", 0.0)
        ax.set_title(f"N{i+1} (Ch {ch+1})\nAct: {val:.2f}\n[{ymax-ymin}x{xmax-xmin}]", fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    history: dict[str, list[float]],
    path: str | Path,
    title: str = "Training and Validation Curves",
) -> None:
    """Plot loss and accuracy curves over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss vs. Epochs")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], "o-", label="Train Accuracy", color="#1f77b4")
    if "val_acc" in history and history["val_acc"]:
        ax2.plot(epochs, history["val_acc"], "s--", label="Val Accuracy", color="#ff7f0e")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy vs. Epochs")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
