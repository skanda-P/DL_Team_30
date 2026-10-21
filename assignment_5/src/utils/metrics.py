"""Evaluation metrics and confusion matrix plotting for Assignment 5."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 classification accuracy."""
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def confusion_matrix(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute a CPU confusion matrix with true classes in rows."""
    p = predictions.reshape(-1).long().cpu()
    y = labels.reshape(-1).long().cpu()
    return torch.bincount(
        y * num_classes + p, minlength=num_classes**2
    ).view(num_classes, num_classes)


def evaluate_split(model: torch.nn.Module, loader, device: torch.device, num_classes: int = 5):
    """Evaluate logits, accuracy, and confusion matrix for one data split."""
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits.append(model(images))
            labels.append(targets)
    all_logits = torch.cat(logits)
    all_labels = torch.cat(labels)
    return (
        accuracy(all_logits, all_labels),
        confusion_matrix(all_logits.argmax(dim=-1), all_labels, num_classes),
    )


def plot_confusion_matrix(
    matrix: torch.Tensor | np.ndarray,
    class_names: Sequence[str],
    title: str,
    save_path: str | Path,
) -> None:
    """Plot and save a formatted confusion matrix heatmap with counts."""
    if isinstance(matrix, torch.Tensor):
        cm = matrix.detach().cpu().numpy()
    else:
        cm = np.asarray(matrix)

    num_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
