"""Classification evaluation, accuracy, confusion matrix, and per-class report helpers."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """Compute accuracy, confusion matrix, normalized CM, and per-class report."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    labels = list(range(len(class_names)))

    acc = float(accuracy_score(y_t, y_p)) if len(y_t) > 0 else 0.0
    cm = confusion_matrix(y_t, y_p, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(np.float64), np.where(row_sums == 0, 1, row_sums))

    report_text = classification_report(
        y_t, y_p, labels=labels, target_names=list(class_names), digits=4, zero_division=0
    )
    return {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 6).tolist(),
        "classification_report_text": report_text,
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """Run an evaluation pass over `loader`, returning loss, accuracy, and predictions."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for x, lengths, labels in loader:
        x = x.to(device)
        labels = labels.to(device)
        logits = model(x, lengths)
        loss = criterion(logits, labels)
        bs = int(labels.size(0))
        total_loss += float(loss.item()) * bs
        total_samples += bs
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    y_true = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    return {
        "loss": float(total_loss / max(total_samples, 1)),
        "accuracy": float((y_pred == y_true).mean()) if total_samples > 0 else 0.0,
        "y_true": y_true,
        "y_pred": y_pred,
    }
