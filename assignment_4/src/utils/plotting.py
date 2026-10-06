# Plotting and visualization functions for loss curves, reconstructed images, confusion matrices, and weight filters.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass


def _save(filename):
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()


def plot_error_vs_epochs(errors, title="Average Error vs Epochs", filename="error_vs_epochs.png",
                         initial_value=None, ylabel="Average training loss"):
    plt.figure(figsize=(8, 5))
    if initial_value is not None:
        values, epochs = [initial_value] + list(errors), list(range(0, len(errors) + 1))
    else:
        values, epochs = list(errors), list(range(1, len(errors) + 1))
    plt.plot(epochs, values, marker="o", markersize=3, linewidth=1.5)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.tight_layout()
    _save(filename)


def plot_confusion_matrix_heatmap(cm, class_names=None, title="Confusion Matrix",
                                  filename="confusion_matrix.png"):
    cm = np.asarray(cm)
    n = cm.shape[0]
    class_names = class_names or [f"Class {i}" for i in range(n)]

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, pad=15, fontweight="bold")
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(n)
    plt.xticks(ticks, class_names, fontsize=11)
    plt.yticks(ticks, class_names, fontsize=11)

    thresh, total = cm.max() / 2.0, cm.sum()
    for i in range(n):
        for j in range(n):
            pct = cm[i, j] / total * 100.0 if total > 0 else 0.0
            plt.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=10, fontweight="bold")

    plt.ylabel("True Label", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(filename)


def plot_accuracy_vs_dimension(dims, test_accs, reference=None, reference_label="Assignment 3 best",
                               title="Test accuracy vs reduced dimension", filename="test_acc_vs_dim.png"):
    """Bar chart of test accuracy (%) per reduced dimension, optional reference line."""
    plt.figure(figsize=(7, 5))
    xs = np.arange(len(dims))
    bars = plt.bar(xs, test_accs, color="#4C78A8", width=0.55)
    for b, v in zip(bars, test_accs):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    if reference is not None:
        plt.axhline(reference, color="#E45756", linestyle="--", linewidth=1.5,
                    label=f"{reference_label} ({reference:.2f}%)")
        plt.legend(loc="upper right")
    lo = min(list(test_accs) + ([reference] if reference is not None else []))
    plt.ylim(max(0, lo - 2.0), 100.0)
    plt.xticks(xs, [str(d) for d in dims], fontsize=11)
    plt.xlabel("Reduced dimension", fontsize=11)
    plt.ylabel("Test accuracy (%)", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(filename)


def plot_val_accuracy_heatmap(arch_names, dims, val_acc_matrix, title="Validation accuracy (%)",
                              filename="val_acc_heatmap.png"):
    """val_acc_matrix[i][j] = validation accuracy of arch i at dimension j (NaN if missing)."""
    mat = np.asarray(val_acc_matrix, dtype=float)
    plt.figure(figsize=(1.3 * len(dims) + 3.5, 0.5 * len(arch_names) + 2))
    plt.imshow(mat, cmap="YlGnBu", aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(dims)), [str(d) for d in dims])
    plt.yticks(range(len(arch_names)), arch_names)
    vmax = np.nanmax(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                best_in_col = mat[i, j] == np.nanmax(mat[:, j])
                plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9,
                         fontweight="bold" if best_in_col else "normal",
                         color="white" if mat[i, j] > 0.5 * (vmax + np.nanmin(mat)) else "black")
    plt.xlabel("Reduced dimension", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(filename)