# Plotting and visualization functions for loss curves, reconstructed images, confusion matrices, and metrics.

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass


def _save(filename):
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()


def save_figure(filename):
    _save(filename)


# ------------------------------------------------------------------ Task 2: Autoencoders
def plot_reconstruction_grid(original, reconstructed, labels, filename, title):
    """Save one original/reconstruction pair for each class."""
    labels = np.asarray(labels)
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)
    classes = sorted(np.unique(labels).tolist())
    selected = [int(np.flatnonzero(labels == cls)[0]) for cls in classes]

    fig, axes = plt.subplots(2, len(selected), figsize=(2.2 * len(selected), 4.5))
    if len(selected) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, index in enumerate(selected):
        axes[0, column].imshow(original[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[1, column].imshow(
            reconstructed[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1
        )
        axes[0, column].set_title(f"Class {classes[column]}")
        axes[0, column].set_ylabel("Original")
        axes[1, column].set_ylabel("Reconstructed")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Loss curves and Task 1
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


def plot_convergence_bar_chart(epochs_summary, title="Convergence Epochs Across Optimizers",
                               filename="convergence_epochs_bar.png"):
    configurations = list(epochs_summary.keys())
    if not configurations:
        return
    first_config = configurations[0]
    optimizers = list(epochs_summary[first_config].keys())

    x = np.arange(len(optimizers))
    num_configs = len(configurations)

    fig_w = max(12, len(optimizers) * 2.0)
    fig_h = 7
    plt.figure(figsize=(fig_w, fig_h))

    width = 0.85 / max(num_configs, 1)

    for i, cfg in enumerate(configurations):
        values = [epochs_summary[cfg].get(opt, 0) for opt in optimizers]
        offset = (i - num_configs / 2 + 0.5) * width
        plt.bar(x + offset, values, width, label=cfg)

    plt.xlabel('Optimizer', fontsize=12, fontweight='bold')
    plt.ylabel('Epochs to Convergence', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, pad=15, fontweight='bold')
    plt.xticks(x, optimizers, rotation=15, fontsize=11)

    if num_configs > 6:
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=2 if num_configs > 15 else 1)
    else:
        plt.legend(fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    _save(filename)


_CLASS_COLORS = plt.get_cmap('tab10').colors


def _class_color_map(all_classes):
    return {int(c): _CLASS_COLORS[i % len(_CLASS_COLORS)]
            for i, c in enumerate(sorted(int(c) for c in all_classes))}


def plot_decision_regions(X_train, y_train, predict_fn, title="Decision Region",
                           filename="decision_region.png", all_classes=None):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    Z = predict_fn(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    if all_classes is None:
        all_classes = np.unique(y_train)
    all_classes = sorted(int(c) for c in all_classes)
    color_map = _class_color_map(all_classes)

    cmap = mcolors.ListedColormap([color_map[c] for c in all_classes])
    boundaries = [c - 0.5 for c in all_classes] + [all_classes[-1] + 0.5]
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    plt.figure(figsize=(9, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap, norm=norm, levels=boundaries)

    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                          edgecolors='k', cmap=cmap, norm=norm, s=40, zorder=3)

    classes = np.unique(y_train)
    handles, _ = scatter.legend_elements()
    plt.legend(handles, [f"Class {int(c)}" for c in classes], loc="best", title="Training Data")

    plt.xlabel('Feature 1 (x1)', fontsize=12)
    plt.ylabel('Feature 2 (x2)', fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()
    _save(filename)


def plot_node_output_surface(X, z, node_label, split_name, filename, title=None):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], z, c=z, cmap='viridis',
               alpha=0.7, s=30, edgecolors='none')

    ax.set_xlabel('x1 values', fontsize=11, labelpad=10)
    ax.set_ylabel('x2 values', fontsize=11, labelpad=10)
    ax.set_zlabel('Node Output', fontsize=11, labelpad=10)
    if title is None:
        title = f"{node_label} ({split_name.capitalize()})"
    ax.set_title(title, fontsize=14, pad=20)
    _save(filename)


def plot_node_output_1d(X, z, node_label, split_name, filename, title=None):
    plt.figure(figsize=(8, 5))
    x_flat = np.asarray(X).reshape(-1)
    sort_idx = np.argsort(x_flat)

    plt.scatter(x_flat, z, color='#9467bd', alpha=0.6, s=25, zorder=2)
    plt.plot(x_flat[sort_idx], np.asarray(z)[sort_idx], color='#9467bd',
              alpha=0.3, linewidth=1.0, zorder=1)

    plt.xlabel('x-values', fontsize=12, fontweight='bold')
    plt.ylabel('Node Output', fontsize=12, fontweight='bold')
    if title is None:
        title = f"{node_label} ({split_name.capitalize()})"
    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()
    _save(filename)


def plot_regression_1d(X, y_true, y_pred, title="1D Regression: Target vs Model", filename="reg_1d.png"):
    plt.figure(figsize=(9, 6))
    plt.scatter(X, y_true, color='#1f77b4', label='Target Output', alpha=0.6, s=30, zorder=2)
    sort_idx = np.argsort(X.flatten())
    plt.plot(X[sort_idx], y_pred[sort_idx], color='#d62728', label='Model Output', linewidth=2.5, zorder=3)

    plt.xlabel('x-values', fontsize=12, fontweight='bold')
    plt.ylabel('y-value', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, pad=15)
    plt.legend(loc="best", fontsize=11)
    plt.tight_layout()
    _save(filename)


def plot_regression_2d(X, y_true, y_pred, title="2D Regression: Target vs Model", filename="reg_2d.png"):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], y_true, color='#1f77b4', label='Target Output',
               alpha=0.6, s=30, edgecolors='none')

    ax.scatter(X[:, 0], X[:, 1], y_pred, color='#d62728', label='Model Output',
               alpha=0.6, s=30, edgecolors='none')

    ax.set_xlabel('x1 values', fontsize=11, labelpad=10)
    ax.set_ylabel('x2 values', fontsize=11, labelpad=10)
    ax.set_zlabel('y-value', fontsize=11, labelpad=10)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc="best")
    _save(filename)


def plot_target_vs_model_scatter(y_true, y_pred, title="Target vs Model Output", filename="target_vs_model.png"):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, color='#2ca02c', alpha=0.6, s=40, edgecolors='w')
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.7, label='Ideal Fit (y=x)')

    plt.xlabel('Target Output', fontsize=12, fontweight='bold')
    plt.ylabel('Model Output', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, pad=15)
    plt.legend(loc="best", fontsize=11)

    plt.axis('equal')
    plt.tight_layout()
    _save(filename)


OPTIMIZER_STYLES = {
    "SGD (batch_size=1)": {"color": "#1f77b4", "linestyle": "-", "marker": "o"},
    "Batch GD (batch_size=N)": {"color": "#17becf", "linestyle": "-", "marker": "s"},
    "SGD + Momentum (batch_size=1)": {"color": "#ff7f0e", "linestyle": "-", "marker": "^"},
    "SGD + NAG (batch_size=1)": {"color": "#d62728", "linestyle": "-", "marker": "v"},
    "AdaGrad (batch_size=N)": {"color": "#2ca02c", "linestyle": "-", "marker": "D"},
    "RMSProp (batch_size=N)": {"color": "#9467bd", "linestyle": "-", "marker": "P"},
    "Adam (batch_size=1)": {"color": "#8c564b", "linestyle": "-", "marker": "*"}
}
FALLBACK_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#17becf', '#e377c2', '#bcbd22']
FALLBACK_MARKERS = ['o', 's', '^', 'v', 'D', 'P', '*']


def plot_superimposed_error_vs_epochs(optimizer_losses, title="Average Training Error vs. Epochs",
                                      filename="superimposed_error_vs_epochs.png", log_scale=False,
                                      initial_values=None):
    plt.figure(figsize=(11, 6.5))

    for i, (opt_name, losses) in enumerate(optimizer_losses.items()):
        style = OPTIMIZER_STYLES.get(opt_name, {
            "color": FALLBACK_COLORS[i % len(FALLBACK_COLORS)],
            "linestyle": "-",
            "marker": FALLBACK_MARKERS[i % len(FALLBACK_MARKERS)]
        })
        color = style["color"]
        marker = style["marker"]

        if initial_values is not None and opt_name in initial_values:
            curve_losses = [initial_values[opt_name]] + list(losses)
            epochs = list(range(0, len(losses) + 1))
        else:
            curve_losses = list(losses)
            epochs = list(range(1, len(losses) + 1))

        plt.plot(epochs, curve_losses, label=f"{opt_name} ({len(losses)} ep)",
                 color=color, linestyle='-', linewidth=1.2, alpha=0.9, zorder=4)
        if len(curve_losses) > 0:
            end_epoch = epochs[-1]
            end_loss = curve_losses[-1]
            plt.scatter([end_epoch], [end_loss], color=color, s=45, marker=marker, zorder=5)
            plt.axvline(x=end_epoch, color=color, linestyle='--', linewidth=0.9, alpha=0.55, zorder=2)

    if initial_values is not None and len(initial_values) > 0:
        first_init = next(iter(initial_values.values()))
        plt.scatter([0], [first_init], color='black', s=45, zorder=6, label=f"Initial Loss ({first_init:.4f})")

    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Cross-Entropy Loss', fontsize=12, fontweight='bold')
    if log_scale:
        plt.yscale('log')
        plt.ylabel('Cross-Entropy Loss (log scale)', fontsize=12, fontweight='bold')

    plt.title(title, fontsize=14, pad=15, fontweight='bold')
    plt.legend(loc="upper right", fontsize=9.5, framealpha=0.95, facecolor='white', edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    _save(filename)
