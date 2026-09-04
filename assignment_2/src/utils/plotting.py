try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    raise ImportError("matplotlib is required. Install it with: pip install matplotlib")
import numpy as np


_CLASS_COLORS = plt.get_cmap('tab10').colors


def _class_color_map(all_classes):
    return {int(c): _CLASS_COLORS[i % len(_CLASS_COLORS)]
            for i, c in enumerate(sorted(int(c) for c in all_classes))}


plt.style.use('seaborn-v0_8-whitegrid')


def plot_error_vs_epochs(errors, title="Average Error vs Epochs", filename="error_vs_epochs.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(errors) + 1), errors, marker='o', markersize=4,
             linestyle='-', color='#1f77b4', linewidth=1.5)

    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Average Error', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


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

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_node_output_surface(X, z, node_label, split_name, filename, title=None):
    # 3D scatter plot of node output over 2D input space
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

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


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

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_regression_2d(X, y_true, y_pred, title="2D Regression: Target vs Model", filename="reg_2d.png"):
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

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


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

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
