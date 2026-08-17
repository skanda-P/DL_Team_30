# Plotting utilities
try:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-not-found]
except ImportError:
    plt = None
    Axes3D = None

import numpy as np
import os

def _require_matplotlib():
    if plt is None:
        raise ImportError("matplotlib is required. Install it with: pip install matplotlib")

# Access the imported 3D axis class to avoid lint warnings in environments where the
# module is available but not statically resolved.
if Axes3D is not None:
    _ = Axes3D

os.makedirs('figures', exist_ok=True)
if plt is not None:
    plt.style.use('seaborn-v0_8-whitegrid')

def plot_error_vs_epochs(errors, title="Average Error vs Epochs", filename="error_vs_epochs.png"):
    """
    Plots the average error vs epochs and saves it to the figures directory.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(errors) + 1), errors, marker='o', markersize=4, 
             linestyle='-', color='#1f77b4', linewidth=1.5)
    
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Average Error', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()
    
    filepath = os.path.join('figures', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def plot_decision_regions(X_train, y_train, predict_fn, title="Decision Region", filename="decision_region.png"):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    Z = predict_fn(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(9, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                          edgecolors='k', cmap='viridis', s=40, zorder=3)
    
    classes = np.unique(y_train)
    handles, _ = scatter.legend_elements()
    plt.legend(handles, [f"Class {int(c)}" for c in classes], loc="best", title="Training Data")
    
    plt.xlabel('Feature 1 (x1)', fontsize=12)
    plt.ylabel('Feature 2 (x2)', fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()
    
    filepath = os.path.join('figures', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
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
    
    filepath = os.path.join('figures', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
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
    
    filepath = os.path.join('figures', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
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
    
    filepath = os.path.join('figures', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()