"""Convolutional Neural Network Pipeline for Handwritten Digit Analysis and Classification.

This module provides an end-to-end implementation for studying convolutional operations,
hierarchical feature extraction, and classification dynamics on the MNIST dataset:
1. Single-filter spatial convolution and output dimension verification.
2. Two-stage sequential convolution feature hierarchies.
3. Supervised classification training using Stochastic Gradient Descent (SGD).
4. Atomic filter weights saved individually in results/partA/filters/ with actual numerical cell values.
5. Atomic feature maps saved individually in results/partA/feature_maps/ per class for seamless report inclusion.
6. Analytical receptive field calculation and top-activated input patch tracing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.metrics import evaluate_split
from utils.plotting import (
    plot_compact_feature_grid,
    plot_distinct_receptive_field_patch,
    plot_training_curves,
    save_atomic_feature_map,
    save_individual_filter,
)
from utils.receptive_field import calculate_receptive_field, neuron_to_bounding_box
from utils.select_images import get_deterministic_images
from utils.seed import set_seed

# Project paths
ROOT = Path(__file__).resolve().parent
MNIST_DIR = ROOT / "data" / "mnist_subset"
RESULTS_DIR = ROOT / "results" / "partA"
MODELS_DIR = ROOT / "models"
FILTERS_DIR = RESULTS_DIR / "filters"
FMAPS_DIR = RESULTS_DIR / "feature_maps"

# Standard MNIST image preprocessing: single-channel grayscale and float tensor normalization
MNIST_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


def conv_out_size(n: int, k: int, s: int = 1, p: int = 0) -> int:
    """Return the spatial output size of a convolution or pooling operation: floor((N - K + 2P)/S) + 1."""
    return (n - k + 2 * p) // s + 1


def manual_conv2d(
    img: torch.Tensor,
    kernel: torch.Tensor,
    bias: float = 0.0,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """Apply a single-channel kernel using an explicit sliding-window traversal."""
    if padding:
        img = F.pad(img, (padding,) * 4)
    kernel_size = kernel.shape[-1]
    height, width = img.shape
    out_h = conv_out_size(height, kernel_size, stride)
    out_w = conv_out_size(width, kernel_size, stride)
    output = torch.zeros(out_h, out_w, dtype=img.dtype)
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            patch = img[
                row * stride : row * stride + kernel_size,
                col * stride : col * stride + kernel_size,
            ]
            output[row, col] = (patch * kernel).sum() + bias
    return output


def expected_stack_shapes(height: int = 28, width: int = 28) -> dict[str, tuple[int, int, int]]:
    """Compute theoretical spatial shapes through two-stage convolution hierarchy."""
    conv1 = conv_out_size(height, 5)
    pool1 = conv_out_size(conv1, 2, 2)
    conv2 = conv_out_size(pool1, 5)
    pool2 = conv_out_size(conv2, 2, 2)
    return {
        "conv1": (6, conv1, conv1),
        "pool1": (6, pool1, pool1),
        "conv2": (16, conv2, conv2),
        "pool2": (16, pool2, pool2),
    }


# ============================================================================
# Device Configuration and Weight Initialization
# ============================================================================

def get_device() -> torch.device:
    """Detect and return the active compute device (CUDA GPU if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""
    print(f"Active compute device: {device}{gpu_name}")
    return device


def initialize_layer_weights(module: nn.Module) -> None:
    """Initialize convolutional layers with Kaiming normal and linear layers with Xavier uniform."""
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


# ============================================================================
# Dataset Loading and Deterministic Sampling
# ============================================================================

def load_mnist_dataloaders(batch_size: int = 64, data_dir: Path = MNIST_DIR):
    """Load train, validation, and test splits into PyTorch DataLoaders with pinned memory."""
    splits = {
        split: datasets.ImageFolder(str(data_dir / split), transform=MNIST_TRANSFORM)
        for split in ("train", "val", "test")
    }
    class_to_idx = splits["train"].class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in splits.items()
    }
    return loaders, class_to_idx, idx_to_class


def sample_representative_images(split: str = "train", data_dir: Path = MNIST_DIR):
    """Retrieve exactly one deterministic sample image per class directly from disk for reproducible analysis."""
    img_tensors, labels, paths = [], [], []
    selected = get_deterministic_images(data_dir, split)
    class_names = [class_name for class_name, _ in selected]
    for idx, (class_name, selected_path) in enumerate(selected):
        paths.append(selected_path)
        with Image.open(selected_path) as pil_img:
            img_tensors.append(MNIST_TRANSFORM(pil_img))
        labels.append(idx)

    return torch.stack(img_tensors, dim=0), labels, class_names, paths


# ============================================================================
# Convolutional Neural Network Architectures
# ============================================================================

class TwoStageConvolution(nn.Module):
    """Two-stage convolutional feature extractor with intermediate ReLU activations and max pooling."""

    def __init__(self):
        super().__init__()
        # Stage 1: Conv(1 -> 6, 5x5, stride=1, pad=0) -> ReLU -> MaxPool(2x2, stride=2)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2: Conv(6 -> 16, 5x5, stride=1, pad=0) -> ReLU -> MaxPool(2x2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        initialize_layer_weights(self)

    def forward(self, x: torch.Tensor):
        """Pass input through both convolutional stages and return activations at all intermediate steps."""
        c1 = self.conv1(x)
        r1 = self.relu1(c1)
        p1 = self.pool1(r1)

        c2 = self.conv2(p1)
        r2 = self.relu2(c2)
        p2 = self.pool2(r2)

        return {
            "conv1": c1,
            "relu1": r1,
            "pool1": p1,
            "conv2": c2,
            "relu2": r2,
            "pool2": p2,
        }


class DigitClassifierCNN(nn.Module):
    """End-to-end convolutional network with two conv-pool stages and dense classification head."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        # Convolutional feature extraction hierarchy
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected classification head: Flatten -> Linear(256, 64) -> ReLU -> Linear(64, num_classes)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

        initialize_layer_weights(self)

    def extract_intermediate_features(self, x: torch.Tensor):
        """Extract intermediate activations for visualization and receptive field analysis."""
        c1 = self.conv1(x)
        r1 = self.relu1(c1)
        p1 = self.pool1(r1)

        c2 = self.conv2(p1)
        r2 = self.relu2(c2)
        p2 = self.pool2(r2)
        return {"conv1": c1, "relu1": r1, "pool1": p1, "conv2": c2, "relu2": r2, "pool2": p2}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor and dense classification head."""
        f = self.extract_intermediate_features(x)
        flat = self.flatten(f["pool2"])
        hidden = self.relu3(self.fc1(flat))
        return self.fc2(hidden)


# ============================================================================
# Single-Filter Spatial Convolution Analysis
# ============================================================================

def single_filter_convolution_analysis(
    images: torch.Tensor,
    class_names: list[str],
    paths: list[Path],
    device: torch.device,
) -> None:
    """Analyze single 5x5 convolution filter transformation and verify output spatial dimensions.

    Saves the filter kernel individually in results/partA/filters/ with actual numerical cell values,
    and saves distinct 3-panel and atomic feature map figures for each class.
    """
    print("\n--- Single 5x5 Convolution Filter Analysis ---")
    FILTERS_DIR.mkdir(parents=True, exist_ok=True)
    FMAPS_DIR.mkdir(parents=True, exist_ok=True)

    # Define a single 5x5 convolutional filter
    conv = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0, bias=True).to(device)
    nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
    nn.init.zeros_(conv.bias)

    filter_w = conv.weight.detach().cpu().squeeze().numpy()

    # Save filter kernel as an atomic individual image with actual cell values
    save_individual_filter(
        filter_w,
        FILTERS_DIR / "single_filter_kernel.png",
        "Single 5x5 Convolution Kernel Weights",
    )

    # Perform forward convolution on representative sample digits
    with torch.no_grad():
        feature_maps = conv(images.to(device)).cpu()

    # Calculate theoretical vs actual output dimensions
    input_h, input_w = images.shape[2], images.shape[3]
    expected_h = conv_out_size(input_h, 5, s=1, p=0)
    expected_w = conv_out_size(input_w, 5, s=1, p=0)
    actual_h, actual_w = feature_maps.shape[2], feature_maps.shape[3]

    print(f"Input dimensions: {tuple(images.shape)}")
    print(f"Output feature map dimensions: {tuple(feature_maps.shape)}")
    print(f"Theoretical output resolution: ({expected_h}, {expected_w})")
    assert (actual_h, actual_w) == (expected_h, expected_w), "Spatial dimension mismatch"

    # Verify library convolution against explicit sliding-window traversal
    kernel_cpu = conv.weight.detach().cpu()[0, 0]
    bias_cpu = conv.bias.detach().cpu()[0].item()
    manual = torch.stack(
        [manual_conv2d(images[i, 0], kernel_cpu, bias_cpu) for i in range(len(images))]
    ).unsqueeze(1)
    assert torch.allclose(manual, feature_maps, atol=1e-5), "manual conv != nn.Conv2d"
    print("Manual sliding-window convolution matches nn.Conv2d")

    # Save distinct figures for each class
    for i, cname in enumerate(class_names):
        class_fmap_dir = FMAPS_DIR / f"class_{cname}"
        class_fmap_dir.mkdir(parents=True, exist_ok=True)

        # 1. Atomic standalone feature map
        save_atomic_feature_map(
            feature_maps[i, 0].numpy(),
            class_fmap_dir / "single_filter_output.png",
            f"Single Filter Output: Class '{cname}'",
        )

        # 2. Distinct 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        axes[0].imshow(images[i, 0].cpu().numpy(), cmap="gray")
        axes[0].set_title(f"Input: '{cname}' (28x28)", fontsize=10)
        axes[0].axis("off")

        im_f = axes[1].imshow(filter_w, cmap="viridis")
        axes[1].set_title("5x5 Kernel Weights", fontsize=10)
        fig.colorbar(im_f, ax=axes[1], fraction=0.046, pad=0.04)
        thresh = (filter_w.min() + filter_w.max()) / 2.0
        for kr in range(5):
            for kc in range(5):
                val = filter_w[kr, kc]
                color = "white" if val < thresh else "black"
                axes[1].text(kc, kr, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8, fontweight="bold")
        axes[1].axis("off")

        im_out = axes[2].imshow(feature_maps[i, 0].numpy(), cmap="magma")
        axes[2].set_title(f"Feature Map ({actual_h}x{actual_w})", fontsize=10)
        axes[2].axis("off")
        fig.colorbar(im_out, ax=axes[2], fraction=0.046, pad=0.04)

        fig.suptitle(f"Single Filter Convolution: Class '{cname}'", fontsize=11)
        fig.tight_layout()
        class_plot_path = RESULTS_DIR / f"single_filter_{cname}.png"
        fig.savefig(class_plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Persist numerical records of filter transformation
    records = {
        "classes": class_names,
        "sample_paths": [str(p) for p in paths],
        "input_shape": list(images.shape),
        "filter_weights": filter_w.tolist(),
        "expected_spatial": [expected_h, expected_w],
        "obtained_spatial": [actual_h, actual_w],
    }
    (RESULTS_DIR / "single_filter_observations.json").write_text(json.dumps(records, indent=2))
    print(f"Saved atomic filter to {FILTERS_DIR / 'single_filter_kernel.png'}")


# ============================================================================
# Hierarchical Stacked Convolution Analysis
# ============================================================================

def stacked_convolution_analysis(
    images: torch.Tensor,
    class_names: list[str],
    device: torch.device,
) -> None:
    """Evaluate sequential hierarchical transformations across two conv-pooling stages.

    Saves each filter individually into results/partA/filters/, and saves atomic feature maps
    in results/partA/feature_maps/class_{class_name}/.
    """
    print("\n--- Two-Stage Stacked Convolution Analysis ---")
    FILTERS_DIR.mkdir(parents=True, exist_ok=True)
    FMAPS_DIR.mkdir(parents=True, exist_ok=True)
    model = TwoStageConvolution().to(device).eval()

    with torch.no_grad():
        out = model(images.to(device))

    # Verify expected tensor shapes at each step in the hierarchy
    expected_shapes = expected_stack_shapes(images.shape[2], images.shape[3])
    for layer_name, exp_dim in expected_shapes.items():
        actual_dim = tuple(out[layer_name].shape[1:])
        print(f"{layer_name}: expected {exp_dim}, got {actual_dim}")
        assert actual_dim == exp_dim, f"Dimension mismatch in {layer_name}"

    conv1_w = model.conv1.weight.detach().cpu()
    conv2_w = model.conv2.weight.detach().cpu()
    selected_filter_indices = [0, 1, 2]

    # Save EACH initialized filter individually in results/partA/filters/
    for idx in selected_filter_indices:
        save_individual_filter(
            conv1_w[idx, 0].numpy(),
            FILTERS_DIR / f"stacked_conv1_filter_{idx+1}.png",
            f"Stacked Conv1 Filter {idx+1} (5x5)",
        )
        save_individual_filter(
            conv2_w[idx].mean(dim=0).numpy(),
            FILTERS_DIR / f"stacked_conv2_filter_{idx+1}.png",
            f"Stacked Conv2 Filter {idx+1} (5x5, Mean over in-ch)",
        )

    # Save atomic and compact feature maps for each class
    for i, cname in enumerate(class_names):
        class_dir = FMAPS_DIR / f"class_{cname}"
        class_dir.mkdir(parents=True, exist_ok=True)

        # Save atomic input image
        plt.imsave(class_dir / "input.png", images[i, 0].cpu().numpy(), cmap="gray")

        # Save atomic feature maps for Conv1 and Conv2 after ReLU
        for idx in selected_filter_indices:
            save_atomic_feature_map(
                out["relu1"][i, idx].cpu().numpy(),
                class_dir / f"stacked_conv1_f{idx+1}_relu.png",
                f"Conv1 Filter {idx+1} (ReLU)",
            )
            save_atomic_feature_map(
                out["relu2"][i, idx].cpu().numpy(),
                class_dir / f"stacked_conv2_f{idx+1}_relu.png",
                f"Conv2 Filter {idx+1} (ReLU)",
            )
            save_atomic_feature_map(
                out["pool1"][i, idx].cpu().numpy(),
                class_dir / f"stacked_pool1_f{idx+1}.png",
                f"Layer 1 Pool1 (12x12)",
            )
            save_atomic_feature_map(
                out["pool2"][i, idx].cpu().numpy(),
                class_dir / f"stacked_pool2_f{idx+1}.png",
                f"Layer 2 Pool2 (4x4)",
            )

        # Save compact, report-friendly grids (clean 1x3 strips for Stage 1 and Stage 2)
        plot_compact_feature_grid(
            out["relu1"][i].cpu(),
            selected_filter_indices,
            FMAPS_DIR / f"stacked_features_{cname}_conv1_relu.png",
            f"Stage 1 (Conv1+ReLU) Feature Maps: Class '{cname}'",
            cols=3,
            stage_name="Conv1 F",
        )
        plot_compact_feature_grid(
            out["relu2"][i].cpu(),
            selected_filter_indices,
            FMAPS_DIR / f"stacked_features_{cname}_conv2_relu.png",
            f"Stage 2 (Conv2+ReLU) Feature Maps: Class '{cname}'",
            cols=3,
            stage_name="Conv2 F",
        )

    # Print representative feature map tensors as requested by assignment
    torch.set_printoptions(precision=3, linewidth=160, sci_mode=False)
    for name in ("conv1", "relu1", "pool1", "conv2", "relu2", "pool2"):
        print(
            f"\n{name} output, class '{class_names[0]}', channel 1:\n"
            f"{out[name][0, 0].cpu()}"
        )
    np.savez(
        RESULTS_DIR / "stacked_feature_maps.npz",
        **{name: value.cpu().numpy() for name, value in out.items()},
    )

    records = {
        "classes": class_names,
        "conv1_shape": list(out["conv1"].shape),
        "pool1_shape": list(out["pool1"].shape),
        "conv2_shape": list(out["conv2"].shape),
        "pool2_shape": list(out["pool2"].shape),
        "selected_filter_indices_0based": selected_filter_indices,
        "conv1_filters_selected": conv1_w[selected_filter_indices, 0].tolist(),
        "conv2_filters_selected_mean_over_inchannels": conv2_w[selected_filter_indices].mean(dim=1).tolist(),
    }
    (RESULTS_DIR / "stacked_convolution_observations.json").write_text(json.dumps(records, indent=2))
    print(f"Saved individual filters to {FILTERS_DIR} and atomic feature maps to {FMAPS_DIR}")


# ============================================================================
# Classification Training and Evaluation
# ============================================================================

def train_classifier_and_visualize_features(
    loaders: dict[str, DataLoader],
    class_names: list[str],
    images: torch.Tensor,
    epochs: int = 15,
    device: torch.device = torch.device("cpu"),
    momentum: float = 0.0,
    eval_only: bool = False,
) -> DigitClassifierCNN:
    """Train the CNN classifier with SGD, evaluate metrics, and visualize filters and receptive fields."""
    print("\n--- Training Convolutional Classifier with SGD ---")
    FILTERS_DIR.mkdir(parents=True, exist_ok=True)
    FMAPS_DIR.mkdir(parents=True, exist_ok=True)

    model = DigitClassifierCNN(num_classes=5).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    checkpoint_path = MODELS_DIR / "mnist_cnn.pth"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    if eval_only and checkpoint_path.exists():
        print(f"Evaluation mode: loading existing checkpoint from {checkpoint_path}")
    else:
        # Epoch training loop with loss tracking and validation checkpointing
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum, correct, total = 0.0, 0, 0
            for imgs, targets in loaders["train"]:
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item() * imgs.size(0)
                correct += (out.argmax(dim=1) == targets).sum().item()
                total += targets.size(0)

            tr_loss = loss_sum / total
            tr_acc = correct / total

            # Evaluate validation performance
            model.eval()
            v_loss_sum, v_correct, v_total = 0.0, 0, 0
            with torch.no_grad():
                for imgs, targets in loaders["val"]:
                    imgs, targets = imgs.to(device), targets.to(device)
                    out = model(imgs)
                    v_loss_sum += criterion(out, targets).item() * imgs.size(0)
                    v_correct += (out.argmax(dim=1) == targets).sum().item()
                    v_total += targets.size(0)

            v_loss = v_loss_sum / v_total
            v_acc = v_correct / v_total

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(v_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(v_acc)

            # Checkpoint model with best validation accuracy
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                torch.save({"model_state_dict": model.state_dict(), "val_acc": v_acc}, checkpoint_path)

            if epoch % 5 == 0 or epoch == epochs:
                print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Acc: {tr_acc*100:.2f}% | Val Acc: {v_acc*100:.2f}%")

        curves_plot = RESULTS_DIR / "mnist_training_curves.png"
        plot_training_curves(history, curves_plot, title="Learning Dynamics: Loss and Accuracy Over Epochs")

    # Load optimal model checkpoint for final evaluation
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Generate confusion matrices for train, validation, and test splits
    metrics = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, split in enumerate(("train", "val", "test")):
        acc, cm = evaluate_split(model, loaders[split], device)
        metrics[split] = {"accuracy": float(acc), "confusion_matrix": cm.cpu().tolist()}
        cm_np = cm.cpu().numpy()

        ax = axes[idx]
        im = ax.imshow(cm_np, cmap="Blues")
        ax.set_title(f"{split.capitalize()} Set (Acc: {acc*100:.2f}%)", fontsize=10)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        thresh = cm_np.max() / 2.0 if cm_np.max() > 0 else 1.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f"{cm_np[i, j]}", ha="center", va="center", color="white" if cm_np[i, j] > thresh else "black")

    fig.suptitle("Classifier Performance: Confusion Matrices", fontsize=12)
    fig.tight_layout()
    cm_plot = RESULTS_DIR / "mnist_confusion_matrices.png"
    fig.savefig(cm_plot, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Save individual atomic confusion matrices for each split
    from utils.metrics import plot_confusion_matrix
    for split in ("train", "val", "test"):
        plot_confusion_matrix(
            np.array(metrics[split]["confusion_matrix"]),
            class_names,
            f"MNIST Classifier - {split.capitalize()} Set ({metrics[split]['accuracy']*100:.2f}%)",
            RESULTS_DIR / f"mnist_{split}_confusion_matrix.png",
        )

    (RESULTS_DIR / "mnist_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ========================================================================
    # Atomic Filter Weights (Saved Individually in results/partA/filters/)
    # ========================================================================
    model.eval()
    with torch.no_grad():
        feat = model.extract_intermediate_features(images.to(device))
    idx1 = list(range(6))
    idx2 = list(range(10))

    all_conv2_idx = list(range(16))

    # Save each trained Conv1 filter individually with actual numerical values
    for ch in idx1:
        save_individual_filter(
            model.conv1.weight[ch, 0].detach().cpu().numpy(),
            FILTERS_DIR / f"trained_conv1_filter_{ch+1}.png",
            f"Trained Conv1 Filter {ch+1} (5x5)",
        )

    # Save all 16 trained Conv2 filters individually with actual numerical values
    for ch in all_conv2_idx:
        save_individual_filter(
            model.conv2.weight[ch].mean(dim=0).detach().cpu().numpy(),
            FILTERS_DIR / f"trained_conv2_filter_{ch+1}.png",
            f"Trained Conv2 Filter {ch+1} (5x5, Mean over in-ch)",
        )

    # ========================================================================
    # Atomic and Compact Feature Maps (Saved in results/partA/feature_maps/)
    # ========================================================================
    for i, cname in enumerate(class_names):
        class_dir = FMAPS_DIR / f"class_{cname}"
        class_dir.mkdir(parents=True, exist_ok=True)

        # Save atomic feature maps for Conv1
        for ch in idx1:
            save_atomic_feature_map(
                feat["relu1"][i, ch].cpu().numpy(),
                class_dir / f"trained_conv1_f{ch+1}.png",
                f"Trained Conv1 Filter {ch+1} (24x24)",
            )

        # Save atomic feature maps for Conv2 (all 16 channels)
        for ch in all_conv2_idx:
            save_atomic_feature_map(
                feat["relu2"][i, ch].cpu().numpy(),
                class_dir / f"trained_conv2_f{ch+1}.png",
                f"Trained Conv2 Filter {ch+1} (8x8)",
            )

        # Save compact, report-friendly grids:
        # Conv1: clean 2x3 grid (6 filters, 6x4 inch)
        plot_compact_feature_grid(
            feat["relu1"][i].cpu(),
            idx1,
            FMAPS_DIR / f"trained_features_{cname}_conv1.png",
            f"Trained Conv1 Feature Maps: Class '{cname}'",
            cols=3,
            stage_name="Conv1 F",
        )
        # Conv2: clean 2x5 grid (10 filters, 8x4 inch)
        plot_compact_feature_grid(
            feat["relu2"][i].cpu(),
            idx2,
            FMAPS_DIR / f"trained_features_{cname}_conv2.png",
            f"Trained Conv2 Feature Maps: Class '{cname}'",
            cols=5,
            stage_name="Conv2 F",
        )

    # Clean up obsolete composite files from earlier iterations
    for obsolete_file in [
        RESULTS_DIR / "trained_filters.png",
        RESULTS_DIR / "stacked_filters.png",
        RESULTS_DIR / "single_filter_convolution.png",
        RESULTS_DIR / "stacked_feature_maps_after_pool.png",
        RESULTS_DIR / "stacked_feature_maps_after_relu.png",
    ]:
        if obsolete_file.exists():
            obsolete_file.unlink()

    for cname in class_names:
        for suffix in ("pool", "relu"):
            old_f = RESULTS_DIR / f"stacked_features_{cname}_{suffix}.png"
            if old_f.exists():
                old_f.unlink()
        old_wide = RESULTS_DIR / f"trained_features_{cname}.png"
        if old_wide.exists():
            old_wide.unlink()

    # ========================================================================
    # Receptive Field Calculation and Input Patch Localization
    # ========================================================================
    rf, jump, start = calculate_receptive_field([(5, 1, 0), (2, 2, 0), (5, 1, 0)])
    conv2_act = feat["conv2"].detach().cpu().numpy()
    patch_info = {}

    for img_idx, cname in enumerate(class_names):
        act = conv2_act[img_idx]
        flat_idx = np.argsort(act.flatten())[::-1][:5]

        top_neurons = []
        for rank, f_i in enumerate(flat_idx):
            ch, r, c = np.unravel_index(f_i, act.shape)
            val = float(act[ch, r, c])
            ymin, xmin, ymax, xmax = neuron_to_bounding_box(r, c, rf, jump, start, image_shape=(28, 28))
            top_neurons.append({
                "rank": rank + 1,
                "channel": int(ch),
                "row": int(r),
                "col": int(c),
                "activation": val,
                "bbox": [ymin, xmin, ymax, xmax],
            })

            # Save individual atomic patch image
            patch_img = images[img_idx, 0, ymin:ymax, xmin:xmax].cpu().numpy()
            save_atomic_feature_map(
                patch_img,
                FMAPS_DIR / f"class_{cname}" / f"rf_patch_rank{rank+1}_ch{ch+1}.png",
                f"Patch N{rank+1} (Ch {ch+1}, Act: {val:.2f})",
                cmap="gray",
            )

        patch_info[cname] = top_neurons

        # Save distinct individual plot for this class showing bounding box and cropped patches
        plot_distinct_receptive_field_patch(
            images[img_idx, 0],
            cname,
            top_neurons,
            RESULTS_DIR / f"receptive_field_patch_{cname}.png",
            f"Receptive Field Localization for Top 5 Activated Neurons: Class '{cname}' (RF: {int(rf)}x{int(rf)})",
            is_rgb=False,
        )

    (RESULTS_DIR / "receptive_field_patch_info.json").write_text(json.dumps(patch_info, indent=2))
    print(f"Saved individual filters to {FILTERS_DIR} and atomic feature maps to {FMAPS_DIR}")
    return model


# ============================================================================
# Main Execution Entrypoint
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convolutional Neural Network Pipeline for Handwritten Digits")
    parser.add_argument("--pipeline", type=str, default="all", choices=["all", "single_filter", "stacked_layers", "classifier"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--momentum", type=float, default=0.0, help="Optional SGD momentum; 0 = plain SGD")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and evaluate using existing checkpoint")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FILTERS_DIR.mkdir(parents=True, exist_ok=True)
    FMAPS_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(42)
    device = get_device()
    images, labels, class_names, paths = sample_representative_images(split="train", data_dir=MNIST_DIR)

    if args.pipeline in ("all", "single_filter"):
        single_filter_convolution_analysis(images, class_names, paths, device=device)
    if args.pipeline in ("all", "stacked_layers"):
        stacked_convolution_analysis(images, class_names, device=device)
    if args.pipeline in ("all", "classifier"):
        loaders, _, _ = load_mnist_dataloaders(batch_size=64, data_dir=MNIST_DIR)
        train_classifier_and_visualize_features(
            loaders, class_names, images, epochs=args.epochs,
            device=device, momentum=args.momentum, eval_only=args.eval_only,
        )


if __name__ == "__main__":
    main()
