"""Custom Convolutional Network Architecture Benchmark on Caltech-101.

This module evaluates 3 custom convolutional architectures across 3 SGD configurations
(3 architectures x 3 SGD configurations = 9 distinct final experimental results):

Architectures:
- ARCH1 (4-layer CNN: 2 Conv + 2 FC)
- ARCH2 (5-layer CNN: 3 Conv + 2 FC)
- ARCH3 (6-layer CNN: 4 Conv + 2 FC)

SGD Configurations:
- normal_sgd: Standard vanilla SGD (lr=0.01, momentum=0.0)
- sgd_momentum: SGD with Classical Momentum (lr=0.01, momentum=0.9)
- sgd_high_lr: SGD with Higher Learning Rate (lr=0.05, momentum=0.0)

Features:
- Dynamic convergence threshold with patience-based Early Stopping (up to max_epochs).
- Separate, individual confusion matrix figures for train, validation, and test splits (no compiled grids).
- Per-architecture training curves (loss and validation accuracy trajectories).
- Layer-wise feature map extraction and analytical receptive field patch localization.
- Systematic naming and comprehensive 9-experiment summary table and visualization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.metrics import confusion_matrix, plot_confusion_matrix
from utils.receptive_field import calculate_receptive_field, neuron_to_bounding_box
from utils.select_images import get_deterministic_images
from utils.seed import set_seed

ROOT = Path(__file__).resolve().parent
CALTECH_DIR = ROOT / "data" / "caltech101_resized"
if not CALTECH_DIR.exists():
    CALTECH_DIR = ROOT / "data" / "caltech101_subset"

RESULTS_DIR = ROOT / "results" / "partB"
MODELS_DIR = ROOT / "models"
TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

SGD_CONFIGS = {
    "normal_sgd": {
        "lr": 0.01,
        "momentum": 0.0,
        "title": "Normal SGD (lr=0.01, momentum=0.0)",
        "color": "#2563EB",
    },
    "sgd_momentum": {
        "lr": 0.01,
        "momentum": 0.9,
        "title": "SGD with Momentum (lr=0.01, momentum=0.9)",
        "color": "#059669",
    },
    "sgd_high_lr": {
        "lr": 0.05,
        "momentum": 0.0,
        "title": "SGD with Higher LR (lr=0.05, momentum=0.0)",
        "color": "#D97706",
    },
}


def get_device() -> torch.device:
    """Detect and return compute device (CUDA GPU if available, else CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""
    print(f"Active compute device: {device}{gpu_name}")
    return device


def initialize_weights(module: nn.Module) -> None:
    """Initialize convolution weights with Kaiming normal and dense weights with Xavier uniform."""
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def load_caltech_dataloaders(batch_size: int = 16, data_dir: Path = CALTECH_DIR):
    """Load train, validation, and test datasets into PyTorch DataLoaders."""
    datasets_by_split = {
        split: datasets.ImageFolder(str(data_dir / split), transform=TRANSFORM)
        for split in ("train", "val", "test")
    }
    mapping = datasets_by_split["train"].class_to_idx
    if any(dataset.class_to_idx != mapping for dataset in datasets_by_split.values()):
        raise ValueError("Class mapping differs between dataset splits")
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets_by_split.items()
    }
    return loaders, mapping, {value: key for key, value in mapping.items()}


def sample_representative_images(split: str = "train", data_dir: Path = CALTECH_DIR):
    """Retrieve exactly one deterministic sample image per class directly from disk."""
    tensors, labels, names, paths = [], [], [], []
    for index, (class_name, path) in enumerate(get_deterministic_images(data_dir, split)):
        with Image.open(path) as image:
            tensors.append(TRANSFORM(image.convert("RGB")))
        labels.append(index)
        names.append(class_name)
        paths.append(path)
    return torch.stack(tensors), labels, names, paths


class HierarchicalCaltechCNN(nn.Module):
    """Custom convolutional network supporting 4-layer, 5-layer, and 6-layer depth configurations."""

    CONFIGS = {
        "arch1": [(8, 11, 4, True), (16, 5, 1, True)],
        "arch2": [(8, 11, 4, True), (16, 5, 1, True), (32, 3, 1, True)],
        "arch3": [(8, 11, 4, True), (16, 5, 1, True), (32, 3, 1, False), (64, 3, 1, True)],
    }

    def __init__(self, architecture: str = "arch1", num_classes: int = 5):
        super().__init__()
        if architecture not in self.CONFIGS:
            raise ValueError(f"Unsupported architecture: {architecture}")
        self.architecture = architecture
        feature_layers: list[nn.Module] = []
        channels = 3
        for output_channels, kernel, stride, pool in self.CONFIGS[architecture]:
            feature_layers.extend([nn.Conv2d(channels, output_channels, kernel, stride), nn.ReLU()])
            if pool:
                feature_layers.append(nn.MaxPool2d(3, 2))
            channels = output_channels
        self.features = nn.Sequential(*feature_layers)
        with torch.no_grad():
            feature_dim = self.features(torch.zeros(1, 3, 224, 224)).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )
        initialize_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction backbone and classification head."""
        return self.classifier(self.features(x))

    def extract_conv_feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract intermediate feature maps after every ReLU non-linearity."""
        outputs, current = [], x
        for layer in self.features:
            current = layer(current)
            if isinstance(layer, nn.ReLU):
                outputs.append(current)
        return outputs


def describe_architecture(model: HierarchicalCaltechCNN, device: torch.device) -> str:
    """Print and return a formatted markdown table detailing per-layer shapes and parameter counts."""
    x = torch.zeros(1, 3, 224, 224, device=device)
    rows = [("Input", tuple(x.shape[1:]), 0)]
    conv_index = 0
    model.eval()
    with torch.no_grad():
        for layer in model.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                conv_index += 1
                rows.append((f"Conv{conv_index} + ReLU", tuple(x.shape[1:]), sum(p.numel() for p in layer.parameters())))
            elif isinstance(layer, nn.MaxPool2d):
                rows.append((f"MaxPool{conv_index}", tuple(x.shape[1:]), 0))
    flat = x.flatten(1).shape[1]
    fc1, fc2 = model.classifier[1], model.classifier[3]
    rows.extend([
        ("Flatten", (flat,), 0),
        ("FC1 + ReLU", (fc1.out_features,), sum(p.numel() for p in fc1.parameters())),
        ("FC2 logits", (fc2.out_features,), sum(p.numel() for p in fc2.parameters())),
    ])
    table = ["| Layer | Output shape | Parameters |", "|---|---:|---:|"]
    table.extend(f"| {name} | {shape} | {params:,} |" for name, shape, params in rows)
    table.append(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    result = "\n".join(table)
    return result


def evaluate_split_with_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 5,
) -> tuple[float, float, torch.Tensor]:
    """Evaluate accuracy, loss, and confusion matrix on a specific data loader."""
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits.append(model(images))
            labels.append(targets)
    all_logits = torch.cat(logits)
    all_labels = torch.cat(labels)
    loss = criterion(all_logits, all_labels).item()
    acc = (all_logits.argmax(dim=-1) == all_labels).float().mean().item()
    cm = confusion_matrix(all_logits.argmax(dim=-1), all_labels, num_classes)
    return acc, loss, cm


def plot_layerwise_feature_maps(fmaps, input_image, path, title, n_maps: int = 8):
    """Plot feature maps from each convolutional layer for a selected image."""
    fig, axes = plt.subplots(len(fmaps) + 1, n_maps, figsize=(2.1 * n_maps, 2.3 * (len(fmaps) + 1)), squeeze=False)
    for column in range(n_maps):
        axes[0, column].axis("off")
    axes[0, 0].imshow(input_image)
    axes[0, 0].set_title("Input image", fontsize=9)
    selected = {}
    for row, feature_map in enumerate(fmaps, start=1):
        channels, height, width = feature_map.shape[1:]
        indices = np.arange(channels) if channels <= n_maps else np.linspace(0, channels - 1, n_maps, dtype=int)
        selected[f"conv{row}"] = [int(index) + 1 for index in indices]
        for column in range(n_maps):
            axes[row, column].axis("off")
            if column < len(indices):
                index = indices[column]
                axes[row, column].imshow(feature_map[0, index].detach().cpu(), cmap="magma")
                axes[row, column].set_title(f"Conv{row} ch{index + 1}\n{height}x{width}", fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return selected


def train_single_model(
    architecture: str,
    config_key: str,
    loaders: dict[str, DataLoader],
    max_epochs: int = 100,
    patience: int = 15,
    min_delta: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Train a single architecture with dynamic convergence / early stopping."""
    cfg = SGD_CONFIGS[config_key]
    lr = cfg["lr"]
    momentum = cfg["momentum"]

    set_seed(42)
    model = HierarchicalCaltechCNN(architecture).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    train_loss_hist = []
    val_loss_hist = []
    val_acc_hist = []

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    converged_epoch = max_epochs
    convergence_reason = f"Completed all {max_epochs} epochs"

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for batch, targets in loaders["train"]:
            batch, targets = batch.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(targets)
            total_samples += len(targets)

        train_loss = running_loss / total_samples
        val_acc, val_loss, _ = evaluate_split_with_loss(model, loaders["val"], criterion, device)

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        improved = False
        if val_acc > best_val_acc + min_delta:
            improved = True
        elif abs(val_acc - best_val_acc) <= min_delta and val_loss < best_val_loss - min_delta:
            improved = True

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if train_loss < 1e-4:
            converged_epoch = epoch
            convergence_reason = f"Converged: training loss dropped below threshold ({train_loss:.6f})"
            break
        if patience_counter >= patience:
            converged_epoch = epoch
            convergence_reason = f"Early stopped: validation metric plateaued for {patience} epochs"
            break

    # Restore checkpoint with highest validation performance
    model.load_state_dict(best_state)

    # Evaluate final metrics on all three dataset splits
    train_acc, train_loss_final, train_cm = evaluate_split_with_loss(model, loaders["train"], criterion, device)
    val_acc, val_loss_final, val_cm = evaluate_split_with_loss(model, loaders["val"], criterion, device)
    test_acc, test_loss_final, test_cm = evaluate_split_with_loss(model, loaders["test"], criterion, device)

    param_count = sum(p.numel() for p in model.parameters())

    print(
        f"  [{config_key.upper()} - {architecture.upper()}] "
        f"Epochs: {converged_epoch} (Best: {best_epoch}) | "
        f"Train: {train_acc * 100:.1f}% | Val: {val_acc * 100:.1f}% | Test: {test_acc * 100:.1f}% | "
        f"({convergence_reason})"
    )

    return {
        "architecture": architecture,
        "config_key": config_key,
        "lr": lr,
        "momentum": momentum,
        "param_count": param_count,
        "total_epochs_run": converged_epoch,
        "best_epoch": best_epoch,
        "convergence_reason": convergence_reason,
        "train_acc": train_acc,
        "train_loss": train_loss_final,
        "train_cm": train_cm.cpu().numpy().tolist(),
        "val_acc": val_acc,
        "val_loss": val_loss_final,
        "val_cm": val_cm.cpu().numpy().tolist(),
        "test_acc": test_acc,
        "test_loss": test_loss_final,
        "test_cm": test_cm.cpu().numpy().tolist(),
        "train_loss_hist": train_loss_hist,
        "val_loss_hist": val_loss_hist,
        "val_acc_hist": val_acc_hist,
        "model": model,
    }


def save_experiment_plots_and_metrics(
    exp: dict,
    class_names: list[str],
    target_dir: Path,
) -> None:
    """Save separate, individual confusion matrices and training curves for an experiment."""
    arch = exp["architecture"]
    cfg_title = SGD_CONFIGS[exp["config_key"]]["title"]

    # 1. Individual Train Confusion Matrix
    plot_confusion_matrix(
        np.array(exp["train_cm"]),
        class_names,
        f"{arch.upper()} Train Confusion Matrix (Acc: {exp['train_acc'] * 100:.1f}%) - {cfg_title}",
        target_dir / f"{arch}_train_confusion_matrix.png",
    )

    # 2. Individual Validation Confusion Matrix
    plot_confusion_matrix(
        np.array(exp["val_cm"]),
        class_names,
        f"{arch.upper()} Val Confusion Matrix (Acc: {exp['val_acc'] * 100:.1f}%) - {cfg_title}",
        target_dir / f"{arch}_val_confusion_matrix.png",
    )

    # 3. Individual Test Confusion Matrix
    plot_confusion_matrix(
        np.array(exp["test_cm"]),
        class_names,
        f"{arch.upper()} Test Confusion Matrix (Acc: {exp['test_acc'] * 100:.1f}%) - {cfg_title}",
        target_dir / f"{arch}_test_confusion_matrix.png",
    )

    # 4. Individual Training Curves (Loss and Validation Accuracy Trajectories)
    fig_curves, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.5))
    epochs_range = range(1, len(exp["train_loss_hist"]) + 1)

    ax_loss.plot(epochs_range, exp["train_loss_hist"], label="Train Loss", color="#2563EB", lw=2)
    ax_loss.plot(epochs_range, exp["val_loss_hist"], label="Val Loss", color="#DC2626", lw=2, linestyle="--")
    ax_loss.axvline(exp["best_epoch"], color="#059669", linestyle=":", label=f"Best Checkpoint (Epoch {exp['best_epoch']})")
    ax_loss.set_title(f"{arch.upper()} Loss Convergence", fontsize=10, fontweight="bold")
    ax_loss.set_xlabel("Epoch", fontsize=9)
    ax_loss.set_ylabel("Cross Entropy Loss", fontsize=9)
    ax_loss.grid(True, linestyle="--", alpha=0.3)
    ax_loss.legend(fontsize=8)

    ax_acc.plot(epochs_range, [v * 100 for v in exp["val_acc_hist"]], label="Val Accuracy", color="#059669", lw=2)
    ax_acc.scatter([exp["best_epoch"]], [exp["val_acc"] * 100], color="#D97706", s=65, zorder=5, label=f"Peak Val ({exp['val_acc'] * 100:.1f}%)")
    ax_acc.set_title(f"{arch.upper()} Validation Accuracy Trajectory", fontsize=10, fontweight="bold")
    ax_acc.set_xlabel("Epoch", fontsize=9)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=9)
    ax_acc.grid(True, linestyle="--", alpha=0.3)
    ax_acc.legend(fontsize=8)

    fig_curves.suptitle(f"{arch.upper()} Optimization Dynamics ({cfg_title})", fontsize=11, fontweight="bold")
    fig_curves.tight_layout()
    fig_curves.savefig(target_dir / f"{arch}_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig_curves)


def run_configuration_pipeline(
    config_key: str,
    loaders: dict[str, DataLoader],
    class_names: list[str],
    images: torch.Tensor,
    max_epochs: int = 100,
    patience: int = 15,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run all 3 architectures under one SGD configuration and save atomic outputs."""
    cfg = SGD_CONFIGS[config_key]
    target_dir = RESULTS_DIR / config_key
    target_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(f"RUNNING CONFIGURATION: {cfg['title']}")
    print(f"Destination folder: {target_dir}")
    print("=" * 75)

    arch_results = {}
    for architecture in ("arch1", "arch2", "arch3"):
        # Save architecture markdown specification
        temp_model = HierarchicalCaltechCNN(architecture).to(device)
        (target_dir / f"{architecture}_architecture.md").write_text(
            describe_architecture(temp_model, device), encoding="utf-8"
        )

        exp = train_single_model(
            architecture, config_key, loaders, max_epochs, patience, 1e-4, device
        )
        arch_results[architecture] = exp

        # Save separate confusion matrices and learning curves for this architecture
        save_experiment_plots_and_metrics(exp, class_names, target_dir)

        # Save checkpoint for this individual architecture and config
        torch.save(
            {
                "architecture": architecture,
                "config_key": config_key,
                "model_state_dict": exp["model"].state_dict(),
                "test_acc": exp["test_acc"],
            },
            MODELS_DIR / f"caltech_{config_key}_{architecture}.pth",
        )

    # Determine best architecture for this configuration based on validation accuracy
    best_arch = max(arch_results, key=lambda a: arch_results[a]["val_acc"])
    best_model = arch_results[best_arch]["model"]
    print(f"[{config_key}] Best architecture: {best_arch.upper()} (Val Acc: {arch_results[best_arch]['val_acc'] * 100:.1f}%, Test Acc: {arch_results[best_arch]['test_acc'] * 100:.1f}%)")

    # Save best model checkpoint for this config
    torch.save(
        {
            "architecture": best_arch,
            "config_key": config_key,
            "model_state_dict": best_model.state_dict(),
            "test_acc": arch_results[best_arch]["test_acc"],
        },
        MODELS_DIR / f"caltech_best_{config_key}.pth",
    )

    # Feature maps of best architecture for this config
    with torch.no_grad():
        feature_maps = best_model.extract_conv_feature_maps(images[:1].to(device))
    plot_layerwise_feature_maps(
        feature_maps,
        np.transpose(images[0].numpy(), (1, 2, 0)),
        target_dir / f"{best_arch}_feature_maps.png",
        f"{best_arch.upper()}: Feature Maps ({cfg['title']})",
    )

    # Receptive field calculation and individual patch localization
    layer_specs = []
    for index, (_, kernel, stride, pool) in enumerate(HierarchicalCaltechCNN.CONFIGS[best_arch]):
        layer_specs.append((kernel, stride, 0))
        if pool and index < len(HierarchicalCaltechCNN.CONFIGS[best_arch]) - 1:
            layer_specs.append((3, 2, 0))
    rf, jump, start = calculate_receptive_field(layer_specs)

    with torch.no_grad():
        activations = best_model.extract_conv_feature_maps(images.to(device))[-1].cpu().numpy()

    patch_info = {}
    for index, class_name in enumerate(class_names):
        image = np.transpose(images[index].numpy(), (1, 2, 0))
        channel, row, col = np.unravel_index(np.argmax(activations[index]), activations[index].shape)
        ymin, xmin, ymax, xmax = neuron_to_bounding_box(row, col, rf, jump, start, (224, 224))
        val = float(activations[index, channel, row, col])
        patch_info[class_name] = {
            "channel": int(channel),
            "row": int(row),
            "col": int(col),
            "activation": val,
            "bbox": [ymin, xmin, ymax, xmax],
        }

        fig_patch, (ax_full, ax_crop) = plt.subplots(1, 2, figsize=(8.5, 4))
        ax_full.imshow(image)
        ax_full.add_patch(
            patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="#ff2222", linewidth=2.5)
        )
        ax_full.set_title(f"Class: '{class_name}'\nMax Act: {val:.2f} (Ch {channel+1})", fontsize=10)
        ax_full.axis("off")

        ax_crop.imshow(image[ymin:ymax, xmin:xmax])
        ax_crop.set_title(
            f"Receptive Field Patch\n[{ymax-ymin}x{xmax-xmin} pixels, RF={int(rf)}x{int(rf)}]",
            fontsize=10,
        )
        ax_crop.axis("off")

        fig_patch.suptitle(f"Receptive Field Patch Localization: Class '{class_name}' ({best_arch.upper()} - {cfg['title']})", fontsize=11)
        fig_patch.tight_layout()
        distinct_patch_path = target_dir / f"custom_cnn_rf_patch_{class_name}.png"
        fig_patch.savefig(distinct_patch_path, dpi=180, bbox_inches="tight")
        plt.close(fig_patch)

    (target_dir / "custom_cnn_patch_info.json").write_text(json.dumps(patch_info, indent=2), encoding="utf-8")

    # Save per-config architecture comparison metrics
    config_metrics = {
        "config": config_key,
        "best_architecture": best_arch,
        "results": {
            arch: {
                "train_acc": exp["train_acc"],
                "train_cm": exp["train_cm"],
                "val_acc": exp["val_acc"],
                "val_cm": exp["val_cm"],
                "test_acc": exp["test_acc"],
                "test_cm": exp["test_cm"],
                "best_epoch": exp["best_epoch"],
                "total_epochs_run": exp["total_epochs_run"],
            }
            for arch, exp in arch_results.items()
        },
    }
    (target_dir / "architecture_comparison_metrics.json").write_text(
        json.dumps(config_metrics, indent=2), encoding="utf-8"
    )

    return arch_results


def generate_consolidated_9_experiments_summary(all_experiments: dict[str, dict[str, dict]]) -> None:
    """Generate consolidated master comparison JSON and visual summary across all 9 experiments."""
    print("\n" + "=" * 75)
    print("GENERATING MASTER 9-EXPERIMENT CONSOLIDATED SUMMARY")
    print("=" * 75)

    experiments_list = []
    for cfg_key, arch_dict in all_experiments.items():
        for arch_key, exp in arch_dict.items():
            experiments_list.append({
                "experiment_id": f"{cfg_key}_{arch_key}",
                "config": cfg_key,
                "architecture": arch_key,
                "learning_rate": exp["lr"],
                "momentum": exp["momentum"],
                "param_count": exp["param_count"],
                "total_epochs_run": exp["total_epochs_run"],
                "best_epoch": exp["best_epoch"],
                "convergence_reason": exp["convergence_reason"],
                "train_accuracy": exp["train_acc"],
                "validation_accuracy": exp["val_acc"],
                "test_accuracy": exp["test_acc"],
                "train_cm": exp["train_cm"],
                "val_cm": exp["val_cm"],
                "test_cm": exp["test_cm"],
            })

    # Find the top overall performing experiment across all 9
    overall_best = max(experiments_list, key=lambda e: (e["validation_accuracy"], e["test_accuracy"]))
    best_exp_obj = all_experiments[overall_best["config"]][overall_best["architecture"]]

    summary_json = {
        "total_experiments": len(experiments_list),
        "overall_best": {
            "experiment_id": overall_best["experiment_id"],
            "config": overall_best["config"],
            "architecture": overall_best["architecture"],
            "validation_accuracy": overall_best["validation_accuracy"],
            "test_accuracy": overall_best["test_accuracy"],
            "train_accuracy": overall_best["train_accuracy"],
            "best_epoch": overall_best["best_epoch"],
        },
        "experiments": experiments_list,
    }

    # Save summary JSON
    summary_json_path = RESULTS_DIR / "custom_cnn_9_experiments_summary.json"
    summary_json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(f"Saved 9-experiment summary JSON to {summary_json_path}")

    # Write root architecture_comparison_metrics.json for part_b_vgg.py
    root_metrics = {
        "best_architecture": overall_best["architecture"],
        "best_config": overall_best["config"],
        "results": {
            arch: {
                "train_acc": all_experiments[overall_best["config"]][arch]["train_acc"],
                "train_cm": all_experiments[overall_best["config"]][arch]["train_cm"],
                "val_acc": all_experiments[overall_best["config"]][arch]["val_acc"],
                "val_cm": all_experiments[overall_best["config"]][arch]["val_cm"],
                "test_acc": all_experiments[overall_best["config"]][arch]["test_acc"],
                "test_cm": all_experiments[overall_best["config"]][arch]["test_cm"],
            }
            for arch in ("arch1", "arch2", "arch3")
        },
    }
    (RESULTS_DIR / "architecture_comparison_metrics.json").write_text(
        json.dumps(root_metrics, indent=2), encoding="utf-8"
    )

    # Save top overall model to models/caltech_best.pth
    torch.save(
        {
            "architecture": overall_best["architecture"],
            "config": overall_best["config"],
            "model_state_dict": best_exp_obj["model"].state_dict(),
            "test_accuracy": overall_best["test_accuracy"],
        },
        MODELS_DIR / "caltech_best.pth",
    )
    print(f"Saved top overall model to {MODELS_DIR / 'caltech_best.pth'} ({overall_best['experiment_id']})")

    # =========================================================================
    # Visual Publication Figure: 9 Experiments Summary (Bar Chart + Table)
    # =========================================================================
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.35)

    ax_bar = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    # Bar chart grouped by architecture (ARCH1, ARCH2, ARCH3) and colored by SGD config
    architectures = ["arch1", "arch2", "arch3"]
    configs = ["normal_sgd", "sgd_momentum", "sgd_high_lr"]
    config_labels = ["Normal SGD\n(lr=0.01)", "SGD + Momentum\n(lr=0.01, m=0.9)", "SGD High LR\n(lr=0.05)"]
    colors = ["#2563EB", "#059669", "#D97706"]

    x = np.arange(len(architectures))
    width = 0.26

    for c_idx, cfg_key in enumerate(configs):
        test_accs = [all_experiments[cfg_key][a]["test_acc"] * 100 for a in architectures]
        val_accs = [all_experiments[cfg_key][a]["val_acc"] * 100 for a in architectures]
        bars = ax_bar.bar(
            x + (c_idx - 1) * width,
            test_accs,
            width,
            label=config_labels[c_idx],
            color=colors[c_idx],
            alpha=0.9,
            edgecolor="#1E293B",
            linewidth=1.2,
        )
        for bar_idx, bar in enumerate(bars):
            h = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 1.2,
                f"{h:.1f}%\n(val:{val_accs[bar_idx]:.0f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(["ARCH1 (4-layer CNN)\n2 Conv + 2 FC", "ARCH2 (5-layer CNN)\n3 Conv + 2 FC", "ARCH3 (6-layer CNN)\n4 Conv + 2 FC"], fontsize=10, fontweight="bold")
    ax_bar.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax_bar.set_ylim(0, 105)
    ax_bar.set_title("Caltech-101 Custom CNN Benchmark: Test Accuracy across All 9 Architecture-SGD Combinations", fontsize=12, fontweight="bold")
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax_bar.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # Performance Table for all 9 experiments
    ax_tbl.axis("off")
    col_labels = [
        "Experiment ID",
        "Architecture",
        "Parameters",
        "SGD Configuration",
        "LR / Momentum",
        "Epochs (Best)",
        "Train Acc",
        "Val Acc",
        "Test Acc",
    ]
    tbl_rows = []
    for exp in experiments_list:
        is_best = exp["experiment_id"] == overall_best["experiment_id"]
        star = " ★" if is_best else ""
        tbl_rows.append([
            f"{exp['experiment_id']}{star}",
            exp["architecture"].upper(),
            f"{exp['param_count']:,}",
            exp["config"],
            f"{exp['learning_rate']} / {exp['momentum']}",
            f"{exp['total_epochs_run']} ({exp['best_epoch']})",
            f"{exp['train_accuracy'] * 100:.1f}%",
            f"{exp['validation_accuracy'] * 100:.1f}%",
            f"{exp['test_accuracy'] * 100:.1f}%",
        ])

    table = ax_tbl.table(
        cellText=tbl_rows,
        colLabels=col_labels,
        colWidths=[0.16, 0.10, 0.10, 0.13, 0.11, 0.12, 0.09, 0.09, 0.10],
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.05, 1.0, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#0F172A")
            cell.set_text_props(color="white", fontweight="bold")
        elif "★" in tbl_rows[row_idx - 1][0]:
            cell.set_facecolor("#FEF3C7")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#F8FAFC" if row_idx % 2 == 0 else "#FFFFFF")

    ax_tbl.set_title("Complete 9-Experiment Empirical Results Summary", fontsize=11, fontweight="bold", pad=12)

    fig_summary_path = RESULTS_DIR / "custom_cnn_9_experiments_summary.png"
    fig.savefig(fig_summary_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved master 9-experiment summary visualization to {fig_summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Custom CNN Architectures Benchmark on Caltech-101 (9 Experiments)")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs per experiment (default: 100)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience in epochs (default: 15)")
    parser.add_argument(
        "--config",
        type=str,
        default="all",
        choices=["all", "normal_sgd", "sgd_momentum", "sgd_high_lr"],
        help="Run all 3 configurations (9 experiments) or a single configuration (3 experiments)",
    )
    args = parser.parse_args()

    set_seed(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    loaders, _, index_to_class = load_caltech_dataloaders(data_dir=CALTECH_DIR)
    class_names = [index_to_class[index] for index in range(len(index_to_class))]
    images, _, _, _ = sample_representative_images(data_dir=CALTECH_DIR)

    if args.config == "all":
        all_experiments = {}
        for config_key in ("normal_sgd", "sgd_momentum", "sgd_high_lr"):
            arch_results = run_configuration_pipeline(
                config_key, loaders, class_names, images, args.max_epochs, args.patience, device
            )
            all_experiments[config_key] = arch_results

        generate_consolidated_9_experiments_summary(all_experiments)
    else:
        arch_results = run_configuration_pipeline(
            args.config, loaders, class_names, images, args.max_epochs, args.patience, device
        )
        # Update root metrics and models for single-config run
        best_arch = max(arch_results, key=lambda a: arch_results[a]["val_acc"])
        root_metrics = {
            "best_architecture": best_arch,
            "best_config": args.config,
            "results": {
                arch: {
                    "train_acc": exp["train_acc"],
                    "train_cm": exp["train_cm"],
                    "val_acc": exp["val_acc"],
                    "val_cm": exp["val_cm"],
                    "test_acc": exp["test_acc"],
                    "test_cm": exp["test_cm"],
                }
                for arch, exp in arch_results.items()
            },
        }
        (RESULTS_DIR / "architecture_comparison_metrics.json").write_text(
            json.dumps(root_metrics, indent=2), encoding="utf-8"
        )
        torch.save(
            {
                "architecture": best_arch,
                "config": args.config,
                "model_state_dict": arch_results[best_arch]["model"].state_dict(),
                "test_accuracy": arch_results[best_arch]["test_acc"],
            },
            MODELS_DIR / "caltech_best.pth",
        )


if __name__ == "__main__":
    main()
