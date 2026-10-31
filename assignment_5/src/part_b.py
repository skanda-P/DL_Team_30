"""Deep Learning - Assignment 5 (Part B, Question 1)
Caltech-101 Custom CNN Architecture Benchmarking using the Adam Optimizer.

This module evaluates three convolutional neural network architectures on a 5-class subset
of the Caltech-101 dataset (butterfly, car_side, grand_piano, scorpion, starfish):
- Architecture-1 (4-layer CNN): 2 Conv blocks + 2 FC layers
- Architecture-2 (5-layer CNN): 3 Conv blocks + 2 FC layers
- Architecture-3 (6-layer CNN): 4 Conv blocks + 2 FC layers

Optimization is driven by the Adam optimizer (Kingma & Ba, 2014) with early stopping
and checkpoint restoration based on validation accuracy.

Outputs include:
- Markdown architectural descriptions detailing output shapes and parameter counts per layer
- Independent train, validation, and test confusion matrices for all three architectures
- Optimization convergence curves (loss and validation accuracy trajectories)
- Feature map visual representations across convolutional layers for the top model
- Analytical receptive field patch localization with bounding boxes on representative samples
- Consolidated comparative summary (visual publication chart, results table, and JSON logs)
- Synchronization with VGG19 comparative test performance
"""

import argparse
import json
from pathlib import Path
from PIL import Image

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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
CUSTOM_RESULTS_DIR = RESULTS_DIR / "custom_cnn"
MODELS_DIR = ROOT / "models"
TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
TRAIN_TRANSFORM = transforms.Compose([
    #transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    #transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
])


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
        split: datasets.ImageFolder(
            str(data_dir / split),
            transform=TRAIN_TRANSFORM if split == "train" else TRANSFORM,
        )
        for split in ("train", "val", "test")
    }
    # Non-augmented copy of the training set, used only for reporting train accuracy / confusion matrix
    train_eval_dataset = datasets.ImageFolder(str(data_dir / "train"), transform=TRANSFORM)
    mapping = datasets_by_split["train"].class_to_idx
    if any(dataset.class_to_idx != mapping for dataset in datasets_by_split.values()):
        raise ValueError("Class mapping differs between dataset splits")

    loaders = {
        split: DataLoader(
            datasets_by_split[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        for split in ("train", "val", "test")
    }
    loaders["train_eval"] = DataLoader(train_eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    class_to_index = datasets_by_split["train"].class_to_idx
    index_to_class = {index: class_name for class_name, index in class_to_index.items()}
    return loaders, class_to_index, index_to_class


def sample_representative_images(split: str = "train", data_dir: Path = CALTECH_DIR):
    """Retrieve exactly one deterministic sample image per class directly from disk."""
    tensors, labels, names, paths = [], [], [], []
    for index, (class_name, path) in enumerate(get_deterministic_images(data_dir, split)):
        with Image.open(path) as image:
            tensors.append(TRANSFORM(image.convert("RGB")))
        labels.append(index)
        names.append(class_name)
        paths.append(path)
    return torch.stack(tensors), torch.tensor(labels), names, paths


class HierarchicalCaltechCNN(nn.Module):
    """Custom convolutional network supporting 4-layer, 5-layer, and 6-layer depth configurations."""

    CONFIGS = {
        "arch1": [(8, 11, 4, True), (16, 5, 1, True)],
        "arch2": [(8, 11, 4, True), (16, 5, 1, True), (32, 3, 1, True)],
        "arch3": [(8, 11, 4, True), (16, 5, 1, True), (32, 3, 1, False), (64, 3, 1, True)],
    }

    def __init__(self, architecture: str = "arch1", num_classes: int = 5, dropout: float = 0.5):
        super().__init__()
        if architecture not in self.CONFIGS:
            raise ValueError(f"Unsupported architecture: {architecture}")
        self.architecture = architecture
        self.dropout = dropout
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
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
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
    fc1, fc2 = model.classifier[1], model.classifier[4]
    rows.extend([
        ("Flatten", (flat,), 0),
        ("FC1 + ReLU", (fc1.out_features,), sum(p.numel() for p in fc1.parameters())),
        (f"Dropout (p={model.dropout:g})", (fc1.out_features,), 0),
        ("FC2 logits", (fc2.out_features,), sum(p.numel() for p in fc2.parameters())),
    ])
    table = ["| Layer | Output shape | Parameters |", "|---|---:|---:|"]
    table.extend(f"| {name} | {shape} | {params:,} |" for name, shape, params in rows)
    table.append(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    return "\n".join(table)


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
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return selected


def train_single_model(
    architecture: str,
    loaders: dict[str, DataLoader],
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 15,
    min_delta: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    dropout: float = 0.5,
) -> dict:
    """Train a single architecture using the Adam optimizer with early stopping."""
    set_seed(42)
    model = HierarchicalCaltechCNN(architecture, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    train_acc, train_loss_final, train_cm = evaluate_split_with_loss(model, loaders["train_eval"], criterion, device)
    val_acc, val_loss_final, val_cm = evaluate_split_with_loss(model, loaders["val"], criterion, device)
    test_acc, test_loss_final, test_cm = evaluate_split_with_loss(model, loaders["test"], criterion, device)

    param_count = sum(p.numel() for p in model.parameters())

    print(
        f"  [ADAM - {architecture.upper()}] "
        f"Epochs: {converged_epoch} (Best: {best_epoch}) | "
        f"Train: {train_acc * 100:.1f}% | Val: {val_acc * 100:.1f}% | Test: {test_acc * 100:.1f}% | "
        f"({convergence_reason})"
    )

    return {
        "architecture": architecture,
        "optimizer": "Adam",
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "param_count": param_count,
        "total_epochs_run": converged_epoch,
        "best_epoch": best_epoch,
        "convergence_reason": convergence_reason,
        "train_acc": train_acc,
        "train_loss": train_loss_final,
        "train_cm": train_cm.cpu().numpy().tolist(),
        "val_acc": val_acc,
        "val_loss": val_loss_final,
        "best_val_loss": best_val_loss,
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
    """Save separate, individual confusion matrices and training curves for an architecture."""
    arch = exp["architecture"]
    title_suffix = f"Adam Optimizer (lr={exp['lr']})"

    # 1. Individual Train Confusion Matrix
    plot_confusion_matrix(
        np.array(exp["train_cm"]),
        class_names,
        f"{arch.upper()} Train Confusion Matrix (Acc: {exp['train_acc'] * 100:.1f}%) - {title_suffix}",
        target_dir / f"{arch}_train_confusion_matrix.png",
    )

    # 2. Individual Validation Confusion Matrix
    plot_confusion_matrix(
        np.array(exp["val_cm"]),
        class_names,
        f"{arch.upper()} Val Confusion Matrix (Acc: {exp['val_acc'] * 100:.1f}%) - {title_suffix}",
        target_dir / f"{arch}_val_confusion_matrix.png",
    )

    # 3. Individual Test Confusion Matrix
    plot_confusion_matrix(
        np.array(exp["test_cm"]),
        class_names,
        f"{arch.upper()} Test Confusion Matrix (Acc: {exp['test_acc'] * 100:.1f}%) - {title_suffix}",
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

    fig_curves.suptitle(f"{arch.upper()} Optimization Dynamics ({title_suffix})", fontsize=11, fontweight="bold")
    fig_curves.tight_layout()
    fig_curves.savefig(target_dir / f"{arch}_training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig_curves)


def generate_consolidated_summary(arch_results: dict[str, dict], target_dir: Path) -> None:
    """Generate consolidated master comparison JSON and visual summary across the three architectures."""
    print("\n" + "=" * 75)
    print("GENERATING CUSTOM CNN CONSOLIDATED SUMMARY (ADAM OPTIMIZER)")
    print("=" * 75)

    architectures = ["arch1", "arch2", "arch3"]
    arch_labels = [
        "ARCH1 (4-layer CNN)\n2 Conv + 2 FC",
        "ARCH2 (5-layer CNN)\n3 Conv + 2 FC",
        "ARCH3 (6-layer CNN)\n4 Conv + 2 FC",
    ]

    # Best architecture selected by validation accuracy (per assignment Part B Q1b)
    best_arch = max(
        arch_results,
        key=lambda a: (arch_results[a]["val_acc"], -arch_results[a]["val_loss"], -arch_results[a]["param_count"]),
    )
    overall_best = arch_results[best_arch]

    summary_json = {
        "optimizer": "Adam",
        "learning_rate": overall_best["lr"],
        "weight_decay": overall_best["weight_decay"],
        "best_architecture": best_arch,
        "overall_best": {
            "architecture": best_arch,
            "train_accuracy": overall_best["train_acc"],
            "validation_accuracy": overall_best["val_acc"],
            "test_accuracy": overall_best["test_acc"],
            "val_loss": overall_best["val_loss"],
            "best_val_loss": overall_best["best_val_loss"],
            "best_epoch": overall_best["best_epoch"],
            "param_count": overall_best["param_count"],
        },
        "architectures": {
            arch: {
                "param_count": exp["param_count"],
                "total_epochs_run": exp["total_epochs_run"],
                "best_epoch": exp["best_epoch"],
                "convergence_reason": exp["convergence_reason"],
                "train_accuracy": exp["train_acc"],
                "validation_accuracy": exp["val_acc"],
                "test_accuracy": exp["test_acc"],
                "val_loss": exp["val_loss"],
                "best_val_loss": exp["best_val_loss"],
                "train_cm": exp["train_cm"],
                "val_cm": exp["val_cm"],
                "test_cm": exp["test_cm"],
            }
            for arch, exp in arch_results.items()
        },
    }

    # Save summary JSON
    summary_json_path = target_dir / "custom_cnn_architectures_summary.json"
    summary_json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(f"Saved custom CNN summary JSON to {summary_json_path}")

    # Standard metrics schema for comparison
    root_metrics = {
        "best_architecture": best_arch,
        "optimizer": "Adam",
        "results": {
            arch: {
                "train_acc": exp["train_acc"],
                "train_cm": exp["train_cm"],
                "val_acc": exp["val_acc"],
                "val_loss": exp["val_loss"],
                "best_val_loss": exp["best_val_loss"],
                "val_cm": exp["val_cm"],
                "test_acc": exp["test_acc"],
                "test_loss": exp["test_loss"],
                "test_cm": exp["test_cm"],
                "best_epoch": exp["best_epoch"],
                "total_epochs_run": exp["total_epochs_run"],
                "param_count": exp["param_count"],
            }
            for arch, exp in arch_results.items()
        },
    }
    (target_dir / "architecture_comparison_metrics.json").write_text(
        json.dumps(root_metrics, indent=2), encoding="utf-8"
    )

    # Save top overall model to models/caltech_best.pth
    torch.save(
        {
            "architecture": best_arch,
            "optimizer": "Adam",
            "model_state_dict": overall_best["model"].state_dict(),
            "test_accuracy": overall_best["test_acc"],
            "val_accuracy": overall_best["val_acc"],
        },
        MODELS_DIR / "caltech_best.pth",
    )
    print(f"Saved top overall model to {MODELS_DIR / 'caltech_best.pth'} ({best_arch.upper()})")

    # =========================================================================
    # Visual Publication Figure: Architecture Summary (Grouped Bar Chart + Table)
    # =========================================================================
    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 0.9], hspace=0.35)

    ax_bar = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    x = np.arange(len(architectures))
    width = 0.24

    train_accs = [arch_results[a]["train_acc"] * 100 for a in architectures]
    val_accs = [arch_results[a]["val_acc"] * 100 for a in architectures]
    test_accs = [arch_results[a]["test_acc"] * 100 for a in architectures]

    bars_train = ax_bar.bar(x - width, train_accs, width, label="Train Accuracy", color="#2563EB", alpha=0.9, edgecolor="#1E293B")
    bars_val = ax_bar.bar(x, val_accs, width, label="Val Accuracy", color="#059669", alpha=0.9, edgecolor="#1E293B")
    bars_test = ax_bar.bar(x + width, test_accs, width, label="Test Accuracy", color="#D97706", alpha=0.9, edgecolor="#1E293B")

    for bars in (bars_train, bars_val, bars_test):
        for bar in bars:
            h = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 1.2,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(arch_labels, fontsize=10, fontweight="bold")
    ax_bar.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax_bar.set_ylim(0, 110)
    ax_bar.set_title("Caltech-101 Custom CNN Benchmark: Architecture Comparison with Adam Optimizer", fontsize=12, fontweight="bold")
    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax_bar.legend(loc="upper right", fontsize=9.5, framealpha=0.95)

    # Performance Table for the 3 architectures
    ax_tbl.axis("off")
    col_labels = [
        "Architecture",
        "Parameters",
        "Optimizer",
        "Learning Rate",
        "Epochs (Best)",
        "Train Acc",
        "Val Acc",
        "Test Acc",
        "Status",
    ]
    tbl_rows = []
    for arch in architectures:
        exp = arch_results[arch]
        is_best = arch == best_arch
        star = " ★ (Best)" if is_best else ""
        tbl_rows.append([
            f"{arch.upper()}{star}",
            f"{exp['param_count']:,}",
            "Adam",
            f"{exp['lr']}",
            f"{exp['total_epochs_run']} ({exp['best_epoch']})",
            f"{exp['train_acc'] * 100:.1f}%",
            f"{exp['val_acc'] * 100:.1f}%",
            f"{exp['test_acc'] * 100:.1f}%",
            "Selected Model" if is_best else "Evaluated",
        ])

    table = ax_tbl.table(
        cellText=tbl_rows,
        colLabels=col_labels,
        colWidths=[0.16, 0.11, 0.10, 0.11, 0.12, 0.10, 0.10, 0.10, 0.10],
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.10, 1.0, 0.80],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#0F172A")
            cell.set_text_props(color="white", fontweight="bold")
        elif "★" in tbl_rows[row_idx - 1][0]:
            cell.set_facecolor("#FEF3C7")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#F8FAFC" if row_idx % 2 == 0 else "#FFFFFF")

    ax_tbl.set_title("Empirical Results Summary across Architectures (Adam Optimizer)", fontsize=11, fontweight="bold", pad=12)

    fig_summary_path = target_dir / "custom_cnn_architectures_summary.png"
    fig.savefig(fig_summary_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved custom CNN architecture summary visualization to {fig_summary_path}")




def main():
    parser = argparse.ArgumentParser(description="Custom CNN Architectures Benchmark on Caltech-101 using Adam Optimizer")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs per architecture (default: 100)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience in epochs (default: 15)")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Adam weight decay (default: 1e-4)")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability after FC1 (default: 0.5; 0 disables)")
    args = parser.parse_args()

    # Keep each dropout setting in its own folders so runs do not overwrite each other
    global CUSTOM_RESULTS_DIR, MODELS_DIR
    run_tag = f"dropout{args.dropout:g}"
    CUSTOM_RESULTS_DIR = RESULTS_DIR / "custom_cnn" / run_tag
    MODELS_DIR = ROOT / "models" / run_tag

    set_seed(42)
    CUSTOM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    loaders, _, index_to_class = load_caltech_dataloaders(data_dir=CALTECH_DIR)
    class_names = [index_to_class[index] for index in range(len(index_to_class))]
    images, _, _, _ = sample_representative_images(data_dir=CALTECH_DIR)

    print("\n" + "=" * 75)
    print(f"RUNNING CALTECH-101 CUSTOM CNN PIPELINE (ADAM OPTIMIZER: lr={args.lr}, dropout={args.dropout})")
    print(f"Target Directory: {CUSTOM_RESULTS_DIR}")
    print("=" * 75)

    arch_results = {}
    for architecture in ("arch1", "arch2", "arch3"):
        # Save architecture markdown specification
        temp_model = HierarchicalCaltechCNN(architecture, dropout=args.dropout).to(device)
        (CUSTOM_RESULTS_DIR / f"{architecture}_architecture.md").write_text(
            describe_architecture(temp_model, device), encoding="utf-8"
        )

        exp = train_single_model(
            architecture, loaders, args.lr, args.weight_decay, args.max_epochs, args.patience, 1e-4, device,
            dropout=args.dropout,
        )
        arch_results[architecture] = exp

        # Save separate confusion matrices and learning curves for this architecture
        save_experiment_plots_and_metrics(exp, class_names, CUSTOM_RESULTS_DIR)

        # Save checkpoint for this individual architecture
        torch.save(
            {
                "architecture": architecture,
                "optimizer": "Adam",
                "dropout": args.dropout,
                "model_state_dict": exp["model"].state_dict(),
                "test_acc": exp["test_acc"],
                "val_acc": exp["val_acc"],
                "val_loss": exp["val_loss"],
                "best_val_loss": exp["best_val_loss"],
            },
            MODELS_DIR / f"caltech_{architecture}.pth",
        )

    # Determine best architecture based on validation accuracy and validation loss
    best_arch = max(
        arch_results,
        key=lambda a: (arch_results[a]["val_acc"], -arch_results[a]["val_loss"], -arch_results[a]["param_count"]),
    )
    best_model = arch_results[best_arch]["model"]
    print(f"\nBest architecture: {best_arch.upper()} (Val Acc: {arch_results[best_arch]['val_acc'] * 100:.1f}%, Val Loss: {arch_results[best_arch]['val_loss']:.4f}, Test Acc: {arch_results[best_arch]['test_acc'] * 100:.1f}%)")

    # Feature maps of best architecture (Q1c)
    with torch.no_grad():
        feature_maps = best_model.extract_conv_feature_maps(images[:1].to(device))
    plot_layerwise_feature_maps(
        feature_maps,
        np.transpose(images[0].numpy(), (1, 2, 0)),
        CUSTOM_RESULTS_DIR / f"{best_arch}_feature_maps.png",
        f"{best_arch.upper()}: Feature Maps (Adam Optimizer)",
    )

    # Receptive field calculation and individual patch localization (Q1d)
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

        fig_patch.suptitle(f"Receptive Field Patch Localization: Class '{class_name}' ({best_arch.upper()} - Adam Optimizer)", fontsize=11, fontweight="bold")
        fig_patch.tight_layout()
        distinct_patch_path = CUSTOM_RESULTS_DIR / f"custom_cnn_rf_patch_{class_name}.png"
        fig_patch.savefig(distinct_patch_path, dpi=180, bbox_inches="tight")
        plt.close(fig_patch)

    (CUSTOM_RESULTS_DIR / "custom_cnn_patch_info.json").write_text(json.dumps(patch_info, indent=2), encoding="utf-8")

    # Generate consolidated comparison summary
    generate_consolidated_summary(arch_results, CUSTOM_RESULTS_DIR)
    print("\nPart B Question 1 benchmarking completed successfully.")


if __name__ == "__main__":
    main()