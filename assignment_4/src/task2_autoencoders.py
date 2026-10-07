"""Task 2: train shallow and deep autoencoders and report reconstructions."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from data_loader import get_data_tensors
from models import build_autoencoder
from utils.metrics import mse
from utils.plotting import plot_reconstruction_grid
from utils.training_config import (
    TASK2_BATCH_SIZE,
    TASK2_LEARNING_RATE,
    TASK2_MIN_DELTA,
    TASK2_PATIENCE,
)


BOTTLENECKS = (32, 64, 128, 256)
ARCHITECTURES = ("ae1", "ae3")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def reconstruction_error(
    model: nn.Module, features: torch.Tensor, batch_size: int
) -> float:
    model.eval()
    outputs = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(features), batch_size=batch_size):
            outputs.append(model(batch).cpu())
    return float(mse(features.cpu().numpy(), torch.cat(outputs).numpy()))


def reconstruct(
    model: nn.Module, features: torch.Tensor, batch_size: int
) -> torch.Tensor:
    model.eval()
    outputs = []
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(features), batch_size=batch_size):
            outputs.append(model(batch).cpu())
    return torch.cat(outputs)


def train_autoencoder(
    model: nn.Module,
    train_features: torch.Tensor,
    val_features: torch.Tensor,
    batch_size: int,
    learning_rate: float,
    patience: int,
    min_delta: float,
) -> list[dict[str, float]]:
    loader = DataLoader(TensorDataset(train_features), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    stale_epochs = 0
    history = []

    epoch = 0
    while True:
        epoch += 1
        model.train()
        train_total = 0.0
        sample_count = 0
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(batch)
            sample_count += len(batch)

        val_error = reconstruction_error(model, val_features, batch_size)
        train_error = train_total / sample_count
        history.append({"epoch": epoch, "train_mse": train_error, "val_mse": val_error})
        print(
            f"epoch {epoch:03d} | train MSE {train_error:.6f} | "
            f"validation MSE {val_error:.6f}"
        )

        if val_error < best_val - min_delta:
            best_val = val_error
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def save_history(history: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_mse", "val_mse"))
        writer.writeheader()
        writer.writerows(history)


def run_task2(
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    batch_size: int = TASK2_BATCH_SIZE,
    learning_rate: float = TASK2_LEARNING_RATE,
    patience: int = TASK2_PATIENCE,
    min_delta: float = TASK2_MIN_DELTA,
    seed: int = 42,
    device: str | None = None,
) -> list[dict[str, object]]:
    set_seed(seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_root = Path(results_dir) if results_dir else CURRENT_DIR / "results" / "task2_autoencoders"
    output_root.mkdir(parents=True, exist_ok=True)

    (train, train_labels), (val, val_labels), (test, test_labels), _, _ = get_data_tensors(
        data_dir, selected_device
    )
    results = []

    for architecture in ARCHITECTURES:
        for bottleneck in BOTTLENECKS:
            name = f"{architecture}_bottleneck_{bottleneck}"
            run_dir = output_root / name
            run_dir.mkdir(parents=True, exist_ok=True)
            model = build_autoencoder(architecture, bottleneck).to(selected_device)
            history = train_autoencoder(
                model, train, val, batch_size, learning_rate, patience, min_delta,
            )
            train_error = reconstruction_error(model, train, batch_size)
            val_error = reconstruction_error(model, val, batch_size)
            test_error = reconstruction_error(model, test, batch_size)
            reconstructed = {
                "train": reconstruct(model, train, batch_size),
                "val": reconstruct(model, val, batch_size),
                "test": reconstruct(model, test, batch_size),
            }
            torch.save(model.state_dict(), run_dir / "model.pt")
            save_history(history, run_dir / "loss_history.csv")
            plot_reconstruction_grid(
                train.cpu(), reconstructed["train"], train_labels.cpu(),
                run_dir / "train_reconstructions.png", f"{name} - train",
            )
            plot_reconstruction_grid(
                val.cpu(), reconstructed["val"], val_labels.cpu(),
                run_dir / "val_reconstructions.png", f"{name} - validation",
            )
            plot_reconstruction_grid(
                test.cpu(), reconstructed["test"], test_labels.cpu(),
                run_dir / "test_reconstructions.png", f"{name} - test",
            )
            result = {
                "architecture": architecture,
                "bottleneck": bottleneck,
                "epochs": len(history),
                "train_mse": train_error,
                "validation_mse": val_error,
                "test_mse": test_error,
            }
            results.append(result)
            print(json.dumps(result))

    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=TASK2_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=TASK2_LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=TASK2_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=TASK2_MIN_DELTA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_task2(**vars(args))
