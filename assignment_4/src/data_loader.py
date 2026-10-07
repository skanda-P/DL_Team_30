"""Load the Assignment 3 five-class MNIST image-folder splits."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir))
    current = Path(__file__).resolve().parent
    candidates.extend((current / "data", current.parent / "data"))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Dataset directory was not found. Expected train/, val/, and test/ under "
        + str(candidates[0])
    )


def _load_split(split_dir: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing dataset split: {split_dir}")
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Lambda(torch.flatten),
        ]
    )
    dataset = datasets.ImageFolder(str(split_dir), transform=transform)
    if not dataset:
        raise ValueError(f"Dataset split is empty: {split_dir}")
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    features, labels = next(iter(loader))
    return features.float(), labels.long(), dataset.class_to_idx


def load_mnist_subset(
    data_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    dict[str, int],
    dict[int, str],
]:
    root = resolve_data_dir(data_dir)
    loaded = {}
    class_to_idx: dict[str, int] | None = None
    for split in ("train", "val", "test"):
        features, labels, mapping = _load_split(root / split)
        if class_to_idx is None:
            class_to_idx = mapping
        elif mapping != class_to_idx:
            raise ValueError(f"Class mapping mismatch in {split}: {mapping} != {class_to_idx}")
        loaded[split] = (features.to(device), labels.to(device))
    assert class_to_idx is not None
    return (
        loaded["train"],
        loaded["val"],
        loaded["test"],
        class_to_idx,
        {index: name for name, index in class_to_idx.items()},
    )


def get_data_tensors(
    data_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
):
    return load_mnist_subset(data_dir, device)


def get_data_loaders(
    batch_size: int = 128,
    data_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
):
    (x_train, y_train), (x_val, y_val), (x_test, y_test), class_to_idx, idx_to_class = (
        load_mnist_subset(data_dir, device)
    )
    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False),
        DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False),
        class_to_idx,
        idx_to_class,
    )
