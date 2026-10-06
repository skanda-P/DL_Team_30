# DataLoader and preprocessing pipeline for the 5-class MNIST dataset (train, validation, and test splits).
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(CURRENT_DIR, "data")


def _is_dataset_dir(path):
    return all(os.path.isdir(os.path.join(path, s)) for s in ("train", "val", "test"))


def resolve_data_dir(data_dir=None):
    
    candidates = [data_dir] if data_dir else []
    candidates += [
        DEFAULT_DATA_DIR,
        os.path.join(os.getcwd(), "src", "data"),
        os.path.join(os.getcwd(), "data"),
        os.path.join(CURRENT_DIR, "..", "..", "assignment_3", "src", "data"),
    ]
    for c in candidates:
        if c and os.path.isdir(c) and _is_dataset_dir(c):
            return os.path.abspath(c)

    raise FileNotFoundError(
        "Could not find a dataset directory containing train/, val/ and test/.\n"
        f"Looked in: {[c for c in candidates if c]}\n"
        "The dataset is the same as Assignment 3: copy the train/, val/, test/ folders "
        "from assignment_3/src/data into assignment_4/src/data (or pass --data_dir)."
    )


def load_mnist_subset(data_dir=None, device="cpu"):
    
    resolved_dir = resolve_data_dir(data_dir)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])

    splits = {}
    class_to_idx = None
    for split in ["train", "val", "test"]:
        dataset = datasets.ImageFolder(os.path.join(resolved_dir, split), transform=transform)
        if class_to_idx is None:
            class_to_idx = dataset.class_to_idx
        elif class_to_idx != dataset.class_to_idx:
            raise ValueError(f"Class mapping mismatch between splits: {class_to_idx} vs {dataset.class_to_idx}")

        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        X, y = next(iter(loader))
        splits[split] = (X.float().to(device), y.long().to(device))

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return splits["train"], splits["val"], splits["test"], class_to_idx, idx_to_class


def get_data_tensors(data_dir=None, device="cpu"):
    return load_mnist_subset(data_dir=data_dir, device=device)


if __name__ == "__main__":
    print("Testing data_loader...")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), c2i, i2c = get_data_tensors()
    print(f"Train samples: {X_tr.shape[0]}, Features: {X_tr.shape[1]}")
    print(f"Val samples:   {X_va.shape[0]}, Features: {X_va.shape[1]}")
    print(f"Test samples:  {X_te.shape[0]}, Features: {X_te.shape[1]}")
    print(f"Class mapping: {c2i}")
    print(f"Pixel min: {X_tr.min().item():.3f}, max: {X_tr.max().item():.3f}")