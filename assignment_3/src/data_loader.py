import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

# Default data path relative to this file's location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(CURRENT_DIR, "data")


def resolve_data_dir(data_dir=None):
    """
    Resolves data directory path robustly, whether running from project root or src/.
    """
    if data_dir is not None and os.path.exists(data_dir):
        return os.path.abspath(data_dir)
    
    candidates = [
        DEFAULT_DATA_DIR,
        os.path.join(os.getcwd(), "src", "data"),
        os.path.join(os.getcwd(), "data"),
        os.path.join(os.path.dirname(CURRENT_DIR), "src", "data")
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    
    raise FileNotFoundError(f"Could not locate dataset directory in candidates: {candidates}")


def load_mnist_subset(data_dir=None, device="cpu"):
    """
    Loads 28x28 grayscale images for the 5 classes from train, val, and test splits.
    Normalizes pixel values to [0, 1] and flattens each image into a 784-dimensional vector.

    Returns:
        tuple: (
            (X_train, y_train),
            (X_val, y_val),
            (X_test, y_test),
            class_to_idx,
            idx_to_class
        )
    """
    resolved_dir = resolve_data_dir(data_dir)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    splits_data = {}
    class_to_idx = None

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(resolved_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory '{split_dir}' not found in {resolved_dir}")

        dataset = datasets.ImageFolder(split_dir, transform=transform)
        if class_to_idx is None:
            class_to_idx = dataset.class_to_idx
        elif class_to_idx != dataset.class_to_idx:
            raise ValueError(f"Class mapping mismatch between splits: {class_to_idx} vs {dataset.class_to_idx}")

        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        X, y = next(iter(loader))
        splits_data[f"X_{split}"] = X.float().to(device)
        splits_data[f"y_{split}"] = y.long().to(device)

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return (
        (splits_data["X_train"], splits_data["y_train"]),
        (splits_data["X_val"], splits_data["y_val"]),
        (splits_data["X_test"], splits_data["y_test"]),
        class_to_idx,
        idx_to_class
    )


def get_data_tensors(data_dir=None, device="cpu"):

    """
    Convenience wrapper returning preloaded tensors on the specified device.
    """
    return load_mnist_subset(data_dir=data_dir, device=device)


def get_data_loaders(batch_size=1, data_dir=None, device="cpu", shuffle_train=True):
    """
    Returns PyTorch DataLoader instances for train, val, and test splits.
    
    Args:
        batch_size (int): Batch size (e.g. 1 for SGD/SGD-Momentum/NAG/Adam, or total samples for BGD/AdaGrad/RMSProp).
        data_dir (str, optional): Dataset directory path.
        device (str): Device to place tensors on.
        shuffle_train (bool): Whether to shuffle training data.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_to_idx, idx_to_class)
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_to_idx, idx_to_class = get_data_tensors(data_dir, device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    return train_loader, val_loader, test_loader, class_to_idx, idx_to_class


if __name__ == "__main__":
    print("Testing data_loader...")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), c2i, i2c = get_data_tensors()
    print(f"Train samples: {X_tr.shape[0]}, Features: {X_tr.shape[1]}")
    print(f"Val samples:   {X_va.shape[0]}, Features: {X_va.shape[1]}")
    print(f"Test samples:  {X_te.shape[0]}, Features: {X_te.shape[1]}")
    print(f"Class mapping: {c2i}")
    print(f"Pixel min: {X_tr.min().item():.3f}, max: {X_tr.max().item():.3f}")
