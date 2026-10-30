"""Dataset verification and optional Caltech-101 224x224 resizing utility.

Usage:
    python verify_dataset.py
    python verify_dataset.py --resize_caltech
"""

import argparse
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

MNIST_CLASSES = ["3", "4", "5", "7", "8"]
CALTECH_CLASSES = ["butterfly", "car_side", "grand_piano", "scorpion", "starfish"]
SPLITS = ["train", "val", "test"]


def verify_mnist() -> bool:
    mnist_dir = DATA_DIR / "mnist_subset"
    print(f"\n[1/2] Verifying MNIST Subset at: {mnist_dir}")
    if not mnist_dir.exists():
        print(f"  [MISSING] Directory not found: {mnist_dir}")
        return False

    all_ok = True
    for split in SPLITS:
        split_dir = mnist_dir / split
        if not split_dir.exists():
            print(f"  [MISSING] Split directory: {split_dir}")
            all_ok = False
            continue
        found_classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
        missing = set(MNIST_CLASSES) - set(found_classes)
        if missing:
            print(f"  [ERROR] {split} is missing classes: {sorted(missing)}")
            all_ok = False
        else:
            total_images = sum(len(list((split_dir / cls).glob("*.*"))) for cls in MNIST_CLASSES)
            print(f"  [OK] {split}: all 5 classes present ({total_images} total images)")
    return all_ok


def verify_caltech() -> bool:
    caltech_subset = DATA_DIR / "caltech101_subset"
    caltech_resized = DATA_DIR / "caltech101_resized"
    target_dir = caltech_resized if caltech_resized.exists() else caltech_subset

    print(f"\n[2/2] Verifying Caltech-101 Subset at: {target_dir}")
    if not target_dir.exists():
        print(f"  [MISSING] Neither {caltech_resized} nor {caltech_subset} exists.")
        return False

    all_ok = True
    for split in SPLITS:
        split_dir = target_dir / split
        if not split_dir.exists():
            print(f"  [MISSING] Split directory: {split_dir}")
            all_ok = False
            continue
        found_classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
        missing = set(CALTECH_CLASSES) - set(found_classes)
        if missing:
            print(f"  [ERROR] {split} is missing classes: {sorted(missing)}")
            all_ok = False
        else:
            total_images = sum(len(list((split_dir / cls).glob("*.*"))) for cls in CALTECH_CLASSES)
            print(f"  [OK] {split}: all 5 classes present ({total_images} total images)")
    return all_ok


def resize_caltech():
    source = DATA_DIR / "caltech101_subset"
    dest = DATA_DIR / "caltech101_resized"
    print(f"\n[Resizing] Resizing Caltech-101 from {source} to {dest} (224x224 RGB)...")
    if not source.exists():
        print(f"  [ERROR] Source folder does not exist: {source}")
        return

    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for split in SPLITS:
        for cls in CALTECH_CLASSES:
            src_cls_dir = source / split / cls
            if not src_cls_dir.exists():
                continue
            dst_cls_dir = dest / split / cls
            dst_cls_dir.mkdir(parents=True, exist_ok=True)
            for img_path in src_cls_dir.glob("*.*"):
                try:
                    with Image.open(img_path) as img:
                        resized = img.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                        target_path = dst_cls_dir / f"{img_path.stem}.jpg"
                        resized.save(target_path, "JPEG", quality=95)
                        count += 1
                except Exception as e:
                    print(f"  [WARNING] Failed to resize {img_path}: {e}")
    print(f"  [DONE] Successfully resized {count} images to 224x224 in {dest}")


def main():
    parser = argparse.ArgumentParser(description="Dataset verification and Caltech-101 resizer")
    parser.add_argument("--resize_caltech", action="store_true", help="Resize Caltech-101 images to 224x224")
    args = parser.parse_args()

    if args.resize_caltech:
        resize_caltech()

    m_ok = verify_mnist()
    c_ok = verify_caltech()

    print("\n" + "=" * 50)
    if m_ok and c_ok:
        print("ALL DATASET CHECKS PASSED! Ready to run experiments.")
    else:
        print("SOME CHECKS FAILED! Please inspect missing directories above.")
    print("=" * 50)


if __name__ == "__main__":
    main()

