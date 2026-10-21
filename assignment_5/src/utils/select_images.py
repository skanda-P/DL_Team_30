"""Deterministic image selection across classes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def get_deterministic_images(
    data_dir: str | Path,
    split: str = "train",
) -> List[Tuple[str, Path]]:
    """Return sorted list of (class_name, image_path) with the first image of each class."""
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    results: List[Tuple[str, Path]] = []
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        images = sorted(
            p for p in class_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not images:
            raise FileNotFoundError(f"No valid images found in {class_dir}")
        results.append((class_dir.name, images[0]))
    return results


def save_selected_images_manifest(
    data_dir: str | Path,
    output_json: str | Path,
    split: str = "train",
) -> Dict[str, str]:
    """Save relative paths of the selected images to a JSON file."""
    base_dir = Path(data_dir)
    items = get_deterministic_images(base_dir, split)
    manifest = {cls: str(path.relative_to(base_dir)) for cls, path in items}
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
