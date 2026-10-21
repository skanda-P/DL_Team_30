"""Analytical receptive field and patch tracing calculations."""
from __future__ import annotations

import math
from typing import Sequence, Tuple


def calculate_receptive_field(
    layers: Sequence[Tuple[int, int, int]],
) -> Tuple[float, float, float]:
    """Calculate the receptive field (RF), effective stride (jump), and top-left center offset.

    Args:
        layers: List of (kernel_size, stride, padding) tuples in sequential order from input to target layer.

    Returns:
        (rf, jump, start):
            rf: receptive field size in input pixels
            jump: effective stride between adjacent neurons in the feature map
            start: center coordinate in the input image corresponding to index 0 of the feature map
    """
    rf = 1.0
    jump = 1.0
    start = 0.5  # coordinate of the center of index 0 in input

    for kernel, stride, padding in layers:
        rf = rf + (kernel - 1) * jump
        start = start + ((kernel - 1) / 2.0 - padding) * jump
        jump = jump * stride

    return rf, jump, start


def neuron_to_bounding_box(
    row: int,
    col: int,
    rf: float,
    jump: float,
    start: float,
    image_shape: Tuple[int, int] = (28, 28),
) -> Tuple[int, int, int, int]:
    """Compute (ymin, xmin, ymax, xmax) bounding box in input space for neuron at (row, col)."""
    center_y = start + row * jump
    center_x = start + col * jump

    half_rf = rf / 2.0
    ymin = int(max(0, math.floor(center_y - half_rf)))
    ymax = int(min(image_shape[0], math.ceil(center_y + half_rf)))
    xmin = int(max(0, math.floor(center_x - half_rf)))
    xmax = int(min(image_shape[1], math.ceil(center_x + half_rf)))

    return ymin, xmin, ymax, xmax
