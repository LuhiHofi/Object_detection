"""Bounding-box helpers for YOLOv1 resize (448×448).

Classic YOLO resizes by stretching (no letterbox). Boxes must be scaled
into network space for training and scaled back for drawing / eval.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np

from paper import INPUT_SIZE as YOLO_INPUT_SIZE

def resize_scale(
    orig_w: int,
    orig_h: int,
    input_size: int = YOLO_INPUT_SIZE,
) -> tuple[float, float]:
    """Return (sx, sy) such that x_net = x_orig * sx, y_net = y_orig * sy."""
    return input_size / orig_w, input_size / orig_h


def boxes_to_input(
    boxes: np.ndarray | Sequence[Sequence[float]],
    orig_w: int,
    orig_h: int,
    input_size: int = YOLO_INPUT_SIZE,
) -> np.ndarray:
    """Map boxes from original image pixels → network input pixels.

    boxes: (N, 4) as xyxy [x1, y1, x2, y2] in original resolution.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    sx, sy = resize_scale(orig_w, orig_h, input_size)
    out = boxes.copy()
    out[:, [0, 2]] *= sx
    out[:, [1, 3]] *= sy
    return out


def boxes_to_original(
    boxes: np.ndarray | Sequence[Sequence[float]],
    orig_w: int,
    orig_h: int,
    input_size: int = YOLO_INPUT_SIZE,
) -> np.ndarray:
    """Map boxes from network input pixels → original image pixels.

    boxes: (N, 4) as xyxy [x1, y1, x2, y2] in 448×448 (or input_size) space.
    Use this after inference before drawing or computing IoU on dataset labels.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    sx, sy = resize_scale(orig_w, orig_h, input_size)
    out = boxes.copy()
    out[:, [0, 2]] /= sx
    out[:, [1, 3]] /= sy
    return out


def clip_boxes_xyxy(
    boxes: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Clip xyxy boxes to image bounds [0, width] × [0, height]."""
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    return boxes


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """xyxy → (cx, cy, w, h)."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    x1, y1, x2, y2 = boxes.T
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(cx, cy, w, h) → xyxy."""
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    cx, cy, w, h = boxes.T
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
