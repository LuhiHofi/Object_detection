"""Bounding-box helpers."""

from typing import Sequence
import numpy as np
import torch

from paper import INPUT_SIZE as YOLO_INPUT_SIZE

def get_boxes_scores(pred: torch.Tensor, num_classes: int, num_boxes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the boxes and scores from the prediction tensor.

    The score is the class-specific confidence: Pr(class) * IoU.
    """
    N, S, _, _ = pred.shape
    C, B = num_classes, num_boxes

    cls = pred[..., :C].clamp(0, 1)                                  # (N, S, S, C)
    boxes = pred[..., C:].reshape(N, S, S, B, 5)
    conf, xywh = boxes[..., 0], boxes[..., 1:]

    rows = torch.arange(S, device=pred.device).view(1, S, 1, 1)
    cols = torch.arange(S, device=pred.device).view(1, 1, S, 1)
    cx = (cols + xywh[..., 0]) / S
    cy = (rows + xywh[..., 1]) / S
    w, h = xywh[..., 2] ** 2, xywh[..., 3] ** 2  # network predicts √w, √h

    xyxy = torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)  # (N,S,S,B,4)
    scores = conf.unsqueeze(-1) * cls.unsqueeze(3)                                # (N,S,S,B,C)
    return xyxy.reshape(N, -1, 4), scores.reshape(N, -1, C)

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


def kmeans_anchors(wh: np.ndarray, k: int, iters: int = 100, seed: int = 0) -> np.ndarray:
    """YOLOv2 "dimension clusters": k-means on box (w, h) with distance 1 - IoU.

    Returns (k, 2) priors sorted by area, in the same units as ``wh``.
    """
    wh = np.asarray(wh, dtype=np.float32)
    rng = np.random.default_rng(seed)
    centers = wh[rng.choice(len(wh), k, replace=False)]
    for _ in range(iters):
        inter = np.minimum(wh[:, None, 0], centers[None, :, 0]) * np.minimum(wh[:, None, 1], centers[None, :, 1])
        iou = inter / (wh.prod(1)[:, None] + centers.prod(1)[None, :] - inter)
        assign = iou.argmax(1)
        new = np.stack([wh[assign == c].mean(0) if (assign == c).any() else centers[c] for c in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return centers[np.argsort(centers.prod(1))]

def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between (N, 4) and (M, 4) boxes → (N, M)."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-6)


def iou_elementwise(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between matching boxes of two equally shaped (..., 4) xyxy tensors."""
    lt = torch.max(a[..., :2], b[..., :2])
    rb = torch.min(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda x: (x[..., 2] - x[..., 0]).clamp(min=0) * (x[..., 3] - x[..., 1]).clamp(min=0)
    return inter / (area(a) + area(b) - inter).clamp(min=1e-6)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Greedy non-maximum suppression to remove overlapping boxes."""
    keep: list[int] = []
    order = scores.argsort(descending=True)
    while order.numel() > 0:
        i = order[0]
        keep.append(int(i))
        if order.numel() == 1:
            break
        ious = box_iou_xyxy(boxes[i].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
