"""YOLOv3 loss, following darknet's yolo layer.

Changes from YOLOv2Loss (yolov2_loss.py), which it subclasses. Everything not
overridden here (anchor-space coordinate regression, the (2 - w*h) weight,
the ignore threshold, target building) is inherited unchanged.

  * Three output grids. Every ground truth is matched against all nine priors
    and trains only the grid that owns the winning prior, so the scale a box
    is learned at is decided by its size.
  * Objectness is binary cross-entropy on the logit with target 1 for the
    responsible predictor and 0 elsewhere. v2 used squared error against the
    achieved IoU.
  * Classes are independent binary cross-entropies (multi-label), replacing
    softmax + squared error.
  * All term weights are 1. ignore_thresh is 0.7 as in yolov3.cfg (the paper
    text says 0.5).
"""

from collections import defaultdict

import torch
import torch.nn.functional as F

from yolov2_loss import YOLOv2Loss


class YOLOv3Loss(YOLOv2Loss):
    def __init__(self, anchors: torch.Tensor, anchor_slices: list[slice], num_classes: int = 1, ignore_thresh: float = 0.7):
        super().__init__(anchors, num_classes, ignore_thresh, w_coord=1.0, w_obj=1.0, w_noobj=1.0, w_cls=1.0)
        self.anchor_slices = anchor_slices

    def forward(self, raws: list[torch.Tensor], boxes: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        N = len(boxes)
        total = 0.0
        parts: dict[str, float] = defaultdict(float)
        for raw, sl in zip(raws, self.anchor_slices):
            grid_total, grid_parts = self.grid_loss(raw, boxes, sl)
            total = total + grid_total
            for k, v in grid_parts.items():
                parts[k] += v
        return total / N, {k: v / N for k, v in parts.items()}

    def objectness_loss(
        self, obj_logit: torch.Tensor, obj: torch.Tensor, noobj: torch.Tensor, iou: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """v3: binary cross-entropy, target 1 where responsible and 0 where not ignored."""
        bce = F.binary_cross_entropy_with_logits(obj_logit, obj.float(), reduction="none")
        return (bce * obj).sum(), (bce * noobj).sum()

    def class_loss(self, cls_logit: torch.Tensor, cls: torch.Tensor, obj: torch.Tensor) -> torch.Tensor:
        """v3: one binary cross-entropy per class, so an object may carry several labels."""
        onehot = F.one_hot(cls, self.C).float()
        return (F.binary_cross_entropy_with_logits(cls_logit, onehot, reduction="none").sum(-1) * obj).sum()
