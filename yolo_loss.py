"""YOLOv1 multi-part loss (Redmon et al., arXiv:1506.02640, equation 3)."""

from __future__ import annotations

import torch
import torch.nn as nn

from paper import GRID_SIZE, LAMBDA_COORD, LAMBDA_NOOBJ, NUM_BOXES


def boxes_iou_cellwise(pred: torch.Tensor, target: torch.Tensor, grid_size: int) -> torch.Tensor:
    """IoU between boxes stored as (x, y, w, h) in YOLOv1 cell convention.

    x, y are offsets within a cell while w, h are image fractions. Both boxes
    belong to the same cell, so the cell origin cancels out and only the
    consistent x/S, y/S scaling matters.
    """
    def to_corners(box: torch.Tensor) -> tuple[torch.Tensor, ...]:
        cx, cy = box[..., 0] / grid_size, box[..., 1] / grid_size
        w, h = box[..., 2], box[..., 3]
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    px1, py1, px2, py2 = to_corners(pred)
    tx1, ty1, tx2, ty2 = to_corners(target)

    inter_w = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    inter_h = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = inter_w * inter_h

    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (ty2 - ty1) - inter
    return inter / union.clamp(min=1e-6)


class YOLOv1Loss(nn.Module):
    """Sum-squared error over coordinates, confidence and class scores.

    Only the predictor with the highest IoU in a cell containing an object is
    made "responsible" for it; everything else is pushed toward zero
    confidence, damped by ``lambda_noobj``.

    As in section 2.2 of the paper the network predicts the square root of
    width and height, so the localisation term is plain squared error against
    ``√w``, ``√h`` of the ground truth and the decoder squares the output.
    """

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        num_boxes: int = NUM_BOXES,
        num_classes: int = 1,
        lambda_coord: float = LAMBDA_COORD,
        lambda_noobj: float = LAMBDA_NOOBJ,
    ):
        super().__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred:   (N, S, S, C + B*5), width/height in square-root space
            target: (N, S, S, C + 5)
        Returns:
            (mean loss over the batch, breakdown for logging)
        """
        N, C, B = pred.shape[0], self.C, self.B

        pred_boxes = pred[..., C:].reshape(*pred.shape[:3], B, 5)
        pred_conf = pred_boxes[..., 0]           # (N, S, S, B)
        pred_xywh = pred_boxes[..., 1:]          # (N, S, S, B, 4) as (x, y, √w, √h)

        obj = target[..., C]                     # (N, S, S)
        true_xywh = target[..., C + 1 : C + 5]   # (N, S, S, 4)

        # (x, y, √w, √h) -> (x, y, w, h)
        pred_geom = torch.cat((pred_xywh[..., :2], pred_xywh[..., 2:] ** 2), dim=-1)
        ious = boxes_iou_cellwise(pred_geom, true_xywh.unsqueeze(3), self.S)  # (N, S, S, B)
        best_iou, best_idx = ious.max(dim=-1)

        responsible = torch.zeros_like(pred_conf).scatter_(-1, best_idx.unsqueeze(-1), 1.0) * obj.unsqueeze(-1)
        not_responsible = 1.0 - responsible

        true_xywh_b = true_xywh.unsqueeze(3).expand_as(pred_xywh) # (N, S, S, B, 4) match the bboxes
        mask = responsible.unsqueeze(-1)
        loss_xy = ((pred_xywh[..., :2] - true_xywh_b[..., :2]) ** 2 * mask).sum()
        loss_wh = ((pred_xywh[..., 2:] - true_xywh_b[..., 2:].sqrt()) ** 2 * mask).sum()

        conf_target = (best_iou.detach().unsqueeze(-1) * responsible).clamp(0, 1)
        loss_obj = ((pred_conf - conf_target) ** 2 * responsible).sum()
        loss_noobj = (pred_conf**2 * not_responsible).sum() # (C_i - C_i_hat)^2 = (C_i^2 - 0)^2 = C_i^2 C_i_hat for no object is 0

        loss_cls = ((pred[..., :C] - target[..., :C]) ** 2 * obj.unsqueeze(-1)).sum()

        total = (
            self.lambda_coord * (loss_xy + loss_wh)
            + loss_obj
            + self.lambda_noobj * loss_noobj
            + loss_cls
        ) / N
        parts = {
            "xy": loss_xy.item() / N,
            "wh": loss_wh.item() / N,
            "obj": loss_obj.item() / N,
            "noobj": loss_noobj.item() / N,
            "cls": loss_cls.item() / N,
        }
        return total, parts
