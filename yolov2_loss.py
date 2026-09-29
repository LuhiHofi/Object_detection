"""YOLOv2 loss, following darknet's region layer (the paper gives no equation).

Changes from YOLOv1Loss (yolo_loss.py):

  * Targets are built here from the raw ground-truth boxes instead of a grid
    tensor from the dataset. A box goes to the cell holding its centre and to
    the *prior whose shape matches it best* (IoU of the two shapes), rather
    than to whichever free-form predictor currently overlaps it most.
  * Coordinates are regressed in anchor space: sigmoid offsets for the centre,
    log size ratios for width and height. The (2 - w*h) weight replaces v1's
    square-root trick for making small boxes count.
  * A predictor that overlaps *any* ground truth by more than ``ignore_thresh``
    is left alone by the no-object term. v1 pushed every non-responsible
    predictor to zero confidence, punishing near-duplicates of a good box.
  * Each anchor has its own class distribution: softmax, squared error.
  * Term weights follow yolov2.cfg: coord 1, object 5, no-object 1, class 1
    (v1: 5 / 1 / 0.5 / 1). The object target is the IoU, as in v1 (rescore=1).

``forward(raw, boxes)`` takes the model's raw grid and a list of per-image
(n_i, 4) normalised xyxy tensors, and returns (loss, per-term breakdown).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bboxes_utils import box_iou_xyxy, iou_elementwise
from yolov2 import grid_to_boxes


class YOLOv2Loss(nn.Module):
    def __init__(
        self,
        anchors: torch.Tensor,
        num_classes: int = 1,
        ignore_thresh: float = 0.6,
        w_coord: float = 1.0,
        w_obj: float = 5.0,
        w_noobj: float = 1.0,
        w_cls: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("anchors", torch.as_tensor(anchors, dtype=torch.float32).clone())
        self.C = num_classes
        self.ignore_thresh = ignore_thresh
        self.w_coord, self.w_obj, self.w_noobj, self.w_cls = w_coord, w_obj, w_noobj, w_cls

    def forward(self, raw: torch.Tensor, boxes: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        N = len(boxes)
        total, parts = self.grid_loss(raw, boxes, slice(0, len(self.anchors)))
        return total / N, {k: v / N for k, v in parts.items()}

    # ------------------------------------------------------------------
    # one output grid; v3 calls this once per scale
    # ------------------------------------------------------------------

    def grid_loss(
        self, raw: torch.Tensor, boxes: list[torch.Tensor], anchor_slice: slice
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Summed loss terms for one grid whose priors are ``self.anchors[anchor_slice]``."""
        raw = raw.float()
        pred_xyxy, obj_logit, cls_logit = grid_to_boxes(raw, self.anchors[anchor_slice])
        t = self.build_targets(boxes, pred_xyxy, anchor_slice)
        obj, noobj = t["obj"], t["noobj"]

        coord_err = (raw[..., :2].sigmoid() - t["txy"]).pow(2).sum(-1) + (raw[..., 2:4] - t["twh"]).pow(2).sum(-1)
        loss_coord = self.w_coord * (coord_err * t["wh_weight"] * obj).sum()

        iou = iou_elementwise(pred_xyxy, t["gt"]).detach()
        loss_obj, loss_noobj = self.objectness_loss(obj_logit, obj, noobj, iou)
        loss_cls = self.class_loss(cls_logit, t["cls"], obj)

        total = loss_coord + loss_obj + loss_noobj + loss_cls
        parts = {
            "coord": loss_coord.item(),
            "obj": loss_obj.item(),
            "noobj": loss_noobj.item(),
            "cls": loss_cls.item(),
        }
        return total, parts

    def build_targets(
        self, boxes: list[torch.Tensor], pred_xyxy: torch.Tensor, anchor_slice: slice
    ) -> dict[str, torch.Tensor]:
        """Assign ground truth to (cell, prior) and mark predictors the no-object term should ignore.

        Matching considers *all* priors so that, for v3, a box trains only the
        grid owning its best prior. For v2 the slice covers every prior.
        """
        N, S, _, A, _ = pred_xyxy.shape
        dev = pred_xyxy.device
        K = self.anchors
        obj = torch.zeros(N, S, S, A, dtype=torch.bool, device=dev)
        noobj = torch.ones(N, S, S, A, dtype=torch.bool, device=dev)
        txy = torch.zeros(N, S, S, A, 2, device=dev)
        twh = torch.zeros(N, S, S, A, 2, device=dev)
        gt = torch.zeros(N, S, S, A, 4, device=dev)
        cls = torch.zeros(N, S, S, A, dtype=torch.long, device=dev)
        wh_weight = torch.ones(N, S, S, A, device=dev)

        for n, b in enumerate(boxes):
            b = b.to(dev).float()
            wh = b[:, 2:] - b[:, :2]
            valid = (wh > 0).all(1)
            b, wh = b[valid], wh[valid]
            if len(b) == 0:
                continue

            # predictors already overlapping some object are neither rewarded nor punished
            best_pred_iou = box_iou_xyxy(pred_xyxy[n].reshape(-1, 4), b).max(1).values
            noobj[n] &= (best_pred_iou <= self.ignore_thresh).view(S, S, A)

            # best prior by shape, then keep the boxes whose prior belongs to this grid
            inter = torch.min(wh[:, None], K[None]).prod(-1)
            best = (inter / (wh.prod(-1)[:, None] + K.prod(-1)[None] - inter)).argmax(1)
            keep = (best >= anchor_slice.start) & (best < anchor_slice.stop)
            if not keep.any():
                continue
            b, wh, best = b[keep], wh[keep], best[keep]
            a = best - anchor_slice.start

            centre = (b[:, :2] + b[:, 2:]) / 2
            j = (centre[:, 0] * S).long().clamp(max=S - 1)  # column
            i = (centre[:, 1] * S).long().clamp(max=S - 1)  # row
            obj[n, i, j, a] = True
            noobj[n, i, j, a] = False
            txy[n, i, j, a] = centre * S - torch.stack((j, i), dim=1)
            twh[n, i, j, a] = (wh / K[best]).log()
            gt[n, i, j, a] = b
            wh_weight[n, i, j, a] = 2 - wh.prod(-1)
            # cls stays 0: the bee dataset carries no labels (single class)

        return dict(obj=obj, noobj=noobj, txy=txy, twh=twh, gt=gt, cls=cls, wh_weight=wh_weight)

    # ------------------------------------------------------------------
    # the two terms v3 redefines
    # ------------------------------------------------------------------

    def objectness_loss(
        self, obj_logit: torch.Tensor, obj: torch.Tensor, noobj: torch.Tensor, iou: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """v2: squared error on sigmoid(to); the object target is the achieved IoU."""
        p = obj_logit.sigmoid()
        return self.w_obj * ((p - iou).pow(2) * obj).sum(), self.w_noobj * (p.pow(2) * noobj).sum()

    def class_loss(self, cls_logit: torch.Tensor, cls: torch.Tensor, obj: torch.Tensor) -> torch.Tensor:
        """v2: squared error between softmax probabilities and the one-hot label."""
        onehot = F.one_hot(cls, self.C).float()
        return self.w_cls * ((cls_logit.softmax(-1) - onehot).pow(2).sum(-1) * obj).sum()
