"""YOLOv3 (Redmon & Farhadi, "YOLOv3: An Incremental Improvement", arXiv:1804.02767).

Changes from YOLOv2 (yolov2.py), whose conv_bn block and grid_to_boxes decoder
are reused unchanged:

  * New backbone, Darknet-53: residual (shortcut) blocks, and downsampling by
    stride-2 convolutions instead of max-pooling. 53 layers counting the
    classifier, hence the name.
  * Predictions at three scales (strides 32, 16, 8 -> 13x13, 26x26, 52x52 at
    416) with an FPN-style top-down path: the coarse features are upsampled
    and concatenated with the matching backbone level. Replaces the single
    passthrough. This is what fixes v1/v2's weakness on small, clustered
    objects.
  * 3 priors per scale, 9 in total, assigned to scales by size: the largest
    priors sit on the coarsest grid.
  * Objectness and every class are independent logistic outputs, trained with
    binary cross-entropy; the class softmax is gone, so labels may overlap.
  * Not implemented here: none of the "things we tried that didn't work".

Raw output per cell is unchanged: (A, 5 + C) = [tx, ty, tw, th, to, cls...].
``forward`` returns a list of three raw grids, coarse to fine.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov2 import conv_bn, grid_to_boxes

INPUT_SIZE = 416
NUM_SCALES = 3
ANCHORS_PER_SCALE = 3
NUM_ANCHORS = NUM_SCALES * ANCHORS_PER_SCALE


class Residual(nn.Module):
    """1x1 bottleneck then 3x3, added to the input. New in v3."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(conv_bn(channels, channels // 2, 1), conv_bn(channels // 2, channels, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


def stage(in_c: int, out_c: int, blocks: int) -> nn.Sequential:
    """Stride-2 conv (the downsampling, replacing v2's max-pool) followed by residual blocks."""
    return nn.Sequential(conv_bn(in_c, out_c, 3, stride=2), *(Residual(out_c) for _ in range(blocks)))


class Darknet53(nn.Module):
    """Table 1 of the paper, without the classifier.

    Returns the stride-8, -16 and -32 feature maps (52x52x256, 26x26x512, 13x13x1024 at 416).
    """

    def __init__(self):
        super().__init__()
        self.stem = conv_bn(3, 32, 3)
        self.s1 = stage(32, 64, 1)
        self.s2 = stage(64, 128, 2)
        self.s3 = stage(128, 256, 8)
        self.s4 = stage(256, 512, 8)
        self.s5 = stage(512, 1024, 4)
        self.out_channels = (256, 512, 1024)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.s2(self.s1(self.stem(x)))
        c3 = self.s3(x)
        c4 = self.s4(c3)
        c5 = self.s5(c4)
        return c3, c4, c5


class Darknet53Classifier(nn.Module):
    """Backbone + global average pool + linear layer, for the classification pretraining stage."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = Darknet53()
        self.head = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, c5 = self.backbone(x)
        return self.head(c5.mean(dim=(2, 3)))


def detection_block(in_c: int, mid: int) -> nn.Sequential:
    """The five alternating 1x1 / 3x3 convs that precede each output in yolov3.cfg."""
    return nn.Sequential(
        conv_bn(in_c, mid, 1), conv_bn(mid, 2 * mid, 3),
        conv_bn(2 * mid, mid, 1), conv_bn(mid, 2 * mid, 3),
        conv_bn(2 * mid, mid, 1),
    )


def output_conv(mid: int, out: int) -> nn.Sequential:
    """One 3x3 conv then the 1x1 prediction conv (bias, no BatchNorm)."""
    return nn.Sequential(conv_bn(mid, 2 * mid, 3), nn.Conv2d(2 * mid, out, 1))


class YOLOv3(nn.Module):
    def __init__(self, anchors, num_classes: int = 1):
        """anchors: (9, 2) prior (w, h) as image fractions, sorted by area (kmeans_anchors does this)."""
        super().__init__()
        self.register_buffer("anchors", torch.as_tensor(anchors, dtype=torch.float32))
        assert len(self.anchors) == NUM_ANCHORS, f"YOLOv3 expects {NUM_ANCHORS} anchors"
        self.num_classes = num_classes
        # scale k (0 = coarsest) owns priors [(2-k)*3, (3-k)*3): biggest boxes on the 13x13 grid
        self.anchor_slices = [
            slice((NUM_SCALES - 1 - k) * ANCHORS_PER_SCALE, (NUM_SCALES - k) * ANCHORS_PER_SCALE)
            for k in range(NUM_SCALES)
        ]
        out = ANCHORS_PER_SCALE * (5 + num_classes)

        self.backbone = Darknet53()
        # stride 32
        self.block5 = detection_block(1024, 512)
        self.out5 = output_conv(512, out)
        # stride 16: upsample the stride-32 features and concatenate the backbone's C4
        self.lateral5 = conv_bn(512, 256, 1)
        self.block4 = detection_block(256 + 512, 256)
        self.out4 = output_conv(256, out)
        # stride 8: same again with C3
        self.lateral4 = conv_bn(256, 128, 1)
        self.block3 = detection_block(128 + 256, 128)
        self.out3 = output_conv(128, out)

    def _to_grid(self, out: torch.Tensor) -> torch.Tensor:
        N, _, S, _ = out.shape
        return out.permute(0, 2, 3, 1).reshape(N, S, S, ANCHORS_PER_SCALE, 5 + self.num_classes)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """(N, 3, 416, 416) -> [(N, 13, 13, 3, 5+C), (N, 26, 26, 3, 5+C), (N, 52, 52, 3, 5+C)]."""
        c3, c4, c5 = self.backbone(x)

        p5 = self.block5(c5)
        up = F.interpolate(self.lateral5(p5), scale_factor=2, mode="nearest")
        p4 = self.block4(torch.cat((up, c4), dim=1))
        up = F.interpolate(self.lateral4(p4), scale_factor=2, mode="nearest")
        p3 = self.block3(torch.cat((up, c3), dim=1))

        return [self._to_grid(self.out5(p5)), self._to_grid(self.out4(p4)), self._to_grid(self.out3(p3))]

    def decode(self, raws: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Three raw grids -> (boxes (N, P, 4) normalised xyxy, scores (N, P, C)), P summed over scales.

        Unlike v2 the classes are independent sigmoids: score = sigmoid(to) * sigmoid(cls).
        """
        boxes, scores = [], []
        for raw, sl in zip(raws, self.anchor_slices):
            xyxy, obj, cls = grid_to_boxes(raw, self.anchors[sl])
            N = raw.shape[0]
            boxes.append(xyxy.reshape(N, -1, 4))
            scores.append((obj.sigmoid().unsqueeze(-1) * cls.sigmoid()).reshape(N, -1, self.num_classes))
        return torch.cat(boxes, dim=1), torch.cat(scores, dim=1)
