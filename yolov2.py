"""YOLOv2 (Redmon & Farhadi, "YOLO9000: Better, Faster, Stronger", arXiv:1612.08242).

Changes from YOLOv1 (yolo.py), in the order the paper's "Better" section lists them:

  * BatchNorm after every convolution (conv bias dropped); dropout removed.
  * New backbone, Darknet-19: 19 convolutions, only 3x3 and 1x1 kernels, and
    a global-average-pool classifier instead of fully connected layers. It is
    pretrained for classification (Darknet19Classifier) and then reused.
  * The fully connected detection head is gone, so the net is fully
    convolutional and the grid follows the input: 416 -> 13x13. 416 was chosen
    so the grid is odd and large objects get a single centre cell.
  * Anchor boxes: each cell predicts B=5 boxes as offsets from k-means priors
    (bboxes_utils.kmeans_anchors), and every box carries its own class
    distribution. v1 shared one class vector per cell and regressed boxes freely.
  * Direct location prediction: centre = sigmoid(t) + cell, size = prior*exp(t).
    v1 regressed raw (x, y, sqrt w, sqrt h).
  * Passthrough: the 26x26 layer is space-to-depth reshaped to 13x13 and
    concatenated onto the final features for finer-grained detail.
  * Not implemented: multi-scale training and the 448 hi-res classifier
    fine-tune

Raw output per cell is (B, 5 + C) = [tx, ty, tw, th, to, cls...]. ``decode``
turns it into normalised xyxy boxes and Pr(object) * Pr(class) scores.
"""

import torch
import torch.nn as nn

from paper import LEAKY_SLOPE

INPUT_SIZE = 416  # 416 / 32 = 13, an odd grid
NUM_ANCHORS = 5   # section 2, "Dimension Clusters": k=5 balances recall and complexity


def conv_bn(in_c: int, out_c: int, k: int, stride: int = 1) -> nn.Sequential:
    """Convolution + BatchNorm + leaky ReLU: the v2 building block (v1 had no BatchNorm)."""
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, stride, (k - 1) // 2, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(LEAKY_SLOPE),
    )


class Darknet19(nn.Module):
    """Table 6 of the paper, without the final 1x1 classifier conv.

    Returns the 26x26x512 layer (for the passthrough) and the 13x13x1024 output.
    """

    def __init__(self):
        super().__init__()

        def pool() -> nn.MaxPool2d:
            return nn.MaxPool2d(2, 2)

        self.to_c4 = nn.Sequential(
            conv_bn(3, 32, 3), pool(),                                              # 208
            conv_bn(32, 64, 3), pool(),                                             # 104
            conv_bn(64, 128, 3), conv_bn(128, 64, 1), conv_bn(64, 128, 3), pool(),  # 52
            conv_bn(128, 256, 3), conv_bn(256, 128, 1), conv_bn(128, 256, 3), pool(),  # 26
            conv_bn(256, 512, 3), conv_bn(512, 256, 1), conv_bn(256, 512, 3),
            conv_bn(512, 256, 1), conv_bn(256, 512, 3),
        )
        self.to_c5 = nn.Sequential(
            pool(),                                                                 # 13
            conv_bn(512, 1024, 3), conv_bn(1024, 512, 1), conv_bn(512, 1024, 3),
            conv_bn(1024, 512, 1), conv_bn(512, 1024, 3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c4 = self.to_c4(x)
        return c4, self.to_c5(c4)


class Darknet19Classifier(nn.Module):
    """The pretraining network of section 3: backbone, 1x1 conv to classes, global average pool.

    Train this on a classification task, then hand ``.backbone`` to YOLOv2.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = Darknet19()
        self.head = nn.Conv2d(1024, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c5 = self.backbone(x)
        return self.head(c5).mean(dim=(2, 3))


class Reorg(nn.Module):
    """Space-to-depth: (N, C, H, W) -> (N, 4C, H/2, W/2). The passthrough layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        x = x.view(N, C, H // 2, 2, W // 2, 2).permute(0, 3, 5, 1, 2, 4)
        return x.reshape(N, 4 * C, H // 2, W // 2)


def grid_to_boxes(raw: torch.Tensor, anchors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct location prediction (section 2) for one output grid.

    raw:     (N, S, S, A, 5 + C) as [tx, ty, tw, th, to, cls...]
    anchors: (A, 2) prior (w, h) as fractions of the image
    Returns xyxy boxes (N, S, S, A, 4) in image fractions, objectness logits
    (N, S, S, A) and class logits (N, S, S, A, C). Shared by v2, v3 and their losses.
    """
    raw = raw.float()
    N, S, _, A, _ = raw.shape
    rows = torch.arange(S, device=raw.device).view(1, S, 1, 1)
    cols = torch.arange(S, device=raw.device).view(1, 1, S, 1)
    cx = (cols + raw[..., 0].sigmoid()) / S
    cy = (rows + raw[..., 1].sigmoid()) / S
    wh = anchors.view(1, 1, 1, A, 2) * raw[..., 2:4].clamp(max=8).exp()  # clamp keeps early training finite
    w, h = wh[..., 0], wh[..., 1]
    xyxy = torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)
    return xyxy, raw[..., 4], raw[..., 5:]


class YOLOv2(nn.Module):
    def __init__(self, anchors, num_classes: int = 1):
        """anchors: (NUM_ANCHORS, 2) prior (w, h) as fractions of the image, e.g. from kmeans_anchors."""
        super().__init__()
        self.register_buffer("anchors", torch.as_tensor(anchors, dtype=torch.float32))  # saved with the weights
        self.num_classes = num_classes
        self.num_anchors = len(self.anchors)
        out = self.num_anchors * (5 + num_classes)

        self.backbone = Darknet19()
        # yolov2.cfg after the backbone: two 3x3 convs on the 13x13 map ...
        self.neck = nn.Sequential(conv_bn(1024, 1024, 3), conv_bn(1024, 1024, 3))
        # ... the passthrough from the 26x26 map (1x1 to 64 channels, then space-to-depth to 256) ...
        self.passthrough = nn.Sequential(conv_bn(512, 64, 1), Reorg())
        # ... concatenated and finished with one 3x3 conv and the 1x1 prediction conv (no BatchNorm, has bias).
        self.head = nn.Sequential(conv_bn(1024 + 256, 1024, 3), nn.Conv2d(1024, out, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3, 416, 416) -> (N, 13, 13, A, 5 + C) raw predictions."""
        c4, c5 = self.backbone(x)
        features = torch.cat((self.passthrough(c4), self.neck(c5)), dim=1)
        out = self.head(features)
        N, _, S, _ = out.shape
        return out.permute(0, 2, 3, 1).reshape(N, S, S, self.num_anchors, 5 + self.num_classes)

    def decode(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw grid -> (boxes (N, P, 4) normalised xyxy, scores (N, P, C)) with P = S*S*A.

        v2 keeps a softmax over classes; the score is Pr(object) * Pr(class | object).
        """
        xyxy, obj, cls = grid_to_boxes(raw, self.anchors)
        scores = obj.sigmoid().unsqueeze(-1) * cls.softmax(-1)
        N = raw.shape[0]
        return xyxy.reshape(N, -1, 4), scores.reshape(N, -1, self.num_classes)
