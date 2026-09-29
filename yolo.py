"""Manual implementations of YOLOv1.

YOLOv1 (Redmon et al., arXiv:1506.02640, Figure 3):
    24 convolutional layers + 2 fully connected layers, output S×S×(B*5+C).
"""

import torch
import torch.nn as nn

from paper import DROPOUT, GRID_SIZE, LEAKY_SLOPE, NUM_BOXES

class YOLOv1(nn.Module):
    def __init__(self, num_classes: int = 20, num_boxes: int = NUM_BOXES, num_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.num_channels = num_channels
        self.grid_size = GRID_SIZE
        self.out_dim = num_boxes * 5 + num_classes

        def conv(in_c: int, out_c: int, k: int, stride: int = 1) -> nn.Sequential:
            pad = (k - 1) // 2
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=k, stride=stride, padding=pad, bias=True),
                nn.LeakyReLU(negative_slope=LEAKY_SLOPE),
            )

        self.features = nn.Sequential(
            # 448×448×3 → 112×112×64
            conv(num_channels, 64, 7, stride=2),
            nn.MaxPool2d(2, stride=2),
            # → 56×56×192
            conv(64, 192, 3),
            nn.MaxPool2d(2, stride=2),
            # → 28×28×512
            conv(192, 128, 1),
            conv(128, 256, 3),
            conv(256, 256, 1),
            conv(256, 512, 3),
            nn.MaxPool2d(2, stride=2),
            # {1×1×256, 3×3×512} × 4
            *(m for _ in range(4) for m in (conv(512, 256, 1), conv(256, 512, 3))),
            # → 14×14×1024
            conv(512, 512, 1),
            conv(512, 1024, 3),
            nn.MaxPool2d(2, stride=2),
            # {1×1×512, 3×3×1024} × 2
            *(m for _ in range(2) for m in (conv(1024, 512, 1), conv(512, 1024, 3))),
            conv(1024, 1024, 3),
            conv(1024, 1024, 3, stride=2),  # → 7×7×1024
            conv(1024, 1024, 3),
            conv(1024, 1024, 3),
        )

        # head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.grid_size * self.grid_size, 4096),
            nn.LeakyReLU(negative_slope=LEAKY_SLOPE),
            nn.Dropout(p=DROPOUT),
            nn.Linear(4096, self.grid_size * self.grid_size * self.out_dim),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        """He initialisation, as Darknet uses for its conv layers."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=LEAKY_SLOPE, nonlinearity="leaky_relu")
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 3, 448, 448)
        Returns:
            (N, S, S, B*5+C)  e.g. (N, 7, 7, 30) for VOC
        """
        x = self.features(x)
        x = self.classifier(x)
        return x.view(-1, self.grid_size, self.grid_size, self.out_dim)
