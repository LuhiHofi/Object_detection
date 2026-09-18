"""Dataset-Ninja bee dataset → YOLOv1 grid targets.

Annotations are Supervisely JSON: one ``<image>.jpeg.json`` per image holding
axis-aligned rectangles with two exterior points.

Tensor layouts used across the project (C = number of classes, B = boxes/cell):
    prediction  (S, S, C + B*5)  [cls..., (conf, x, y, √w, √h) * B]
    target      (S, S, C + 5)    [cls..., obj, x, y, w, h]
where x, y are offsets inside the cell and w, h are fractions of the image.
Predictions carry the square root of w, h as in section 2.2 of the paper.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from bboxes_utils import boxes_to_input, clip_boxes_xyxy
from paper import GRID_SIZE, HSV_FACTOR, INPUT_SIZE


def read_annotation(path: Path) -> tuple[np.ndarray, int, int]:
    """Return (boxes_xyxy, width, height) for one Supervisely JSON file."""
    with open(path) as f:
        data = json.load(f)

    size = data["size"]
    width, height = int(size["width"]), int(size["height"])

    boxes = []
    for obj in data.get("objects", []):
        if obj.get("geometryType") != "rectangle":
            continue
        (x1, y1), (x2, y2) = obj["points"]["exterior"]
        boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), width, height
    return np.asarray(boxes, dtype=np.float32), width, height


class BeeDataset(Dataset):
    """Bee images resized (stretched) to ``img_size`` with YOLOv1 grid targets."""

    def __init__(
        self,
        root: str | Path = "Bees",
        split: str = "train",
        img_size: int = INPUT_SIZE,
        grid_size: int = GRID_SIZE,
        num_classes: int = 1,
        augment: bool = False,
        val_fraction: float = 0.1,
        seed: int = 0,
    ):
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")

        self.root = Path(root)
        self.img_dir = self.root / "ds" / "img"
        self.ann_dir = self.root / "ds" / "ann"
        if not self.ann_dir.is_dir():
            raise FileNotFoundError(f"No annotations at {self.ann_dir}. Unpack the dataset tar first.")

        self.img_size = img_size
        self.S = grid_size
        self.C = num_classes
        self.augment = augment and split == "train"

        stems = sorted(p.name for p in self.ann_dir.glob("*.json"))
        rng = random.Random(seed)
        rng.shuffle(stems)
        n_val = max(1, int(len(stems) * val_fraction))
        self.samples = stems[n_val:] if split == "train" else stems[:n_val]

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal flip plus HSV exposure/saturation jitter (as in the paper)."""
        if random.random() < 0.5:
            image = image[:, ::-1]
            if len(boxes):
                w = image.shape[1]
                boxes = boxes.copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= random.uniform(1 / HSV_FACTOR, HSV_FACTOR)  # saturation
        hsv[..., 2] *= random.uniform(1 / HSV_FACTOR, HSV_FACTOR)  # exposure
        hsv[..., 1:] = np.clip(hsv[..., 1:], 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return image, boxes

    def encode_target(self, boxes_norm: np.ndarray) -> torch.Tensor:
        """Normalised xyxy boxes → (S, S, C + 5) grid target.

        A cell can only hold one object, so later boxes landing in an occupied
        cell are dropped. That is the YOLOv1 limitation, not a bug.
        """
        target = torch.zeros(self.S, self.S, self.C + 5, dtype=torch.float32)
        for x1, y1, x2, y2 in boxes_norm.tolist():
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            j = min(int(cx * self.S), self.S - 1)  # column
            i = min(int(cy * self.S), self.S - 1)  # row
            if target[i, j, self.C] == 1:
                continue
            target[i, j, 0] = 1.0  # object is bee
            target[i, j, self.C] = 1.0 # confidence of object
            target[i, j, self.C + 1] = cx * self.S - j
            target[i, j, self.C + 2] = cy * self.S - i
            target[i, j, self.C + 3] = bw
            target[i, j, self.C + 4] = bh
        return target

    def __getitem__(self, idx: int) -> dict:
        name = self.samples[idx]
        boxes, width, height = read_annotation(self.ann_dir / name)

        image = Image.open(self.img_dir / name[: -len(".json")]).convert("RGB")
        image = np.asarray(image.resize((self.img_size, self.img_size), Image.BILINEAR))
        boxes = boxes_to_input(boxes, width, height, self.img_size)

        if self.augment:
            image, boxes = self._augment(image, boxes)

        boxes = clip_boxes_xyxy(boxes, self.img_size, self.img_size)
        if len(boxes):  # drop boxes that augmentation collapsed
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
        boxes_norm = boxes / self.img_size

        # Darknet/YOLOv1 preprocessing: pixels scaled to [0, 1], nothing else.
        pixels = np.ascontiguousarray(image, dtype=np.float32) / 255.0

        return {
            "image": torch.from_numpy(pixels).permute(2, 0, 1),
            "target": self.encode_target(boxes_norm),
            "boxes": torch.from_numpy(np.ascontiguousarray(boxes_norm, dtype=np.float32)),
            "size": torch.tensor([width, height]),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Stack images/targets; keep per-image ground truth as a ragged list."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "size": torch.stack([b["size"] for b in batch]),
    }
