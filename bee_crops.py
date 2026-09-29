"""Bee / background crops for classification pretraining."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from bee_dataset import BeeDataset, open_image, read_annotation


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    lt = np.maximum(box[:2], boxes[:, :2])
    rb = np.minimum(box[2:], boxes[:, 2:])
    inter = np.clip(rb - lt, 0, None).prod(1)
    area = lambda b: np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)  # noqa: E731
    return inter / np.maximum(area(box) + area(boxes) - inter, 1e-6)


class BeeCropDataset(Dataset):
    def __init__(
        self,
        root: str | Path = "Bees",
        split: str = "train",
        crop_size: int = 224,
        crops_per_image: int = 4,
        context: float = 0.2,
        augment: bool = True,
        val_fraction: float = 0.1,
        seed: int = 0,
    ):
        base = BeeDataset(root, split, val_fraction=val_fraction, seed=seed)
        self.samples, self.img_dir, self.ann_dir = base.samples, base.img_dir, base.ann_dir
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.context = context  # extra margin around a bee, as a fraction of its size
        self.augment = augment and split == "train"

    def __len__(self) -> int:
        return len(self.samples) * self.crops_per_image

    def _positive(self, boxes: np.ndarray) -> np.ndarray:
        """A random annotated bee with some surrounding context and a little jitter."""
        x1, y1, x2, y2 = boxes[random.randrange(len(boxes))]
        w, h = x2 - x1, y2 - y1
        side = max(w, h) * (1 + self.context)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if self.augment:
            cx += random.uniform(-0.1, 0.1) * w
            cy += random.uniform(-0.1, 0.1) * h
            side *= random.uniform(0.9, 1.1)
        return np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], dtype=np.float32)

    def _negative(self, boxes: np.ndarray, width: int, height: int) -> np.ndarray | None:
        """A bee-sized square that overlaps no annotated bee; None if 20 tries fail."""
        for _ in range(20):
            if len(boxes):
                x1, y1, x2, y2 = boxes[random.randrange(len(boxes))]
                side = max(x2 - x1, y2 - y1) * (1 + self.context)
            else:
                side = random.uniform(0.1, 0.3) * min(width, height)
            side = min(side, width, height)
            x = random.uniform(0, width - side)
            y = random.uniform(0, height - side)
            crop = np.array([x, y, x + side, y + side], dtype=np.float32)
            if len(boxes) == 0 or _iou_one_to_many(crop, boxes).max() < 0.05:
                return crop
        return None

    def __getitem__(self, idx: int) -> dict:
        name = self.samples[idx % len(self.samples)]
        boxes, width, height = read_annotation(self.ann_dir / name)
        image = open_image(self.img_dir / name[: -len(".json")])

        crop, label = None, 0
        if len(boxes) and random.random() < 0.5:
            crop, label = self._positive(boxes), 1
        else:
            crop = self._negative(boxes, width, height)
            if crop is None:  # crowded image with no free space: fall back to a bee
                crop, label = self._positive(boxes), 1

        patch = image.crop(tuple(int(round(v)) for v in crop)).resize((self.crop_size, self.crop_size), Image.BILINEAR)
        pixels = np.asarray(patch, dtype=np.float32) / 255.0
        if self.augment and random.random() < 0.5:
            pixels = pixels[:, ::-1]

        return {
            "image": torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1),
            "label": torch.tensor(label),
        }
