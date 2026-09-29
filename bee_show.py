"""Show bee images with the detector's predicted boxes.

Loads a checkpoint written by bee_detection.py, runs it on validation images
and draws ground truth (green) next to predictions (red, with score).

Examples:
    python bee_show.py                                        # runs/bee_yolov3/best.pt, 12 val images
    python bee_show.py --weights runs/bee_yolov1/best.pt --num 6
    python bee_show.py --conf_threshold 0.5                   # save individual annotated images

Output goes to ``--out`` (default <checkpoint dir>/vis): one <name>_pred.jpg per image.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import yolov2
import yolov3
from bee_dataset import BeeDataset, open_image, read_annotation
from bee_detection import detect
from paper import INPUT_SIZE, NUM_BOXES
from yolo import YOLOv1

VAL_FRACTION = 0.1  # same split as bee_detection.py, so these images were never trained on

parser = argparse.ArgumentParser(description="Visualise bee detections")
parser.add_argument("--weights", default="runs/bee_yolov3/best.pt", type=str, help="Checkpoint from bee_detection.py.")
parser.add_argument("--data", default="Bees", type=str, help="Dataset root containing ds/img and ds/ann.")
parser.add_argument("--split", default="val", choices=["train", "val"], help="Which split to draw from.")
parser.add_argument("--num", default=12, type=int, help="How many images to show.")
parser.add_argument("--shuffle", default=False, action="store_true", help="Pick random images instead of the first ones.")
parser.add_argument("--seed", default=0, type=int, help="Seed for --shuffle and the train/val split.")
parser.add_argument("--conf_threshold", default=0.3, type=float, help="Minimum score to draw a prediction.")
parser.add_argument("--nms_threshold", default=0.5, type=float, help="IoU above which NMS drops a duplicate.")
parser.add_argument("--no_gt", default=False, action="store_true", help="Hide the ground-truth boxes.")
parser.add_argument("--out", default=None, type=str, help="Output directory; defaults to <checkpoint dir>/vis.")
parser.add_argument("--device", default=None, type=str, help="Torch device; autodetected when omitted.")


def load_model(weights: Path, device: torch.device) -> tuple[torch.nn.Module, str, int]:
    """Rebuild the network the checkpoint was trained with and load its weights."""
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    version = args.get("model", "v1")
    num_classes = args.get("num_classes", 1)
    state = ckpt["model"]

    if version == "v1":
        model = YOLOv1(num_classes=num_classes, num_boxes=NUM_BOXES)
        img_size = INPUT_SIZE
    else:
        anchors = state["anchors"]  # priors are saved as a buffer, so the checkpoint carries its own
        model = (yolov2.YOLOv2 if version == "v2" else yolov3.YOLOv3)(anchors, num_classes=num_classes)
        img_size = yolov2.INPUT_SIZE
    model.load_state_dict(state)
    model.to(device).eval()

    extra = f", mAP@0.5 {ckpt['best_map']:.3f}" if "best_map" in ckpt else ""
    print(f"loaded YOLO{version} from {weights} (epoch {ckpt.get('epoch', '?')}{extra})")
    return model, version, img_size


def draw_boxes(image: Image.Image, boxes_xyxy: np.ndarray, colour: str, scores=None, width: int = 3) -> None:
    """Draw pixel-space xyxy boxes (and optional scores) onto a PIL image in place."""
    draw = ImageDraw.Draw(image)
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy.tolist()):
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=width)
        if scores is not None:
            text = f"{float(scores[i]):.2f}"
            tx, ty = x1 + 2, max(0, y1 - 14)
            tw = 7 * len(text) + 4
            draw.rectangle([tx - 2, ty, tx + tw, ty + 13], fill=colour)
            draw.text((tx, ty), text, fill="black")


@torch.no_grad()
def main(args: argparse.Namespace):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    weights = Path(args.weights)
    out_dir = Path(args.out or weights.parent / "vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, version, img_size = load_model(weights, device)
    dataset = BeeDataset(args.data, split=args.split, img_size=img_size, augment=False,
                         val_fraction=VAL_FRACTION, seed=args.seed)

    indices = list(range(len(dataset)))
    if args.shuffle:
        random.Random(args.seed).shuffle(indices)
    indices = indices[: args.num]

    total_pred, total_gt = 0, 0
    for idx in indices:
        name = dataset.samples[idx]
        gt_boxes, _, _ = read_annotation(dataset.ann_dir / name)
        image = open_image(dataset.img_dir / name[: -len(".json")])  # upright; matches JSON boxes
        width, height = image.size

        # same preprocessing as training: stretch to the network size, pixels in [0, 1]
        pixels = np.asarray(image.resize((img_size, img_size), Image.BILINEAR), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)
        pred = model(tensor)
        det = detect(model, pred, args.conf_threshold, args.nms_threshold)[0]

        # predictions are normalised xyxy, so scale back to the original picture
        scale = np.array([width, height, width, height], dtype=np.float32)
        pred_boxes = det["boxes"].numpy() * scale

        if not args.no_gt:
            draw_boxes(image, gt_boxes, "lime", width=3)
        draw_boxes(image, pred_boxes, "red", scores=det["scores"], width=3)

        stem = Path(name[: -len(".json")]).stem  # "IMG_4672.jpeg.json" -> "IMG_4672"
        image.save(out_dir / f"{stem}_pred.jpg", quality=90)
        total_pred += len(pred_boxes)
        total_gt += len(gt_boxes)
        print(f"{stem}: {len(gt_boxes)} bees annotated, {len(pred_boxes)} predicted")

    print(f"\n{total_gt} annotated / {total_pred} predicted over {len(indices)} images")
    print(f"green = ground truth, red = YOLO{version} prediction (score >= {args.conf_threshold})")
    print(f"saved individual predictions to {out_dir}/")


if __name__ == "__main__":
    main(parser.parse_args())