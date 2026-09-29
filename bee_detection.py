"""Train and evaluate YOLOv1 / v2 / v3 on the bee dataset.

Examples:
    python bee_detection.py --model v1
    python bee_pretrain.py  --model v3
    python bee_detection.py --model v3 --backbone_weights runs/pretrain_v3/backbone.pt
    python bee_detection.py --model v3 --eval_only --weights runs/bee_yolov3/best.pt

Checkpoints land in ``--save_dir``:
    last.pt  overwritten every ``--save_every`` steps
    best.pt  written whenever validation mAP@0.5 improves
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import yolov2
import yolov3
from bboxes_utils import box_iou_xyxy, kmeans_anchors, nms, get_boxes_scores
from bee_dataset import BeeDataset, collate_fn, read_annotation
from paper import BATCH_SIZE, EPOCHS, GRID_SIZE, INPUT_SIZE, IOU_THRESHOLD, NUM_BOXES, WEIGHT_DECAY
from yolo import YOLOv1
from yolo_loss import YOLOv1Loss
from yolov2_loss import YOLOv2Loss
from yolov3_loss import YOLOv3Loss

VAL_FRACTION = 0.1  # train/val split

parser = argparse.ArgumentParser(description="Train YOLO on the bee dataset")
# model
parser.add_argument("--model", default="v1", choices=["v1", "v2", "v3"], help="Which YOLO generation to train.")
parser.add_argument("--backbone_weights", default=None, type=str, help="backbone.pt from bee_pretrain.py (v2/v3 only).")
# dataset
parser.add_argument("--data", default="Bees", type=str, help="Dataset root containing ds/img and ds/ann.")
parser.add_argument("--num_classes", default=1, type=int, help="Number of classes (C); the paper used VOC's 20.")
parser.add_argument("--no_augment", default=False, action="store_true", help="Disable flip and HSV jitter.")
parser.add_argument(
    "--overfit", default=None, type=int,
    help="Pipeline sanity check: train and validate on the same N images without augmentation. "
    "A working detector reaches mAP ~1.0 on them within a few hundred steps.",
)
# optimisation: the paper's SGD schedule has no AdamW equivalent, so these stay free
parser.add_argument("--max_steps", default=None, type=int, help="Stop after this many steps; useful for smoke tests.")
parser.add_argument("--lr", default=1e-4, type=float, help="Peak learning rate.")
parser.add_argument("--min_lr_ratio", default=0.01, type=float, help="Cosine floor as a fraction of --lr.")
parser.add_argument("--warmup_steps", default=500, type=int, help="Linear warmup steps before the cosine decay.")
parser.add_argument("--clip_grad", default=10.0, type=float, help="Gradient-norm clipping threshold.")
parser.add_argument("--workers", default=8, type=int, help="Dataloader workers; 0 loads in the main process.")
parser.add_argument("--amp", default=False, action="store_true", help="Mixed precision (CUDA only).")
parser.add_argument("--device", default=None, type=str, help="Torch device; autodetected when omitted.")
parser.add_argument("--seed", default=0, type=int, help="Random seed.")
# detection thresholds
parser.add_argument("--conf_threshold", default=0.05, type=float, help="Minimum score to keep a detection.")
parser.add_argument("--nms_threshold", default=0.5, type=float, help="IoU above which NMS drops a duplicate.")
parser.add_argument("--eval_batches", default=None, type=int, help="Cap batches per evaluation for speed.")
# checkpoints
parser.add_argument("--save_dir", default=None, type=str, help="Where last.pt and best.pt are written; defaults to runs/bee_yolo<model>.")
parser.add_argument("--save_every", default=1000, type=int, help="Steps between last.pt overwrites.")
parser.add_argument("--eval_every", default=1000, type=int, help="Steps between evaluations (0 = epoch only).")
parser.add_argument("--weights", default=None, type=str, help="Checkpoint to load for --resume or --eval_only.")
parser.add_argument("--resume", default=False, action="store_true", help="Continue training from last.pt.")
parser.add_argument("--eval_only", default=False, action="store_true", help="Evaluate --weights and exit.")


def postprocess(
    xyxy: torch.Tensor, scores: torch.Tensor, conf_threshold: float, nms_threshold: float
) -> list[dict[str, torch.Tensor]]:
    """Threshold and per-class NMS: (N, P, 4) boxes + (N, P, C) scores → per-image detections."""
    results = []
    for img_boxes, img_scores in zip(xyxy.clamp(0, 1), scores):
        kept_boxes, kept_scores, kept_labels = [], [], []
        for c in range(img_scores.shape[1]):
            sc = img_scores[:, c]
            mask = sc >= conf_threshold
            if not mask.any():
                continue
            b, s = img_boxes[mask], sc[mask]
            keep = nms(b, s, nms_threshold)
            kept_boxes.append(b[keep])
            kept_scores.append(s[keep])
            kept_labels.append(torch.full((len(keep),), c, dtype=torch.long, device=xyxy.device))

        if kept_boxes:
            results.append(
                {"boxes": torch.cat(kept_boxes), "scores": torch.cat(kept_scores), "labels": torch.cat(kept_labels)}
            )
        else:
            empty = torch.zeros((0, 4), device=xyxy.device)
            results.append({"boxes": empty, "scores": empty[:, 0], "labels": empty[:, 0].long()})
    return results


def detect(model: torch.nn.Module, pred, conf_threshold: float, nms_threshold: float) -> list[dict[str, torch.Tensor]]:
    """Raw network output of any generation → per-image detections on the CPU."""
    if isinstance(model, YOLOv1):
        xyxy, scores = get_boxes_scores(pred.float(), model.num_classes, model.num_boxes)
    else:
        xyxy, scores = model.decode(pred)
    return postprocess(xyxy.float().cpu(), scores.float().cpu(), conf_threshold, nms_threshold)


def average_precision(
    predictions: list[dict[str, torch.Tensor]],
    ground_truths: list[torch.Tensor],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Single-class AP with greedy score-ordered matching (VOC all-point)."""
    n_gt = sum(len(g) for g in ground_truths)
    if n_gt == 0:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0, "n_pred": 0, "n_gt": 0}

    flat = [
        (float(s), img_idx, box)
        for img_idx, pred in enumerate(predictions)
        for box, s in zip(pred["boxes"], pred["scores"])
    ]
    if not flat:
        return {"ap": 0.0, "precision": 0.0, "recall": 0.0, "n_pred": 0, "n_gt": n_gt}
    flat.sort(key=lambda t: t[0], reverse=True)

    matched = [torch.zeros(len(g), dtype=torch.bool) for g in ground_truths]
    tp = torch.zeros(len(flat))
    fp = torch.zeros(len(flat))

    for rank, (_, img_idx, box) in enumerate(flat):
        gt = ground_truths[img_idx]
        if len(gt) == 0:
            fp[rank] = 1
            continue
        ious = box_iou_xyxy(box.unsqueeze(0).cpu(), gt.cpu()).squeeze(0)
        ious[matched[img_idx]] = -1.0  # each ground truth can only be hit once
        best, best_idx = ious.max(dim=0)
        if best >= iou_threshold:
            tp[rank] = 1
            matched[img_idx][best_idx] = True
        else:
            fp[rank] = 1

    tp_cum, fp_cum = tp.cumsum(0), fp.cumsum(0)
    recalls = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum).clamp(min=1e-9)

    # all-point interpolation: integrate the monotonically decreasing envelope
    mrec = torch.cat((torch.zeros(1), recalls, torch.ones(1)))
    mpre = torch.cat((torch.zeros(1), precisions, torch.zeros(1)))
    mpre = mpre.flip(0).cummax(0).values.flip(0)
    idx = (mrec[1:] != mrec[:-1]).nonzero().squeeze(1)
    ap = float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())

    return {
        "ap": ap,
        "precision": float(precisions[-1]),
        "recall": float(recalls[-1]),
        "n_pred": len(flat),
        "n_gt": n_gt,
    }


# Eval
def loss_target(model: torch.nn.Module, batch: dict, device: torch.device):
    """v1 learns from the dataset's grid tensor; v2/v3 build anchor targets from the raw boxes."""
    if isinstance(model, YOLOv1):
        return batch["target"].to(device, non_blocking=True)
    return [b.to(device, non_blocking=True) for b in batch["boxes"]]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    conf_threshold: float = 0.05,
    nms_threshold: float = 0.5,
    iou_threshold: float = IOU_THRESHOLD,
    max_batches: int | None = None,
    desc: str = "eval",
) -> dict[str, float]:
    """Run the validation set and report mAP@0.5 plus the mean loss."""
    model.eval()

    predictions: list[dict[str, torch.Tensor]] = []
    ground_truths: list[torch.Tensor] = []
    loss_sum, n_batches = 0.0, 0

    for i, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)

        pred = model(images)
        loss, _ = criterion(pred, loss_target(model, batch, device))
        loss_sum += float(loss)
        n_batches += 1

        predictions.extend(detect(model, pred, conf_threshold, nms_threshold))
        ground_truths.extend(batch["boxes"])

    stats = average_precision(predictions, ground_truths, iou_threshold)
    stats["loss"] = loss_sum / max(n_batches, 1)
    return stats


# Train

def save_checkpoint(
    path: Path, model, optimizer, scheduler, step: int, epoch: int, best_map: float, args, training_state: bool = True
):
    """Write atomically so an interrupt cannot leave a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_map": best_map,
        "args": vars(args),
    }
    if training_state:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()

    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def build_loaders(args) -> tuple[DataLoader, DataLoader]:
    common = dict(
        root=args.data,
        img_size=INPUT_SIZE if args.model == "v1" else yolov2.INPUT_SIZE,  # 448 vs 416
        grid_size=GRID_SIZE,
        num_classes=args.num_classes,
        val_fraction=VAL_FRACTION,
        seed=args.seed,
    )
    if args.overfit:
        # same N images for training and validation; repeat them so an epoch keeps its usual length
        val_set = BeeDataset(split="train", augment=False, **common)
        val_set.samples = val_set.samples[: args.overfit]
        train_set = BeeDataset(split="train", augment=False, **common)
        train_set.samples = val_set.samples * max(1, len(train_set.samples) // len(val_set.samples))
        print(f"overfitting {len(val_set)} images ({len(train_set)} per epoch with repeats)")
    else:
        train_set = BeeDataset(split="train", augment=not args.no_augment, **common)
        val_set = BeeDataset(split="val", augment=False, **common)
        print(f"train images: {len(train_set)}   val images: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=min(BATCH_SIZE, len(train_set)),
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=min(BATCH_SIZE, len(val_set)),
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    return train_loader, val_loader


def training_box_sizes(train_set: BeeDataset) -> np.ndarray:
    """(w, h) of every training box as image fractions, read from the annotations only."""
    sizes = []
    for name in dict.fromkeys(train_set.samples):  # unique, in case --overfit repeated them
        boxes, width, height = read_annotation(train_set.ann_dir / name)
        if len(boxes):
            sizes.append((boxes[:, 2:] - boxes[:, :2]) / np.array([width, height], dtype=np.float32))
    return np.concatenate(sizes)


def build_model(args, train_set: BeeDataset, device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Model and matching loss for the requested generation.

    v2/v3 priors are k-means clusters of the training boxes. They are stored
    as a buffer, so a checkpoint's own priors overwrite these on load.
    """
    C = args.num_classes
    if args.model == "v1":
        model = YOLOv1(num_classes=C, num_boxes=NUM_BOXES).to(device)
        return model, YOLOv1Loss(GRID_SIZE, NUM_BOXES, C).to(device)

    k = yolov2.NUM_ANCHORS if args.model == "v2" else yolov3.NUM_ANCHORS
    anchors = kmeans_anchors(training_box_sizes(train_set), k, seed=args.seed)
    print(f"{k} k-means priors (w, h as image fractions):\n{np.array2string(anchors, precision=3)}")

    if args.model == "v2":
        model = yolov2.YOLOv2(anchors, num_classes=C).to(device)
        criterion = YOLOv2Loss(model.anchors, C).to(device)
    else:
        model = yolov3.YOLOv3(anchors, num_classes=C).to(device)
        criterion = YOLOv3Loss(model.anchors, model.anchor_slices, C).to(device)

    if args.backbone_weights:
        ckpt = torch.load(args.backbone_weights, map_location=device)
        if ckpt.get("model", args.model) != args.model:
            raise ValueError(f"{args.backbone_weights} holds a {ckpt['model']} backbone, not {args.model}")
        model.backbone.load_state_dict(ckpt["backbone"])
        print(f"backbone initialised from {args.backbone_weights} (pretrain val acc {ckpt.get('val_acc', float('nan')):.3f})")
    return model, criterion


def main(args: argparse.Namespace):
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = Path(args.save_dir or f"runs/bee_yolo{args.model}")
    print(f"device: {device}")

    train_loader, val_loader = build_loaders(args)
    model, criterion = build_model(args, train_loader.dataset, device)
    print(f"YOLO{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    eval_kwargs = dict(
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        iou_threshold=IOU_THRESHOLD,
    )

    if args.eval_only:
        weights = Path(args.weights) if args.weights else save_dir / "best.pt"
        ckpt = torch.load(weights, map_location=device)
        model.load_state_dict(ckpt["model"])
        stats = evaluate(model, criterion, val_loader, device, max_batches=args.eval_batches, **eval_kwargs)
        print(
            f"mAP@{IOU_THRESHOLD}: {stats['ap']:.4f}  loss {stats['loss']:.4f}  "
            f"precision {stats['precision']:.4f}  recall {stats['recall']:.4f}  "
            f"({stats['n_pred']} predictions / {stats['n_gt']} objects)"
        )
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    total_steps = max(1, len(train_loader) * EPOCHS)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:  # avoid the early divergence the paper warns about
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(args.min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -1 so the very first evaluation always produces a best.pt to fall back on
    step, start_epoch, best_map = 0, 0, -1.0
    if args.resume:
        ckpt_path = Path(args.weights) if args.weights else save_dir / "last.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step, start_epoch, best_map = ckpt["step"], ckpt["epoch"], ckpt["best_map"]
        print(f"resumed from {ckpt_path} at step {step} (best mAP {best_map:.4f})")

    def run_eval(tag: str, epoch: int) -> float:
        nonlocal best_map
        stats = evaluate(model, criterion, val_loader, device, max_batches=args.eval_batches, **eval_kwargs)
        print(
            f"  [{tag}] mAP@{IOU_THRESHOLD} {stats['ap']:.4f}  val loss {stats['loss']:.4f}  "
            f"P {stats['precision']:.3f}  R {stats['recall']:.3f}"
        )
        if stats["ap"] > best_map:
            best_map = stats["ap"]
            save_checkpoint(
                save_dir / "best.pt", model, optimizer, scheduler, step, epoch, best_map, args, training_state=False
            )
            print(f"  new best mAP {best_map:.4f} → {save_dir / 'best.pt'}")
        model.train()
        return stats["ap"]

    print(f"training for {EPOCHS} epochs ({total_steps} steps)")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{EPOCHS}")

        for batch in bar:
            images = batch["image"].to(device, non_blocking=True)
            targets = loss_target(model, batch, device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, parts = criterion(model(images), targets)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            seen += 1
            batch_loss = float(loss.detach())
            running += batch_loss
            bar.set_postfix(
                loss=f"{batch_loss:.3f}",
                **{k: f"{v:.2f}" for k, v in parts.items()},
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            if step % args.save_every == 0:
                save_checkpoint(save_dir / "last.pt", model, optimizer, scheduler, step, epoch, best_map, args)
            if args.eval_every and step % args.eval_every == 0:
                run_eval(f"step {step}", epoch)
            if args.max_steps and step >= args.max_steps:
                break

        print(f"epoch {epoch + 1}: train loss {running / max(1, seen):.4f}  ({time.time() - t0:.0f}s)")
        run_eval(f"epoch {epoch + 1}", epoch)
        save_checkpoint(save_dir / "last.pt", model, optimizer, scheduler, step, epoch + 1, best_map, args)
        if args.max_steps and step >= args.max_steps:
            break

    print(f"done. best mAP@{IOU_THRESHOLD}: {best_map:.4f}")


if __name__ == "__main__":
    main(parser.parse_args())
