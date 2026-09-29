"""Classification pretraining of the Darknet backbones on bee / background crops.

Example:
    python bee_pretrain.py --model v3 --epochs 10
    python bee_detection.py --model v3 --backbone_weights runs/pretrain_v3/backbone.pt

The papers pretrain at 224 and then fine-tune the classifier briefly at the
detection resolution; ``--crop_size`` lets you do the same in a second run.
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from bee_crops import BeeCropDataset
from paper import WEIGHT_DECAY
from yolov2 import Darknet19Classifier
from yolov3 import Darknet53Classifier

VAL_FRACTION = 0.1  # same split as detection, so no detection-val image leaks into pretraining

parser = argparse.ArgumentParser(description="Pretrain a Darknet backbone on bee / background crops")
parser.add_argument("--model", default="v3", choices=["v2", "v3"], help="Which backbone: Darknet-19 (v2) or Darknet-53 (v3).")
parser.add_argument("--data", default="Bees", type=str, help="Dataset root containing ds/img and ds/ann.")
parser.add_argument("--crop_size", default=224, type=int, help="Crops are resized to this; the papers pretrain at 224.")
parser.add_argument("--crops_per_image", default=4, type=int, help="Random crops drawn per image per epoch.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--max_steps", default=None, type=int, help="Stop after this many steps; useful for smoke tests.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
parser.add_argument("--lr", default=1e-3, type=float, help="Peak learning rate.")
parser.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup steps before the cosine decay.")
parser.add_argument("--workers", default=8, type=int, help="Dataloader workers.")
parser.add_argument("--amp", default=False, action="store_true", help="Mixed precision (CUDA only).")
parser.add_argument("--device", default=None, type=str, help="Torch device; autodetected when omitted.")
parser.add_argument("--seed", default=0, type=int, help="Random seed.")
parser.add_argument("--save_dir", default=None, type=str, help="Defaults to runs/pretrain_<model>.")
parser.add_argument("--resume_backbone", default=None, type=str, help="Start from an earlier backbone.pt (e.g. to fine-tune at a higher --crop_size).")


@torch.no_grad()
def accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for batch in tqdm(loader, desc="val", leave=False):
        logits = model(batch["image"].to(device, non_blocking=True))
        labels = batch["label"].to(device)
        correct += int((logits.argmax(1) == labels).sum())
        total += len(labels)
    model.train()
    return correct / max(total, 1)


def main(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    save_dir = Path(args.save_dir or f"runs/pretrain_{args.model}")
    print(f"device: {device}")

    common = dict(root=args.data, crop_size=args.crop_size, crops_per_image=args.crops_per_image,
                  val_fraction=VAL_FRACTION, seed=args.seed)
    train_set = BeeCropDataset(split="train", augment=True, **common)
    val_set = BeeCropDataset(split="val", augment=False, **common)
    print(f"train crops/epoch: {len(train_set)}   val crops: {len(val_set)}")
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                         persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    Classifier = Darknet19Classifier if args.model == "v2" else Darknet53Classifier
    model = Classifier(num_classes=2).to(device)
    if args.resume_backbone:
        model.backbone.load_state_dict(torch.load(args.resume_backbone, map_location=device)["backbone"])
        print(f"backbone initialised from {args.resume_backbone}")
    print(f"{args.model} backbone: {sum(p.numel() for p in model.backbone.parameters()) / 1e6:.1f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, len(train_loader) * args.epochs)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step, best_acc = 0, -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in bar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = F.cross_entropy(model(images), labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1
            seen += 1
            running += float(loss.detach())
            bar.set_postfix(loss=f"{float(loss):.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
            if args.max_steps and step >= args.max_steps:
                break

        acc = accuracy(model, val_loader, device)
        print(f"epoch {epoch + 1}: train loss {running / max(1, seen):.4f}  val acc {acc:.4f}  ({time.time() - t0:.0f}s)")
        if acc > best_acc:
            best_acc = acc
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"backbone": model.backbone.state_dict(), "model": args.model,
                 "crop_size": args.crop_size, "val_acc": acc, "args": vars(args)},
                save_dir / "backbone.pt",
            )
            print(f"  new best acc {acc:.4f} → {save_dir / 'backbone.pt'}")
        if args.max_steps and step >= args.max_steps:
            break

    print(f"done. best val acc {best_acc:.4f}")


if __name__ == "__main__":
    main(parser.parse_args())
