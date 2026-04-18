from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import AcneSegmentationDataset
from metrics import dice_coeff, iou
from unet import UNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def train(
    *,
    images_dir: Path,
    masks_dir: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    val_frac: float,
    num_workers: int,
    bce_pos_weight: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    ds = AcneSegmentationDataset(images_dir, masks_dir, augment=True)
    n_val = max(1, int(len(ds) * val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = UNet(in_ch=3, out_ch=1, base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    if bce_pos_weight > 0:
        pw = torch.tensor([bce_pos_weight], device=device, dtype=torch.float32)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        bce = nn.BCEWithLogitsLoss()

    best = -math.inf
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"train {epoch}/{epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = 0.5 * bce(logits, y) + 0.5 * dice_loss_from_logits(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += float(loss.detach().cpu().item())
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        dices = []
        dices_soft = []
        dices_t035 = []
        ious = []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"val {epoch}/{epochs}", leave=False):
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = 0.5 * bce(logits, y) + 0.5 * dice_loss_from_logits(logits, y)
                val_loss += float(loss.detach().cpu().item())
                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float()
                dices.append(dice_coeff(pred, y).mean().item())
                dices_soft.append(dice_coeff(probs, y).mean().item())
                dices_t035.append(dice_coeff((probs > 0.35).float(), y).mean().item())
                ious.append(iou(pred, y).mean().item())
        val_loss /= max(1, len(val_loader))
        dice_m = float(sum(dices) / max(1, len(dices)))
        dice_soft_m = float(sum(dices_soft) / max(1, len(dices_soft)))
        dice_t035_m = float(sum(dices_t035) / max(1, len(dices_t035)))
        iou_m = float(sum(ious) / max(1, len(ious)))

        # Hard @0.5 is harsh when logits stay <0.5 early; soft / 0.35 track real progress.
        score = max(dice_m, dice_soft_m, dice_t035_m)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dice": dice_m,
            "dice_soft": dice_soft_m,
            "dice_t0.35": dice_t035_m,
            "iou": iou_m,
        }
        history.append(row)

        print(json.dumps(row))

        if score > best:
            best = score
            ckpt = {
                "model": model.state_dict(),
                "meta": {
                    "epoch": epoch,
                    "dice": dice_m,
                    "dice_soft": dice_soft_m,
                    "dice_t0.35": dice_t035_m,
                    "iou": iou_m,
                    "images_dir": str(images_dir),
                    "masks_dir": str(masks_dir),
                    "bce_pos_weight": bce_pos_weight,
                },
            }
            torch.save(ckpt, out_dir / "best.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Images directory (RGB)")
    ap.add_argument("--masks", required=True, help="Masks directory (PNG masks)")
    ap.add_argument("--out-dir", required=True, help="Output directory for checkpoints/logs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument(
        "--bce-pos-weight",
        type=float,
        default=80.0,
        help="BCE positive class weight (~#bg/#fg). Use 0 to disable. Helps sparse lesion masks.",
    )
    args = ap.parse_args()

    train(
        images_dir=Path(args.images),
        masks_dir=Path(args.masks),
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        val_frac=args.val_frac,
        num_workers=args.num_workers,
        bce_pos_weight=float(args.bce_pos_weight),
    )


if __name__ == "__main__":
    main()

