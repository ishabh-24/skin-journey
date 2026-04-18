from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "ml" / ".mplcache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import AcneSegmentationDataset
from metrics import dice_coeff, iou
from unet import UNet


def load_model(ckpt_path: Path, device: torch.device) -> UNet:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = UNet(in_ch=3, out_ch=1, base=32).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    m = (mask > 0).astype(np.uint8)
    out[:, :, 0] = np.clip(out[:, :, 0] + m * 80, 0, 255)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = AcneSegmentationDataset(Path(args.images), Path(args.masks), augment=False)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=int(args.num_workers))

    model = load_model(Path(args.ckpt), device)

    dices = []
    ious = []
    saved = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="eval"):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()
            dices.append(dice_coeff(pred, y).mean().item())
            ious.append(iou(pred, y).mean().item())

            if saved < args.limit:
                x_np = (x.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
                p_np = pred.detach().cpu().numpy()[:, 0, :, :]
                y_np = y.detach().cpu().numpy()[:, 0, :, :]
                for i in range(x_np.shape[0]):
                    if saved >= args.limit:
                        break
                    fig = plt.figure(figsize=(9, 3))
                    ax1 = fig.add_subplot(1, 3, 1)
                    ax2 = fig.add_subplot(1, 3, 2)
                    ax3 = fig.add_subplot(1, 3, 3)
                    ax1.imshow(x_np[i])
                    ax1.set_title("image")
                    ax1.axis("off")
                    ax2.imshow(overlay(x_np[i], y_np[i]))
                    ax2.set_title("gt overlay")
                    ax2.axis("off")
                    ax3.imshow(overlay(x_np[i], p_np[i]))
                    ax3.set_title("pred overlay")
                    ax3.axis("off")
                    fig.tight_layout()
                    fig.savefig(out_dir / f"sample_{saved:04d}.png", dpi=150)
                    plt.close(fig)
                    saved += 1

    dice_m = float(sum(dices) / max(1, len(dices)))
    iou_m = float(sum(ious) / max(1, len(ious)))
    (out_dir / "metrics.txt").write_text(f"dice={dice_m:.4f}\niou={iou_m:.4f}\n")
    print(f"dice={dice_m:.4f} iou={iou_m:.4f}")


if __name__ == "__main__":
    main()

