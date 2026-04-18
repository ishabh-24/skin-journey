"""
Smoke-eval the trained severity head on a few images per Acne04 `levle` bucket.

Run from repo root or from ml/:

  python ml/eval_severity_head_samples.py --images ml/data/acne04v2/images --per-level 3
  cd ml && python eval_severity_head_samples.py --images data/acne04v2/images --per-level 3

Requires: services/api/models/unet_acne.pt and services/api/models/severity_head.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ML_DIR = Path(__file__).resolve().parent
REPO_ROOT = ML_DIR.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import torch
import torch.nn as nn

from train_severity_head import (  # noqa: E402
    _bucket_3class,
    _label_from_filename,
    featurize_image_path_use_unet,
)


def _bucket_name(y: int) -> str:
    return ("mild", "moderate", "severe")[y]


def _iter_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts and "levle" in p.name.lower():
            yield p


def _load_severity_mlp(ckpt_path: Path) -> tuple[nn.Module, int]:
    obj = torch.load(str(ckpt_path), map_location="cpu")
    state = obj["model"]
    meta = obj.get("meta", {})
    in_dim = int(meta.get("in_dim", 12))
    out_dim = int(state["net.3.weight"].shape[0])

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 16),
                nn.ReLU(),
                nn.Dropout(p=0.0),
                nn.Linear(16, out_dim),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x)

    m = MLP()
    m.load_state_dict(state, strict=True)
    m.eval()
    return m, out_dim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True, help="Acne04 images directory (recursive)")
    ap.add_argument(
        "--severity-ckpt",
        type=Path,
        default=None,
        help="Default: <repo>/services/api/models/severity_head.pt",
    )
    ap.add_argument("--per-level", type=int, default=3, help="Max samples per levle digit 0..3")
    args = ap.parse_args()

    sev_path = args.severity_ckpt or (REPO_ROOT / "services" / "api" / "models" / "severity_head.pt")
    if not sev_path.exists():
        raise SystemExit(f"missing severity head: {sev_path}")

    images_dir = args.images.expanduser().resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"not a directory: {images_dir}")

    by_level: dict[int, list[Path]] = {0: [], 1: [], 2: [], 3: []}
    for p in _iter_images(images_dir):
        try:
            lvl = _label_from_filename(p.name)
        except ValueError:
            continue
        if lvl in by_level and len(by_level[lvl]) < int(args.per_level):
            by_level[lvl].append(p)

    ordered: list[tuple[Path, int]] = []
    for lvl in (0, 1, 2, 3):
        for p in by_level[lvl]:
            ordered.append((p, lvl))

    if not ordered:
        raise SystemExit(
            f"No matching images under {images_dir} (need filenames containing levle0..3). "
            "Download/extract Acne04 images there, then re-run."
        )

    model, out_dim = _load_severity_mlp(sev_path)
    names = ["mild", "moderate", "severe"][:out_dim]

    print(f"severity_ckpt={sev_path}  unet={REPO_ROOT / 'services' / 'api' / 'models' / 'unet_acne.pt'}")
    print(f"{'file':<42} {'levle':>5} {'true_3cls':>10} {'pred':>10}  " + " ".join(f"p_{n:>8}" for n in names))
    correct = 0
    for path, lvl in ordered:
        x = featurize_image_path_use_unet(path, size=256)
        xt = torch.from_numpy(x).float().unsqueeze(0)
        with torch.no_grad():
            logits = model(xt)
            pr = torch.softmax(logits, dim=1)[0].numpy()
        pred_i = int(np.argmax(pr))
        true_y = _bucket_3class(lvl)
        if pred_i == true_y:
            correct += 1
        prob_str = " ".join(f"{float(pr[i]):8.3f}" for i in range(len(names)))
        print(
            f"{path.name:<42} {lvl:5d} {_bucket_name(true_y):>10} {_bucket_name(pred_i):>10}  {prob_str}"
        )
    print(f"\nacc_on_printed_subset={correct}/{len(ordered)}")


if __name__ == "__main__":
    main()
