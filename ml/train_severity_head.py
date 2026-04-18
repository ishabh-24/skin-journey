from __future__ import annotations

import argparse
import json
import io
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def _label_from_filename(name: str) -> int:
    """
    Acne04-v2 convention: filenames like 'levle2_123_...jpg'
    Returns integer level 0..3.
    """
    base = Path(name).name.lower()
    if "levle" not in base:
        raise ValueError(f"cannot parse label from filename: {name}")
    i = base.index("levle") + len("levle")
    if i >= len(base) or not base[i].isdigit():
        raise ValueError(f"cannot parse label from filename: {name}")
    return int(base[i])


def _bucket_3class(level_0_3: int) -> int:
    """
    Map 4 levels -> 3 classes: mild/moderate/severe
    - level 0,1 => mild
    - level 2   => moderate
    - level 3   => severe
    """
    if level_0_3 <= 1:
        return 0
    if level_0_3 == 2:
        return 1
    return 2


def _to_rgb01(img: Image.Image, max_side: int = 512) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def _simple_face_crop(arr: np.ndarray) -> np.ndarray:
    """
    Same heuristic as API: use a centered region approximating the face.
    Returns cropped [Hf, Wf, 3].
    """
    h, w, _ = arr.shape
    y0, y1 = int(0.12 * h), int(0.95 * h)
    x0, x1 = int(0.10 * w), int(0.90 * w)
    return arr[y0:y1, x0:x1]


def _redness_map(rgb01: np.ndarray) -> np.ndarray:
    r = rgb01[:, :, 0]
    g = rgb01[:, :, 1]
    b = rgb01[:, :, 2]
    return np.clip(r - 0.5 * g - 0.2 * b, 0.0, 1.0)


def _features_from_image_and_inflammation(img: Image.Image, infl01: np.ndarray) -> np.ndarray:
    """
    Produce the same 8-dim features used at inference time:
    [mass, topk, area_t20, area_t35, red_mean, red_top, red_mean*mass, red_top*topk]

    `infl01` is a 2D array in 0..1 (e.g. U-Net prob mask) that will be resized to match the face crop.
    """
    rgb01 = _to_rgb01(img, max_side=512)
    rgb_face = _simple_face_crop(rgb01)
    red = _redness_map(rgb_face)

    mh, mw = rgb_face.shape[:2]
    infl_img = Image.fromarray((np.clip(infl01, 0.0, 1.0) * 255).astype(np.uint8), mode="L").resize(
        (mw, mh), resample=Image.BILINEAR
    )
    infl = np.asarray(infl_img).astype(np.float32) / 255.0

    p = infl.reshape(-1)
    k = max(1, int(0.02 * p.size))
    topk = float(np.mean(np.partition(p, -k)[-k:]))
    mass = float(np.mean(infl))
    area_t20 = float(np.mean(infl > 0.20))
    area_t35 = float(np.mean(infl > 0.35))

    r = red.reshape(-1)
    red_mean = float(np.mean(red))
    red_top = float(np.mean(np.partition(r, -k)[-k:]))

    feats = np.asarray(
        [mass, topk, area_t20, area_t35, red_mean, red_top, red_mean * mass, red_top * topk],
        dtype=np.float32,
    )
    return feats


def _index_masks(mask_dir: Path) -> Dict[str, Path]:
    """
    Masks are named like '<image_stem>.png' but stems can include .jpg in the name
    (e.g. 'levle0_21_jpg.rf....png'). We key by full stem.
    """
    masks: Dict[str, Path] = {}
    for p in sorted(mask_dir.glob("*.png")):
        masks[p.stem] = p
    return masks


def _build_examples(images_dir: Path, masks_dir: Path | None) -> List[Tuple[Path, Path | None, int]]:
    masks = _index_masks(masks_dir) if masks_dir is not None else {}
    ex: List[Tuple[Path, Path | None, int]] = []

    for img_path in sorted(images_dir.rglob("*.jpg")):
        stem = img_path.stem
        mask_path = masks.get(stem) if masks_dir is not None else None
        if masks_dir is not None and mask_path is None:
            continue
        lvl = _label_from_filename(img_path.name)
        y = _bucket_3class(lvl)
        ex.append((img_path, mask_path, y))

    if not ex:
        raise RuntimeError(f"no training images found under images={images_dir}")
    return ex


def _split(examples: List[Tuple[Path, Path | None, int]], seed: int = 1337) -> Tuple[list, list]:
    rng = random.Random(seed)
    # simple shuffle split; with this dataset size it's fine for a quick hackathon model
    ex = examples[:]
    rng.shuffle(ex)
    n_val = max(1, int(0.2 * len(ex)))
    return ex[n_val:], ex[:n_val]


@dataclass(frozen=True)
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    class_counts: List[float]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Images directory (expects .jpg). Can include train/valid subfolders.")
    ap.add_argument("--masks", default="", help="Optional GT masks directory. If provided, can be used as fallback.")
    ap.add_argument("--out", required=True, help="Output path for checkpoint (.pt).")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num-workers", type=int, default=0)  # kept for compatibility; forced to 0
    ap.add_argument("--use-unet", action="store_true", help="Featurize using U-Net probability masks (matches API).")
    args = ap.parse_args()

    images_dir = Path(args.images)
    masks_dir = Path(args.masks) if str(args.masks).strip() else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    examples = _build_examples(images_dir, masks_dir)
    train_ex, val_ex = _split(examples, seed=args.seed)

    _unet_cached = None

    # Precompute features (fast enough for this dataset; keeps training simple)
    def compute(ex_list):
        X = np.zeros((len(ex_list), 8), dtype=np.float32)
        y = np.zeros((len(ex_list),), dtype=np.int64)
        if args.use_unet:
            import torch

            try:
                # When running as `python ml/train_severity_head.py`, `ml` may not be importable.
                from unet import UNet  # type: ignore
            except Exception:  # noqa: BLE001
                from ml.unet import UNet  # type: ignore

            def predict_prob_mask_bytes(image_bytes: bytes, *, size: int = 256) -> np.ndarray:
                nonlocal _unet_cached
                if _unet_cached is None:
                    ckpt_path = Path("services/api/models/unet_acne.pt")
                    if not ckpt_path.exists():
                        raise RuntimeError(f"missing U-Net weights at {ckpt_path}")
                    ckpt = torch.load(str(ckpt_path), map_location="cpu")
                    model = UNet(in_ch=3, out_ch=1, base=32)
                    model.load_state_dict(ckpt["model"], strict=True)
                    model.eval()
                    _unet_cached = (model, torch)

                model, torch_mod = _unet_cached
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((size, size))
                x = np.asarray(img).astype(np.float32) / 255.0
                x_t = torch_mod.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
                with torch_mod.no_grad():
                    logits = model(x_t)
                    prob = torch_mod.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
                return prob

        for i, (ip, mp, yy) in enumerate(ex_list):
            img = Image.open(ip)
            if args.use_unet:
                with ip.open("rb") as f:
                    b = f.read()
                infl01 = predict_prob_mask_bytes(b, size=256)
            else:
                if mp is None:
                    raise RuntimeError("masks are required when --use-unet is not set")
                mask = Image.open(mp)
                infl01 = (np.asarray(mask.convert("L")).astype(np.float32) / 255.0)

            X[i] = _features_from_image_and_inflammation(img, infl01)
            y[i] = yy
            if (i + 1) % 200 == 0 or (i + 1) == len(ex_list):
                print(f"featurized {i+1}/{len(ex_list)}", flush=True)
        return X, y

    Xtr, ytr = compute(train_ex)
    Xva, yva = compute(val_ex)

    class_counts = np.bincount(ytr, minlength=3).astype(np.float32)
    # inverse frequency weights, normalized
    w = (class_counts.sum() / np.clip(class_counts, 1.0, None))
    w = w / w.mean()

    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Dropout(p=0.0),
                nn.Linear(16, 3),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x)

    device = torch.device("cpu")
    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32))

    def batches(X, y, bs):
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        for s in range(0, len(X), bs):
            j = idx[s : s + bs]
            yield X[j], y[j]

    best_acc = -1.0
    history: List[EpochLog] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses: List[float] = []
        for xb, yb in batches(Xtr, ytr, int(args.batch_size)):
            x = torch.as_tensor(xb, dtype=torch.float32, device=device)
            yt = torch.as_tensor(yb, dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, yt)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=device)
            yva_t = torch.as_tensor(yva, dtype=torch.long, device=device)
            logits = model(xva_t)
            val_loss = float(loss_fn(logits, yva_t).item())
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            val_acc = float((pred == yva).mean())

        log = EpochLog(
            epoch=epoch,
            train_loss=float(np.mean(train_losses)) if train_losses else float("nan"),
            val_loss=val_loss,
            val_acc=val_acc,
            class_counts=[float(x) for x in class_counts.tolist()],
        )
        history.append(log)
        print(json.dumps(asdict(log)), flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                "model": model.state_dict(),
                "meta": {
                    "features": [
                        "mass",
                        "topk",
                        "area_t20",
                        "area_t35",
                        "red_mean",
                        "red_top",
                        "red_mean_x_mass",
                        "red_top_x_topk",
                    ],
                    "label_mapping": {"0": "mild", "1": "moderate", "2": "severe"},
                    "source": {"images": str(images_dir), "masks": str(masks_dir)},
                },
            }
            torch.save(ckpt, str(out_path))
            # keep a tiny training trace next to the checkpoint
            hist_path = out_path.with_suffix(".history.json")
            with hist_path.open("w") as f:
                json.dump([asdict(h) for h in history], f, indent=2)


if __name__ == "__main__":
    # Avoid MKL thread oversubscription on laptops
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

