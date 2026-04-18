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


def _bucket_4class(level_0_3: int) -> int:
    """
    Identity mapping: 4 levels -> 4 classes.
    """
    if level_0_3 < 0 or level_0_3 > 3:
        raise ValueError(f"level out of range 0..3: {level_0_3}")
    return level_0_3


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


def _count_components(binary: np.ndarray) -> int:
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    cnt = 0
    for y0 in range(h):
        for x0 in range(w):
            if binary[y0, x0] == 0 or visited[y0, x0] == 1:
                continue
            cnt += 1
            stack = [(y0, x0)]
            visited[y0, x0] = 1
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and visited[ny, nx] == 0 and binary[ny, nx] == 1:
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
    return cnt


def _normalize01(x: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(x, 5))
    hi = float(np.percentile(x, 95))
    if hi - lo < 1e-6:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _texture_map(img: Image.Image) -> np.ndarray:
    from PIL import ImageFilter

    gray = img.convert("L").filter(ImageFilter.FIND_EDGES)
    return np.asarray(gray).astype(np.float32) / 255.0


def _index_masks(mask_dir: Path) -> Dict[str, Path]:
    """
    Masks are named like '<image_stem>.png' but stems can include .jpg in the name
    (e.g. 'levle0_21_jpg.rf....png'). We key by full stem.
    """
    masks: Dict[str, Path] = {}
    for p in sorted(mask_dir.glob("*.png")):
        masks[p.stem] = p
    return masks


def featurize_from_pil_and_infl(img: Image.Image, infl01: np.ndarray) -> np.ndarray:
    """12-dim vector aligned with API / training (U-Net or GT mask as infl01)."""
    base8 = _features_from_image_and_inflammation(img, infl01)

    rgb01 = _to_rgb01(img, max_side=512)
    face_rgb = _simple_face_crop(rgb01)
    mh, mw = face_rgb.shape[:2]
    infl_img = Image.fromarray((np.clip(infl01, 0.0, 1.0) * 255).astype(np.uint8), mode="L").resize(
        (mw, mh), resample=Image.BILINEAR
    )
    infl_face = np.asarray(infl_img).astype(np.float32) / 255.0

    intensity = float(np.mean(infl_face))
    lesion_bin = (infl_face > 0.5).astype(np.uint8)
    lesion_area = float(np.mean(lesion_bin))
    small = Image.fromarray((lesion_bin * 255).astype(np.uint8), mode="L").resize((160, 160), resample=Image.NEAREST)
    lesion_count = float(_count_components((np.asarray(small) > 0).astype(np.uint8)))

    face_img = Image.fromarray((face_rgb * 255).astype(np.uint8), mode="RGB")
    tex = _normalize01(_texture_map(face_img))
    p = infl_face.reshape(-1)
    k = max(1, int(0.02 * p.size))
    tex_p = tex.reshape(-1)
    tex_top = float(np.mean(np.partition(tex_p, -k)[-k:]))
    tex_mean = float(np.mean(tex))

    red_top_times_topk = float(base8[5] * base8[1])

    return np.asarray(
        [
            float(base8[0]),
            float(base8[1]),
            float(base8[2]),
            float(base8[3]),
            float(base8[4]),
            float(base8[5]),
            lesion_area,
            lesion_count,
            intensity,
            tex_mean,
            tex_top,
            red_top_times_topk,
        ],
        dtype=np.float32,
    )


_FEAT_UNET_CACHE: Tuple[object, object] | None = None


def _get_unet_for_featurize():
    global _FEAT_UNET_CACHE
    if _FEAT_UNET_CACHE is None:
        import torch

        try:
            from unet import UNet  # type: ignore
        except Exception:  # noqa: BLE001
            from ml.unet import UNet  # type: ignore

        ckpt_path = _find_unet_ckpt()
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        model = UNet(in_ch=3, out_ch=1, base=32)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        _FEAT_UNET_CACHE = (model, torch)
    return _FEAT_UNET_CACHE


def featurize_image_path_use_unet(image_path: Path, *, size: int = 256) -> np.ndarray:
    """Run U-Net on bytes then build 12-dim features (same as training --use-unet)."""
    model, torch_mod = _get_unet_for_featurize()
    with image_path.open("rb") as f:
        b = f.read()
    img = Image.open(io.BytesIO(b))
    im_small = img.convert("RGB").resize((size, size))
    x = np.asarray(im_small).astype(np.float32) / 255.0
    x_t = torch_mod.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    with torch_mod.no_grad():
        logits = model(x_t)
        infl01 = torch_mod.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    return featurize_from_pil_and_infl(img, infl01)


def _build_examples(images_dir: Path, masks_dir: Path | None) -> List[Tuple[Path, Path | None, int]]:
    masks = _index_masks(masks_dir) if masks_dir is not None else {}
    ex: List[Tuple[Path, Path | None, int]] = []
    _img_exts = {".jpg", ".jpeg", ".png", ".webp"}

    for img_path in sorted(images_dir.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in _img_exts:
            continue
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


def _find_unet_ckpt() -> Path:
    """Resolve U-Net weights whether you run from repo root or ml/."""
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "services" / "api" / "models" / "unet_acne.pt",
        Path("services/api/models/unet_acne.pt"),
        Path("../services/api/models/unet_acne.pt"),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise RuntimeError(f"missing U-Net weights; tried: {[str(c) for c in candidates]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images",
        default="",
        help="Images directory (expects .jpg). Can include train/valid subfolders.",
    )
    ap.add_argument(
        "--data-dir",
        default="",
        help="Acne04-style layout: use <data-dir>/images (and optional <data-dir>/masks if --masks omitted). "
        "Alternative to --images.",
    )
    ap.add_argument("--masks", default="", help="Optional GT masks directory. If provided, can be used as fallback.")
    ap.add_argument("--out", required=True, help="Output path for checkpoint (.pt).")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num-workers", type=int, default=0)  # kept for compatibility; forced to 0
    ap.add_argument("--use-unet", action="store_true", help="Featurize using U-Net probability masks (matches API).")
    ap.add_argument("--num-classes", type=int, default=3, choices=[3, 4], help="Train 3 or 4 severity classes.")
    ap.add_argument(
        "--balanced-batches",
        action="store_true",
        help="Draw each training minibatch with replacement, probability ∝ 1/class_count. "
        "Stops the MLP from seeing almost-only mild examples. Uses plain CE (no loss class-weights) "
        "so you do not double-upweight rare classes.",
    )
    args = ap.parse_args()

    if bool(args.images) == bool(args.data_dir):
        ap.error("Provide exactly one of: --images DIR  or  --data-dir DIR (e.g. data/acne04v2)")

    if args.data_dir:
        images_dir = Path(args.data_dir) / "images"
        masks_dir = Path(args.masks) if str(args.masks).strip() else None
        if masks_dir is None and not args.use_unet:
            md = Path(args.data_dir) / "masks"
            if md.exists():
                masks_dir = md
    else:
        images_dir = Path(args.images)
        masks_dir = Path(args.masks) if str(args.masks).strip() else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Build examples and map labels
    examples_raw = _build_examples(images_dir, masks_dir)
    examples = []
    for ip, mp, _y3 in examples_raw:
        lvl = _label_from_filename(ip.name)
        if args.num_classes == 4:
            yy = _bucket_4class(lvl)
        else:
            yy = _bucket_3class(lvl)
        examples.append((ip, mp, yy))
    train_ex, val_ex = _split(examples, seed=args.seed)

    # Precompute features (fast enough for this dataset; keeps training simple)
    def compute(ex_list):
        X = np.zeros((len(ex_list), 12), dtype=np.float32)
        y = np.zeros((len(ex_list),), dtype=np.int64)
        for i, (ip, mp, yy) in enumerate(ex_list):
            img = Image.open(ip)
            if args.use_unet:
                X[i] = featurize_image_path_use_unet(ip, size=256)
            else:
                if mp is None:
                    raise RuntimeError("masks are required when --use-unet is not set")
                mask = Image.open(mp)
                infl01 = (np.asarray(mask.convert("L")).astype(np.float32) / 255.0)
                X[i] = featurize_from_pil_and_infl(img, infl01)
            y[i] = yy
            if (i + 1) % 200 == 0 or (i + 1) == len(ex_list):
                print(f"featurized {i+1}/{len(ex_list)}", flush=True)
        return X, y

    Xtr, ytr = compute(train_ex)
    Xva, yva = compute(val_ex)

    class_counts = np.bincount(ytr, minlength=int(args.num_classes)).astype(np.float64)
    # inverse frequency weights for loss (only when not using balanced batching)
    w = (class_counts.sum() / np.clip(class_counts, 1.0, None))
    # Option 1: Heavily penalize minority class errors with squared inverse frequency
    w = w ** 2
    w = w / w.mean()

    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(12, 16),
                nn.ReLU(),
                nn.Dropout(p=0.0),
                nn.Linear(16, int(args.num_classes)),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x)

    device = torch.device("cpu")
    model = MLP().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    if args.balanced_batches:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32))

    def batches_shuffle(X: np.ndarray, y: np.ndarray, bs: int, rng: np.random.Generator):
        idx = np.arange(len(X))
        rng.shuffle(idx)
        for s in range(0, len(X), bs):
            j = idx[s : s + bs]
            yield X[j], y[j]

    def batches_balanced(X: np.ndarray, y: np.ndarray, bs: int, rng: np.random.Generator):
        n = len(X)
        cc = np.bincount(y, minlength=int(args.num_classes)).astype(np.float64)
        inv = 1.0 / np.clip(cc[y], 1.0, None)
        p = inv / inv.sum()
        steps = int(np.ceil(n / bs))
        for _ in range(steps):
            j = rng.choice(n, size=bs, replace=True, p=p)
            yield X[j], y[j]

    best_acc = -1.0
    history: List[EpochLog] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_losses: List[float] = []
        rng = np.random.default_rng(int(args.seed) + epoch)
        batch_fn = batches_balanced if args.balanced_batches else batches_shuffle
        for xb, yb in batch_fn(Xtr, ytr, int(args.batch_size), rng):
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
            if int(args.num_classes) == 4:
                label_mapping = {"0": "level0", "1": "level1", "2": "level2", "3": "level3"}
            else:
                label_mapping = {"0": "mild", "1": "moderate", "2": "severe"}

            ckpt = {
                "model": model.state_dict(),
                "meta": {
                    "in_dim": 12,
                    "features": [
                        "mass",
                        "topk",
                        "area_t20",
                        "area_t35",
                        "red_mean",
                        "red_top",
                        "lesion_area_bin_t50",
                        "lesion_count_t50",
                        "intensity_mean",
                        "texture_mean",
                        "texture_top",
                        "red_top_x_topk",
                    ],
                    "label_mapping": label_mapping,
                    "num_classes": int(args.num_classes),
                    "balanced_batches": bool(args.balanced_batches),
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

