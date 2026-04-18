"""
Generate high-quality pixel-level masks for Acne04-v2 by prompting SAM
(Segment Anything) with the expert circle annotations.

Input:
  data-dir/
    Acne04-v2_annotations.json      # {images:[{id,file_name,width,height}...],
                                    #  annotations:[{image_id,coordinates:[x,y],radius,class_name}...]}
    images/*.jpg                    # Roboflow-renamed copies OK; resolver matches by stem

Output:
  out-dir/
    masks/<image_stem>.png          # single-channel {0,255} binary mask, resized to --size
    overlays/<image_stem>.jpg       # (optional) visual sanity-check overlays

Prompting strategy per lesion:
  - positive point   = circle center
  - tight box prompt = [cx-r, cy-r, cx+r, cy+r]  (clamped to image bounds)
  - multimask_output = False  (take single best mask)
  - discard masks whose area is > max_lesion_area_frac of image area
    (SAM occasionally segments the whole face when prompt is ambiguous).

SAM variants supported:
  - segment_anything:  vit_b / vit_l / vit_h
  - mobile_sam:        vit_t (small, fast; good for 1000+ images on CPU/MPS)

Typical run (MobileSAM, fast):
  python -m ml.datasets.acne04v2_sam_masks \
      --data-dir ml/data/acne04v2 \
      --out-dir  ml/data/acne04v2_sam_masks \
      --sam-variant mobile_sam \
      --sam-ckpt ml/data/sam_ckpts/mobile_sam.pt \
      --size 512 --max-side 1024 --write-overlays --overlay-every 50
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float


# ---------------------------------------------------------------------------
# IO helpers (match rasterize_masks.py semantics for filename resolution)
# ---------------------------------------------------------------------------

def _load_annotations(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing annotations json: {path}")
    return json.loads(path.read_text())


def _circles_by_image_id(ann: dict) -> Dict[int, List[Circle]]:
    by: Dict[int, List[Circle]] = {}
    for a in ann.get("annotations", []):
        image_id = int(a["image_id"])
        coords = a.get("coordinates") or a.get("center") or a.get("coord")
        if coords is None or len(coords) < 2:
            continue
        r = float(a.get("radius", 0.0))
        if r <= 0:
            continue
        by.setdefault(image_id, []).append(Circle(cx=float(coords[0]), cy=float(coords[1]), r=r))
    return by


def _images_by_id(ann: dict) -> Dict[int, dict]:
    return {int(img["id"]): img for img in ann.get("images", [])}


def _resolve_image_path(images_dir: Path, file_name: str) -> Optional[Path]:
    p = images_dir / file_name
    if p.exists():
        return p
    stem = Path(file_name).stem
    candidates = list(images_dir.glob(f"{stem}*"))
    if candidates:
        return sorted(candidates)[0]
    # look one level deeper (train/valid/test)
    for sub in ("train", "valid", "test"):
        subdir = images_dir / sub
        if not subdir.exists():
            continue
        p2 = subdir / file_name
        if p2.exists():
            return p2
        candidates = list(subdir.glob(f"{stem}*"))
        if candidates:
            return sorted(candidates)[0]
    return None


# ---------------------------------------------------------------------------
# SAM predictor wrapper (supports segment_anything + mobile_sam)
# ---------------------------------------------------------------------------

def _pick_device(user_choice: str) -> torch.device:
    if user_choice != "auto":
        return torch.device(user_choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS is often slower/less stable for SAM image encoder; prefer CPU fallback
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_sam_predictor(variant: str, ckpt_path: Path, device: torch.device):
    """
    Returns an object with:
      .set_image(rgb_ndarray)
      .predict(point_coords=..., point_labels=..., box=..., multimask_output=...)
          -> (masks, scores, logits)
    where masks is (N, H, W) bool.
    """
    variant = variant.lower()
    if variant in {"vit_b", "vit_l", "vit_h"}:
        try:
            from segment_anything import SamPredictor, sam_model_registry  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "segment_anything not installed. Run:\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from e
        sam = sam_model_registry[variant](checkpoint=str(ckpt_path))
        sam.to(device=device)
        return SamPredictor(sam)

    if variant in {"mobile_sam", "vit_t"}:
        try:
            from mobile_sam import SamPredictor, sam_model_registry  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "mobile_sam not installed. Run:\n"
                "  pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
            ) from e
        sam = sam_model_registry["vit_t"](checkpoint=str(ckpt_path))
        sam.to(device=device)
        sam.eval()
        return SamPredictor(sam)

    raise ValueError(f"Unknown --sam-variant: {variant}")


# ---------------------------------------------------------------------------
# Core rasterization with SAM
# ---------------------------------------------------------------------------

def _downscale_if_needed(img: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Downscale RGB image so that max(h,w) <= max_side. Returns (resized, scale)."""
    h, w = img.shape[:2]
    m = max(h, w)
    if max_side <= 0 or m <= max_side:
        return img, 1.0
    scale = max_side / m
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _batched(it: Iterable, n: int):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def _predict_circle_masks(
    predictor,
    rgb: np.ndarray,
    circles: List[Circle],
    scale: float,
    max_lesion_area_frac: float,
    min_score: float,
) -> np.ndarray:
    """Run SAM one lesion at a time. Returns binary mask uint8 {0,255}."""
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # SAM requires one set_image per image
    predictor.set_image(rgb)

    img_area = float(h * w)

    for c in circles:
        cx = c.cx * scale
        cy = c.cy * scale
        r = c.r * scale
        # clamp box to image
        x0 = max(0.0, cx - r)
        y0 = max(0.0, cy - r)
        x1 = min(w - 1.0, cx + r)
        y1 = min(h - 1.0, cy + r)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue

        point_coords = np.array([[cx, cy]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        box = np.array([x0, y0, x1, y1], dtype=np.float32)

        try:
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=False,
            )
        except Exception:  # noqa: BLE001
            # fall back to filled circle
            cv2.circle(mask, (int(round(cx)), int(round(cy))), max(1, int(round(r))), 255, thickness=-1)
            continue

        if masks is None or len(masks) == 0:
            cv2.circle(mask, (int(round(cx)), int(round(cy))), max(1, int(round(r))), 255, thickness=-1)
            continue

        m = masks[0].astype(bool)
        score = float(scores[0]) if scores is not None and len(scores) else 0.0

        area_frac = float(m.sum()) / max(1.0, img_area)
        if score < min_score or area_frac > max_lesion_area_frac or area_frac <= 0:
            # SAM likely produced a junk/huge mask. Fall back to disk from the circle.
            cv2.circle(mask, (int(round(cx)), int(round(cy))), max(1, int(round(r))), 255, thickness=-1)
            continue

        mask[m] = 255

    return mask


def run(
    *,
    data_dir: Path,
    out_dir: Path,
    annotations_path: Optional[Path],
    images_subdir: str,
    sam_variant: str,
    sam_ckpt: Path,
    device_str: str,
    size: int,
    max_side: int,
    write_overlays: bool,
    overlay_every: int,
    max_lesion_area_frac: float,
    min_score: float,
    limit: int,
) -> None:
    ann_path = annotations_path or (data_dir / "Acne04-v2_annotations.json")
    ann = _load_annotations(ann_path)
    circles_by_id = _circles_by_image_id(ann)
    images_by_id = _images_by_id(ann)
    images_dir = data_dir / images_subdir
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")

    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    if write_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    device = _pick_device(device_str)
    print(f"[sam] variant={sam_variant} ckpt={sam_ckpt} device={device}")
    predictor = _load_sam_predictor(sam_variant, sam_ckpt, device)

    wrote = 0
    missing = 0
    no_anns = 0
    items = list(images_by_id.items())
    if limit > 0:
        items = items[:limit]

    for idx, (image_id, meta) in enumerate(tqdm(items, desc="sam-rasterize", unit="img")):
        fname = meta["file_name"]
        img_path = _resolve_image_path(images_dir, fname)
        if img_path is None:
            missing += 1
            continue

        circles = circles_by_id.get(image_id, [])
        if not circles:
            no_anns += 1
            continue

        pil = Image.open(img_path).convert("RGB")
        rgb_full = np.asarray(pil)  # H,W,3 uint8

        rgb, scale = _downscale_if_needed(rgb_full, max_side)
        try:
            mask = _predict_circle_masks(
                predictor,
                rgb,
                circles,
                scale=scale,
                max_lesion_area_frac=max_lesion_area_frac,
                min_score=min_score,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rgb_small, scale2 = _downscale_if_needed(rgb_full, max(256, max_side // 2))
            mask = _predict_circle_masks(
                predictor,
                rgb_small,
                circles,
                scale=scale2,
                max_lesion_area_frac=max_lesion_area_frac,
                min_score=min_score,
            )
            rgb = rgb_small

        # Resize to training --size
        mask_pil = Image.fromarray(mask, mode="L").resize((size, size), resample=Image.NEAREST)
        rgb_pil = Image.fromarray(rgb).resize((size, size), resample=Image.BILINEAR)

        out_name = Path(fname).stem + ".png"
        mask_pil.save(mask_dir / out_name)
        wrote += 1

        if write_overlays and (idx % max(1, overlay_every) == 0):
            arr = np.asarray(rgb_pil).copy()
            m = (np.asarray(mask_pil) > 0).astype(np.uint8)
            arr[:, :, 0] = np.clip(arr[:, :, 0].astype(np.int32) + m * 80, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(overlay_dir / (Path(fname).stem + ".jpg"))

    print(f"[sam] wrote_masks={wrote} missing_images={missing} images_without_annotations={no_anns}")
    print(f"[sam] out_dir={mask_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Base dir containing Acne04-v2_annotations.json + images/")
    ap.add_argument("--out-dir", required=True, help="Output dir. Masks go under out-dir/masks/")
    ap.add_argument("--annotations", default=None, help="Override annotations json path")
    ap.add_argument("--images-subdir", default="images")

    ap.add_argument(
        "--sam-variant",
        default="mobile_sam",
        choices=["mobile_sam", "vit_t", "vit_b", "vit_l", "vit_h"],
        help="Which SAM checkpoint family to load",
    )
    ap.add_argument("--sam-ckpt", required=True, help="Path to the SAM checkpoint (.pt / .pth)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])

    ap.add_argument("--size", type=int, default=512, help="Output square resolution for mask+image")
    ap.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="Downscale images so max(h,w) <= this before SAM (speeds up CPU/MPS).",
    )
    ap.add_argument(
        "--max-lesion-area-frac",
        type=float,
        default=0.01,
        help="Reject single-lesion SAM masks bigger than this fraction of the image area "
        "(falls back to rasterized circle). 0.01 = 1%%.",
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Reject single-lesion SAM masks with predicted IoU below this threshold.",
    )
    ap.add_argument("--write-overlays", action="store_true")
    ap.add_argument("--overlay-every", type=int, default=50, help="Write overlay every N images")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N images (0 = all)")

    args = ap.parse_args()

    run(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        annotations_path=Path(args.annotations) if args.annotations else None,
        images_subdir=args.images_subdir,
        sam_variant=args.sam_variant,
        sam_ckpt=Path(args.sam_ckpt),
        device_str=args.device,
        size=int(args.size),
        max_side=int(args.max_side),
        write_overlays=bool(args.write_overlays),
        overlay_every=int(args.overlay_every),
        max_lesion_area_frac=float(args.max_lesion_area_frac),
        min_score=float(args.min_score),
        limit=int(args.limit),
    )


if __name__ == "__main__":
    main()
