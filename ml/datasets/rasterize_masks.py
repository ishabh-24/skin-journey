from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float


def _load_annotations(data_dir: Path) -> dict:
    ann = data_dir / "Acne04-v2_annotations.json"
    if not ann.exists():
        raise FileNotFoundError(f"Missing {ann}. Provide --annotations for other formats (e.g. COCO).")
    return json.loads(ann.read_text())


def _circles_by_image_id(ann: dict) -> Dict[int, List[Circle]]:
    by: Dict[int, List[Circle]] = {}
    for a in ann.get("annotations", []):
        image_id = int(a["image_id"])
        coords = a.get("coordinates") or a.get("center") or a.get("coord") or a.get("points")
        if coords is None:
            # expected fields: coordinates=[x,y], radius=r
            continue
        cx, cy = float(coords[0]), float(coords[1])
        r = float(a.get("radius", 0.0))
        by.setdefault(image_id, []).append(Circle(cx=cx, cy=cy, r=r))
    return by


def _images_by_id(ann: dict) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for img in ann.get("images", []):
        out[int(img["id"])] = img
    return out


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _coco_images_by_id(coco: dict) -> Dict[int, dict]:
    return {int(i["id"]): i for i in coco.get("images", [])}


def _coco_ann_by_image_id(coco: dict) -> Dict[int, list[dict]]:
    by: Dict[int, list[dict]] = {}
    for a in coco.get("annotations", []):
        by.setdefault(int(a["image_id"]), []).append(a)
    return by


def _resolve_image_path(images_dir: Path, file_name: str) -> Optional[Path]:
    """
    Attempt to find an image path even when exporters rename files
    (e.g. Roboflow adds `.rf...` suffix).
    """
    p = images_dir / file_name
    if p.exists():
        return p

    # Try to match by stem prefix.
    stem = Path(file_name).stem
    # common exporter change: `foo.jpg` -> `foo_jpg.rf....jpg`
    candidates = list(images_dir.glob(f"{stem}*"))
    if candidates:
        # pick first deterministic
        return sorted(candidates)[0]

    # last resort: look one level deeper (train/valid/test)
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


def rasterize(
    *,
    data_dir: Path,
    out_dir: Path,
    size: int = 512,
    images_subdir: str = "images",
    masks_subdir: str = "masks",
    write_overlays: bool = False,
    annotations_path: Optional[Path] = None,
    fmt: str = "acne04v2",
) -> None:
    if annotations_path is None:
        if fmt != "acne04v2":
            raise ValueError("--annotations is required for non-acne04v2 formats")
        ann = _load_annotations(data_dir)
    else:
        ann = _load_json(annotations_path)

    images_dir = data_dir / images_subdir
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir {images_dir}")

    wrote = 0
    missing = 0

    if fmt == "acne04v2":
        circles = _circles_by_image_id(ann)
        images = _images_by_id(ann)
        iterator = images.items()
    elif fmt == "coco":
        images = _coco_images_by_id(ann)
        ann_by_img = _coco_ann_by_image_id(ann)
        iterator = images.items()
    else:
        raise ValueError(f"Unknown fmt {fmt}")

    mask_dir = out_dir / masks_subdir
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    if write_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    for image_id, meta in tqdm(iterator, desc="rasterize", unit="img"):
        fname = meta["file_name"]
        img_path = _resolve_image_path(images_dir, fname)
        if img_path is None or not img_path.exists():
            missing += 1
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        mask = np.zeros((h, w), dtype=np.uint8)

        if fmt == "acne04v2":
            for c in circles.get(image_id, []):
                if c.r <= 0:
                    continue
                cv2.circle(mask, (int(round(c.cx)), int(round(c.cy))), int(round(c.r)), 255, thickness=-1)
        else:
            # COCO: prefer polygon segmentation; otherwise bbox mask.
            for a in ann_by_img.get(image_id, []):
                seg = a.get("segmentation")
                if isinstance(seg, list) and seg and isinstance(seg[0], list):
                    # polygon list-of-lists
                    for poly in seg:
                        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
                else:
                    bbox = a.get("bbox")
                    if bbox and len(bbox) == 4:
                        x, y, bw, bh = bbox
                        try:
                            x = float(x)
                            y = float(y)
                            bw = float(bw)
                            bh = float(bh)
                        except Exception:  # noqa: BLE001
                            continue
                        x0 = int(max(0, round(x)))
                        y0 = int(max(0, round(y)))
                        x1 = int(min(w, round(x + bw)))
                        y1 = int(min(h, round(y + bh)))
                        if x1 > x0 and y1 > y0:
                            mask[y0:y1, x0:x1] = 255

        # resize to training size
        img_r = img.resize((size, size))
        mask_r = Image.fromarray(mask, mode="L").resize((size, size), resample=Image.NEAREST)

        mask_out = mask_dir / (Path(fname).stem + ".png")
        mask_r.save(mask_out)
        wrote += 1

        if write_overlays:
            arr = np.asarray(img_r).copy()
            m = (np.asarray(mask_r) > 0).astype(np.uint8)
            arr[:, :, 0] = np.clip(arr[:, :, 0] + m * 80, 0, 255)
            Image.fromarray(arr).save(overlay_dir / (Path(fname).stem + ".jpg"))

    print(f"wrote_masks={wrote} missing_images={missing} out_dir={mask_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Base directory containing images/")
    ap.add_argument(
        "--annotations",
        default=None,
        help="Path to annotations JSON. If omitted, defaults to Acne04-v2_annotations.json inside --data-dir",
    )
    ap.add_argument("--format", default="acne04v2", choices=["acne04v2", "coco"], help="Annotation format")
    ap.add_argument("--out-dir", required=True, help="Output directory for masks/")
    ap.add_argument("--size", type=int, default=512, help="Output square resolution")
    ap.add_argument("--images-subdir", default="images", help="Subdirectory under --data-dir where images live")
    ap.add_argument("--write-overlays", action="store_true", help="Write sample overlays for inspection")
    args = ap.parse_args()

    rasterize(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        size=args.size,
        images_subdir=str(args.images_subdir),
        write_overlays=bool(args.write_overlays),
        annotations_path=Path(args.annotations) if args.annotations else None,
        fmt=str(args.format),
    )


if __name__ == "__main__":
    main()

