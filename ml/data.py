from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path


def _resolve_image_for_mask_stem(all_images: List[Path], mask_stem: str, by_stem: dict[str, Path]) -> Optional[Path]:
    """
    Match mask levle0_196.png to an on-disk image.

    Acne04 / Roboflow often renames files (e.g. levle0_196_jpg.rf-....jpg) while SAM masks use the
    annotation file_name stem (levle0_196). Same resolution idea as ml/datasets/rasterize_masks.py.
    """
    if mask_stem in by_stem:
        return by_stem[mask_stem]
    prefixed = [p for p in all_images if p.stem.startswith(mask_stem + "_")]
    if prefixed:
        return sorted(prefixed, key=lambda p: (len(p.stem), p.name))[0]
    dotted = [p for p in all_images if p.name.lower().startswith(mask_stem.lower() + ".")]
    if dotted:
        return sorted(dotted, key=lambda p: (len(p.name), p.name))[0]
    return None


def _list_pairs(images_dir: Path, masks_dir: Path) -> List[Sample]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
    ]
    by_stem = {p.stem: p for p in images}
    pairs: List[Sample] = []
    for m in sorted(masks_dir.glob("*.png")):
        img = _resolve_image_for_mask_stem(images, m.stem, by_stem)
        if img:
            pairs.append(Sample(image_path=img, mask_path=m))
    return sorted(pairs, key=lambda s: s.image_path.name)


class AcneSegmentationDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, *, augment: bool = False):
        self.samples = _list_pairs(images_dir, masks_dir)
        if not self.samples:
            exts = {".jpg", ".jpeg", ".png", ".webp"}
            n_img = len(
                [p for p in Path(images_dir).rglob("*") if p.is_file() and p.suffix.lower() in exts]
            )
            n_msk = len(list(Path(masks_dir).glob("*.png")))
            parts = [
                f"No image/mask pairs under images={images_dir!s} masks={masks_dir!s}",
                f"(found {n_img} image files, {n_msk} mask PNGs).",
            ]
            if n_img == 0 and n_msk > 0:
                parts.append(
                    "Images are missing: download/copy RGB files into images/ so each mask stem "
                    "(e.g. levle0_196.png) matches an image stem (e.g. levle0_196.jpg). "
                    "See ml/README.md — python -m ml.datasets.acne04v2_download --out-dir data/acne04v2"
                )
            elif n_msk == 0:
                parts.append("Mask directory is empty: run rasterize_masks or acne04v2_sam_masks first.")
            elif n_img > 0 and n_msk > 0:
                parts.append(
                    "No mask stem matched an image: masks use annotation stems (e.g. levle0_196.png); "
                    "images should be levle0_196.jpg or Roboflow-style levle0_196_....jpg under images/. "
                    "Train from repo root with paths ml/data/... or cd ml and use data/acne04v2/images."
                )
            raise RuntimeError(" ".join(parts))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        mask = Image.open(s.mask_path).convert("L")
        # Ensure consistent tensor sizes for batching by resizing image to mask size.
        # (Many datasets store varying-resolution images, but masks are already normalized to a fixed size.)
        img = Image.open(s.image_path).convert("RGB").resize(mask.size, resample=Image.BILINEAR)

        x = np.asarray(img).astype(np.float32) / 255.0
        y = (np.asarray(mask).astype(np.float32) / 255.0)[..., None]

        # lightweight aug (no extra deps)
        if self.augment:
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=1).copy()
                y = np.flip(y, axis=1).copy()
            # brightness/contrast jitter
            if np.random.rand() < 0.7:
                b = 0.9 + 0.2 * np.random.rand()
                c = 0.9 + 0.2 * np.random.rand()
                x = np.clip((x * c) * b, 0.0, 1.0)

        x_t = torch.from_numpy(x).permute(2, 0, 1)  # C,H,W
        y_t = torch.from_numpy(y).permute(2, 0, 1)  # 1,H,W
        return x_t, y_t

