from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Sample:
    image_path: Path
    mask_path: Path


def _list_pairs(images_dir: Path, masks_dir: Path) -> List[Sample]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    by_stem = {p.stem: p for p in images}
    pairs: List[Sample] = []
    for m in masks_dir.glob("*.png"):
        img = by_stem.get(m.stem)
        if img:
            pairs.append(Sample(image_path=img, mask_path=m))
    return sorted(pairs, key=lambda s: s.image_path.name)


class AcneSegmentationDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, *, augment: bool = False):
        self.samples = _list_pairs(images_dir, masks_dir)
        if not self.samples:
            raise RuntimeError(f"No image/mask pairs found in {images_dir} + {masks_dir}")
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

