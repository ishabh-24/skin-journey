from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def _rand_color(base: Tuple[int, int, int], jitter: int = 18) -> Tuple[int, int, int]:
    return tuple(int(np.clip(c + random.randint(-jitter, jitter), 0, 255)) for c in base)


def generate_sample(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (image_rgb_uint8, mask_uint8_0_255)
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    # Skin-ish background gradient + noise
    base = np.array([210, 170, 150], dtype=np.int16)
    yy = np.linspace(-1, 1, size).reshape(size, 1, 1)
    xx = np.linspace(-1, 1, size).reshape(1, size, 1)
    grad = (base + (yy * 10 + xx * 6)).astype(np.int16)
    noise = np.random.normal(0, 6, size=(size, size, 3)).astype(np.int16)
    img[:] = np.clip(grad + noise, 0, 255).astype(np.uint8)

    # Random lesions as circles (some red bumps)
    n = random.randint(6, 40)
    for _ in range(n):
        r = random.randint(3, 14)
        cx = random.randint(r + 2, size - r - 3)
        cy = random.randint(r + 2, size - r - 3)
        cv2.circle(mask, (cx, cy), r, 255, thickness=-1)

        color = _rand_color((190, 70, 70), jitter=30)
        cv2.circle(img, (cx, cy), r, color, thickness=-1)
        # subtle highlight
        cv2.circle(img, (cx - r // 3, cy - r // 3), max(1, r // 3), _rand_color((235, 190, 175), 10), thickness=-1)

    # Slight blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    out = Path(args.out_dir)
    images = out / "images"
    masks = out / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(args.n), desc="synthetic", unit="img"):
        img, mask = generate_sample(args.size)
        Image.fromarray(img, mode="RGB").save(images / f"syn_{i:05d}.jpg", quality=92)
        Image.fromarray(mask, mode="L").save(masks / f"syn_{i:05d}.png")


if __name__ == "__main__":
    main()

