from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import os

import requests
from tqdm import tqdm


HF_RAW_BASE = "https://huggingface.co/datasets/AIpourlapeau/acne04v2/resolve/main"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/AIpourlapeau/acne04v2/main"


def _download(url: str, dst: Path, *, hf_token: Optional[str] = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    if hf_token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {hf_token}"
    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", "0") or "0")
        with open(dst, "wb") as f:
            pbar = tqdm(total=total if total > 0 else None, unit="B", unit_scale=True, desc=dst.name)
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
            pbar.close()


def _try_urls(rel_path: str, out: Path, *, hf_token: Optional[str]) -> None:
    errors = []
    for base in (HF_RAW_BASE, GITHUB_RAW_BASE):
        url = f"{base}/{rel_path}"
        try:
            _download(url, out, hf_token=hf_token)
            return
        except Exception as e:  # noqa: BLE001
            errors.append((url, str(e)))
    msg = "\n".join([f"- {u}: {err}" for u, err in errors])
    raise RuntimeError(f"Failed to download {rel_path}. Tried:\n{msg}")


def _try_many(rel_paths: list[str], out: Path, *, hf_token: Optional[str]) -> str:
    """
    Try multiple relative paths; return the one that succeeded.
    """
    last_err: Optional[Exception] = None
    for rp in rel_paths:
        try:
            _try_urls(rp, out, hf_token=hf_token)
            return rp
        except Exception as e:  # noqa: BLE001
            last_err = e
    assert last_err is not None
    raise last_err


def download_acne04v2(
    out_dir: Path,
    *,
    images_rel_dir: str = "images",
    annotations_name: str = "Acne04-v2_annotations.json",
    hf_token: Optional[str] = None,
    max_images: Optional[int] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = out_dir / annotations_name
    if not annotations_path.exists():
        _try_urls(annotations_name, annotations_path, hf_token=hf_token)

    annotations = json.loads(annotations_path.read_text())
    images = annotations.get("images", [])
    if not images:
        raise RuntimeError("No 'images' field found in annotations json.")

    images_dir = out_dir / images_rel_dir
    images_dir.mkdir(parents=True, exist_ok=True)

    if max_images is not None:
        images = images[: int(max_images)]

    for img in tqdm(images, desc="images", unit="img"):
        fname = img["file_name"]
        dst = images_dir / fname
        if dst.exists():
            continue
        # Acne04-v2 annotations historically use a typo "levle" instead of "level".
        # Also, some distributions provide annotations without images; skip missing files and report.
        candidates = [f"{images_rel_dir}/{fname}"]
        if "levle" in fname:
            candidates.append(f"{images_rel_dir}/{fname.replace('levle', 'level')}")
        try:
            _try_many(candidates, dst, hf_token=hf_token)
        except Exception as e:  # noqa: BLE001
            # Keep going; the user may provide images from another source (original ACNE04).
            tqdm.write(f"[warn] missing image {fname}: {e}")
            continue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="Output directory (will contain images/ + Acne04-v2_annotations.json)")
    ap.add_argument("--max-images", type=int, default=None, help="Optional cap for quicker experiments")
    ap.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token (or set env HUGGINGFACE_TOKEN). Required if dataset is gated.",
    )
    args = ap.parse_args()
    token = args.hf_token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    download_acne04v2(Path(args.out_dir), hf_token=token, max_images=args.max_images)


if __name__ == "__main__":
    main()

