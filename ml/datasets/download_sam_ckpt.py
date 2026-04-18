"""
Download a SAM / MobileSAM checkpoint to a local path.

If your environment blocks the direct download URL (HTTP 403 from HF etc.),
just grab the file manually and place it at the --out path yourself:

  - MobileSAM (39 MB)   : https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt
  - SAM ViT-B (375 MB)  : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  - SAM ViT-L (1.2 GB)  : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
  - SAM ViT-H (2.4 GB)  : https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Usage:
  python -m ml.datasets.download_sam_ckpt --variant mobile_sam \
      --out ml/data/sam_ckpts/mobile_sam.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm


URLS = {
    "mobile_sam": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
    "vit_t": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}


def _download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        tmp = out.with_suffix(out.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=out.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))
        tmp.replace(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(URLS.keys()))
    ap.add_argument("--out", required=True, help="Destination .pt / .pth path")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"already exists: {out} ({out.stat().st_size/1e6:.1f} MB)")
        return

    url = URLS[args.variant]
    print(f"downloading {args.variant} -> {out}")
    try:
        _download(url, out)
    except Exception as e:  # noqa: BLE001
        print(f"download failed: {e}", file=sys.stderr)
        print("fallback: download it manually and move to --out. URL list in this file.", file=sys.stderr)
        raise
    print(f"done: {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
