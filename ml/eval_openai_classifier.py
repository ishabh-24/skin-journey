"""
Run OpenAI vision severity on sample images (e.g. Acne04 overlays or raw images).

  PYTHONPATH=services/api python ml/eval_openai_classifier.py \\
      --images ml/data/acne04v2_sam_masks/overlays --per-level 2

Loads ``OPENAI_API_KEY`` from the environment, or from ``--dotenv`` (default: repo/.env).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = Path(__file__).resolve().parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

API_DIR = REPO_ROOT / "services" / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _iter_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts and "levle" in p.name.lower():
            yield p


def _levle_from_name(name: str) -> int | None:
    m = re.search(r"levle(\d)", name, re.I)
    if not m:
        return None
    return int(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--images",
        type=Path,
        default=REPO_ROOT / "ml/data/acne04v2_sam_masks/overlays",
        help="Directory of images (recursive), filenames with levle* for labels",
    )
    ap.add_argument("--per-level", type=int, default=2, help="Max images per levle 0..3")
    ap.add_argument(
        "--dotenv",
        type=Path,
        default=REPO_ROOT / ".env",
        help="Optional .env path (only sets vars missing from the environment)",
    )
    args = ap.parse_args()

    _load_dotenv(args.dotenv.expanduser().resolve())

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (export it or use --dotenv with a .env file).")

    from app.openai_classifier import OpenAIClassifierUnavailable, classify_severity

    images_dir = args.images.expanduser().resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"not a directory: {images_dir}")

    by_level: dict[int, list[Path]] = {0: [], 1: [], 2: [], 3: []}
    for p in _iter_images(images_dir):
        lv = _levle_from_name(p.name)
        if lv is None or lv not in by_level:
            continue
        if len(by_level[lv]) < args.per_level:
            by_level[lv].append(p)

    rows: list[tuple[Path, int | None, str, float, bool]] = []
    for lv in sorted(by_level):
        for p in by_level[lv]:
            data = p.read_bytes()
            try:
                r = classify_severity(data)
                rows.append(
                    (
                        p,
                        lv,
                        r.severity_bucket,
                        r.severity_score_0_10,
                        r.model_bucket_matched_score,
                    )
                )
            except OpenAIClassifierUnavailable as e:
                rows.append((p, lv, f"FAIL: {e}", float("nan"), False))

    w = max(len(str(p)) for p, *_ in rows) if rows else 10
    print(f"images_dir={images_dir}")
    print(f"{'path':<{w}}  levle  bucket   score  model_bucket_ok")
    for p, lv, bucket, score, ok in rows:
        lv_s = "" if lv is None else str(lv)
        if isinstance(bucket, str) and bucket.startswith("FAIL"):
            print(f"{str(p):<{w}}  {lv_s:<5}  {bucket}")
        else:
            print(
                f"{str(p):<{w}}  {lv_s:<5}  {bucket:<8}  {score:4.1f}  "
                f"{'yes' if ok else 'no'}"
            )


if __name__ == "__main__":
    main()
