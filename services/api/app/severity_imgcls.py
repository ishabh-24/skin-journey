from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal

import numpy as np
from PIL import Image


SeverityBucket = Literal["mild", "moderate", "severe"]


class SeverityImgClsUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SeverityImgClsPred:
    level_pred_0_3: int
    level_probs_0_3: Dict[int, float]
    bucket: SeverityBucket
    bucket_probs: Dict[SeverityBucket, float]


_CACHED = None


def _load():
    """
    Loads a TorchScript model saved at services/api/models/severity_imgcls.ts
    so the API runtime does NOT need torchvision.
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise SeverityImgClsUnavailable(f"torch unavailable: {e}") from e

    ts_path = Path(__file__).resolve().parents[1] / "models" / "severity_imgcls.ts"
    if not ts_path.exists():
        raise SeverityImgClsUnavailable(f"missing TorchScript weights at {ts_path}")

    try:
        model = torch.jit.load(str(ts_path), map_location="cpu")
        model.eval()
    except Exception as e:
        raise SeverityImgClsUnavailable(f"failed to load TorchScript model: {e}") from e

    _CACHED = (model, torch)
    return _CACHED


def predict_level_from_image_bytes(image_bytes: bytes) -> SeverityImgClsPred:
    model, torch = _load()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    x = np.asarray(img).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W

    with torch.no_grad():
        logits = model(x_t)
        probs_t = torch.softmax(logits, dim=1)[0]

    level_probs = {i: float(probs_t[i].item()) for i in range(4)}
    level_pred = int(max(level_probs, key=lambda k: level_probs[k]))

    bucket_probs: Dict[SeverityBucket, float] = {
        "mild": float(level_probs[0] + level_probs[1]),
        "moderate": float(level_probs[2]),
        "severe": float(level_probs[3]),
    }
    bucket: SeverityBucket = max(bucket_probs, key=lambda k: bucket_probs[k])  # type: ignore[assignment]

    return SeverityImgClsPred(level_pred_0_3=level_pred, level_probs_0_3=level_probs, bucket=bucket, bucket_probs=bucket_probs)

