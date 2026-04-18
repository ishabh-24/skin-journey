from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


class ModelUnavailable(Exception):
    pass


@dataclass(frozen=True)
class MaskPrediction:
    prob_mask: np.ndarray  # HxW float32 0..1


_MODEL = None
_DEVICE = None


def _get_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "unet_acne.pt"


def _load_model():
    global _MODEL, _DEVICE  # noqa: PLW0603
    if _MODEL is not None:
        return _MODEL

    model_path = _get_model_path()
    if not model_path.exists():
        raise ModelUnavailable(f"Model weights not found at {model_path}")

    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise ModelUnavailable("torch not installed in API environment") from e

    from .unet_torch import UNet

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=_DEVICE)
    model = UNet(in_ch=3, out_ch=1, base=32).to(_DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    _MODEL = model
    return _MODEL


def predict_prob_mask(image_bytes: bytes, *, size: int = 256) -> MaskPrediction:
    """
    Returns a square probability mask at `size`x`size`.
    """
    model = _load_model()

    import torch

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((size, size))
    x = np.asarray(img).astype(np.float32) / 255.0
    x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    x_t = x_t.to(_DEVICE)

    with torch.no_grad():
        logits = model(x_t)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

    return MaskPrediction(prob_mask=probs)

