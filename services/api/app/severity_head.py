from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal


SeverityBucket = Literal["mild", "moderate", "severe"]


class SeverityHeadUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SeverityPred:
    bucket: SeverityBucket
    probs: Dict[SeverityBucket, float]


_CACHED = None


def _load():
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    try:
        import torch
        import torch.nn as nn
    except Exception as e:  # pragma: no cover
        raise SeverityHeadUnavailable(f"torch unavailable: {e}") from e

    ckpt_path = Path(__file__).resolve().parents[1] / "models" / "severity_head.pt"
    if not ckpt_path.exists():
        raise SeverityHeadUnavailable(f"missing weights at {ckpt_path}")

    class _MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Must match training architecture (checkpoint keys: net.0.*, net.3.*)
            self.net = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Dropout(p=0.0),
                nn.Linear(16, 3),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x)

    model = _MLP()

    obj = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        raise SeverityHeadUnavailable(f"failed to load severity head weights: {e}") from e
    model.eval()

    _CACHED = (model, torch)
    return _CACHED


def predict_severity(features_8: "object") -> SeverityPred:
    """
    features_8: array-like length 8 (float32 preferred)
    Output classes are ordered: mild, moderate, severe.
    """
    model, torch = _load()

    x = torch.as_tensor(features_8, dtype=torch.float32).reshape(1, 8)
    with torch.no_grad():
        logits = model(x)
        probs_t = torch.softmax(logits, dim=1)[0]

    probs: Dict[SeverityBucket, float] = {
        "mild": float(probs_t[0].item()),
        "moderate": float(probs_t[1].item()),
        "severe": float(probs_t[2].item()),
    }
    bucket: SeverityBucket = max(probs, key=lambda k: probs[k])  # type: ignore[assignment]
    return SeverityPred(bucket=bucket, probs=probs)

