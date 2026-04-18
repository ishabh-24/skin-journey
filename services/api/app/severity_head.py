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
    probs: Dict[str, float]


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
                nn.Linear(16, 3),  # placeholder; replaced after reading checkpoint
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x)

    obj = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    out_dim = int(state_dict["net.3.weight"].shape[0]) if isinstance(state_dict, dict) else 3
    meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
    in_dim = int(meta.get("in_dim", 8))

    class _MLPOut(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 16),
                nn.ReLU(),
                nn.Dropout(p=0.0),
                nn.Linear(16, out_dim),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x)

    model = _MLPOut(in_dim, out_dim)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        raise SeverityHeadUnavailable(f"failed to load severity head weights: {e}") from e
    model.eval()

    label_mapping = meta.get("label_mapping", {})
    _CACHED = (model, torch, out_dim, label_mapping, in_dim)
    return _CACHED


def predict_severity(features_8: "object") -> SeverityPred:
    """
    features_8: array-like length 8 (float32 preferred)
    Output classes are ordered: mild, moderate, severe.
    """
    model, torch, out_dim, label_mapping, in_dim = _load()

    x = torch.as_tensor(features_8, dtype=torch.float32).reshape(1, int(in_dim))
    with torch.no_grad():
        logits = model(x)
        probs_t = torch.softmax(logits, dim=1)[0]

    probs_list = [float(probs_t[i].item()) for i in range(out_dim)]

    # 3-class mild/moderate/severe (recommended for stability)
    p0, p1, p2 = probs_list[:3]
    probs_bucket = {"mild": p0, "moderate": p1, "severe": p2}

    bucket: SeverityBucket = max(probs_bucket, key=lambda k: probs_bucket[k])  # type: ignore[assignment]
    return SeverityPred(bucket=bucket, probs=probs_bucket)

