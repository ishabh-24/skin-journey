"""
OpenAI Vision-based atopic dermatitis / eczema pattern assessment.

Classifies visible facial skin into coarse buckets and falls back when
``OPENAI_API_KEY`` is missing or the request fails.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from .openai_vision_util import extract_json_object, image_bytes_to_jpeg_data_uri

logger = logging.getLogger(__name__)

_VALID_BUCKETS = frozenset({"none", "mild_eczema", "severe_eczema"})


class OpenAIEczemaClassifierUnavailable(Exception):
    """Raised when the eczema vision classifier cannot be used."""


@dataclass(frozen=True)
class OpenAIEczemaResult:
    """Bucket is aligned to ``eczema_likelihood_0_10`` after parsing."""

    eczema_bucket: str  # "none" | "mild_eczema" | "severe_eczema"
    eczema_likelihood_0_10: float  # 0 = very unlikely eczema pattern, 10 = very likely / severe flare
    model_bucket_matched_score: bool


_SYSTEM_PROMPT = """\
You are a dermatology-focused vision assistant. You will see a photograph \
of a person's face (selfie lighting and angle may vary).

Task: assess how consistent the image is with **eczema (atopic dermatitis)** \
on visible facial skin — not acne. Use texture, redness patterns, scale/crust, \
ill-defined plaques, periorbital or cheek involvement typical of eczema flares, \
and spared vs involved areas.

Return ONLY a JSON object with exactly two keys:
  - "eczema_bucket": one of:
      - "none" — normal skin or generic dryness/oiliness without a pattern \
suggestive of eczema; do not label routine flaking or isolated dry patches as \
eczema without supportive signs.
      - "mild_eczema" — eczema-suggestive dryness/scaling or subtle patches with \
limited erythema, without an intense widespread flare.
      - "severe_eczema" — prominent erythema, irritation, weeping/crusting, or \
widespread inflamed eczematous involvement on the face.

  - "eczema_likelihood_0_10": a float from 0.0 to 10.0 expressing overall \
confidence that an **eczema-type** pattern is present and how active/severe \
it appears (0 = no eczema pattern, 10 = strong / severe eczema pattern).

Guidance for the numeric score (use it consistently with the bucket):
  • 0.0–2.4  → bucket should be "none"
  • 2.5–6.4  → bucket should be "mild_eczema"
  • 6.5–10.0 → bucket should be "severe_eczema"

Be conservative: prefer "none" when findings are ambiguous or could be \
non-eczema irritation or poor image quality. Do NOT equate acne papules alone \
with eczema.

Respond with ONLY the JSON object—no markdown fences, no extra text.
"""


def _bucket_from_eczema_score(score: float) -> str:
    score = max(0.0, min(10.0, score))
    if score < 2.5:
        return "none"
    if score < 6.5:
        return "mild_eczema"
    return "severe_eczema"


def _normalize_bucket_label(raw: str) -> str:
    s = raw.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "not_at_all": "none",
        "no": "none",
        "absent": "none",
        "clear": "none",
        "normal": "none",
        "mild": "mild_eczema",
        "mildeczema": "mild_eczema",
        "moderate": "mild_eczema",
        "severe": "severe_eczema",
        "severe_eczema": "severe_eczema",
        "severeflare": "severe_eczema",
    }
    s = aliases.get(s, s)
    if s in _VALID_BUCKETS:
        return s
    raise ValueError(f"Unexpected eczema bucket label: {raw!r}")


def classify_eczema(image_bytes: bytes) -> OpenAIEczemaResult:
    """Call OpenAI vision for eczema-pattern classification."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIEczemaClassifierUnavailable("OPENAI_API_KEY not set")

    try:
        from openai import OpenAI  # type: ignore[import-untyped]
    except ImportError as exc:
        raise OpenAIEczemaClassifierUnavailable(
            "openai package is not installed"
        ) from exc

    try:
        data_uri = image_bytes_to_jpeg_data_uri(image_bytes)
    except Exception as exc:
        raise OpenAIEczemaClassifierUnavailable(
            f"could not decode image for OpenAI: {exc}"
        ) from exc

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri, "detail": "low"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Does this facial photo show patterns consistent "
                                "with eczema (atopic dermatitis), and how severe? "
                                "Return only the JSON specified in your instructions."
                            ),
                        },
                    ],
                },
            ],
            max_tokens=200,
            temperature=0.0,
        )

        raw = response.choices[0].message.content or ""
        raw_json = extract_json_object(raw)
        parsed = json.loads(raw_json)

        model_bucket = _normalize_bucket_label(str(parsed["eczema_bucket"]))
        score = float(parsed["eczema_likelihood_0_10"])
        score = max(0.0, min(10.0, score))

        derived = _bucket_from_eczema_score(score)
        matched = model_bucket == derived
        if not matched:
            logger.info(
                "OpenAI eczema reconcile: model_bucket=%s score=%.2f -> bucket=%s",
                model_bucket,
                score,
                derived,
            )

        logger.info("OpenAI eczema => bucket=%s likelihood=%.1f", derived, score)
        return OpenAIEczemaResult(
            eczema_bucket=derived,
            eczema_likelihood_0_10=score,
            model_bucket_matched_score=matched,
        )

    except Exception as exc:
        logger.warning("OpenAI eczema classifier failed: %s", exc, exc_info=True)
        raise OpenAIEczemaClassifierUnavailable(str(exc)) from exc
