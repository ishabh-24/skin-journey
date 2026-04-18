"""
OpenAI Vision-based severity classifier.

Sends the uploaded skin image to GPT-4o and asks for a structured
severity assessment (bucket + numeric score).  Falls back gracefully
when OPENAI_API_KEY is missing or the request fails.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass

from PIL import Image

logger = logging.getLogger(__name__)

# Max edge length before JPEG encode (keeps payloads reasonable for the API).
_MAX_IMAGE_SIDE = 1536
_JPEG_QUALITY = 88


class OpenAIClassifierUnavailable(Exception):
    """Raised when the OpenAI classifier cannot be used."""


@dataclass(frozen=True)
class OpenAISeverityResult:
    """Final bucket always matches ``severity_score_0_10`` (same bands as ``analysis``)."""

    severity_bucket: str  # "mild" | "moderate" | "severe"
    severity_score_0_10: float  # 0.0 – 10.0
    model_bucket_matched_score: bool  # False if the model's bucket disagreed with the score band


_SYSTEM_PROMPT = """\
You are a board-certified dermatologist AI assistant. You will be shown a \
photograph of a person's face. Evaluate the acne severity visible in the image.

Return ONLY a JSON object with exactly two keys:
  - "severity_bucket": one of "mild", "moderate", or "severe"
  - "severity_score_0_10": a float between 0.0 and 10.0

Grading guidelines:
  • mild   (0–3.4): mostly clear skin, a few small comedones or papules
  • moderate (3.5–6.4): noticeable papules/pustules, moderate inflammation, \
multiple active lesions
  • severe  (6.5–10): widespread pustules/cysts, significant redness and \
swelling, scarring

The numeric score is authoritative; choose the bucket that matches your score \
under these bands.

Be accurate and unbiased. Do NOT default to mild.
Respond with ONLY the JSON object—no markdown fences, no extra text.
"""


def _bucket_from_score_0_10(score: float) -> str:
    """Same thresholds as ``analysis.analyze_image_bytes`` (before OpenAI override)."""
    if score < 3.5:
        return "mild"
    if score < 6.5:
        return "moderate"
    return "severe"


def _image_bytes_to_jpeg_data_uri(image_bytes: bytes) -> str:
    """Decode arbitrary image bytes, normalize to RGB JPEG, correct MIME for the API."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        w, h = img.size
        m = max(w, h)
        if m > _MAX_IMAGE_SIDE:
            scale = _MAX_IMAGE_SIDE / m
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        jpeg_bytes = buf.getvalue()
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _extract_json_object(raw: str) -> str:
    """Handle markdown fences, optional language tags, leading/trailing prose, one-line fences."""
    text = raw.strip()
    # Opening ``` or ```json
    text = re.sub(r"^\s*```[a-zA-Z0-9]*\s*", "", text)
    text = text.strip()
    # Closing fence (handles single-line ```{...}```)
    if text.endswith("```"):
        text = text[: text.rfind("```")].strip()
    # Brace slice if model added a preamble (e.g. "Here is the JSON: {...}")
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return text.strip()


def classify_severity(image_bytes: bytes) -> OpenAISeverityResult:
    """Call the OpenAI API to classify acne severity from an image.

    Raises ``OpenAIClassifierUnavailable`` when the API key is missing
    or the request fails for any reason, so callers can fall back to
    local scoring.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIClassifierUnavailable("OPENAI_API_KEY not set")

    # Lazy import so the module can be imported even without the
    # openai package installed (the import error is caught below).
    try:
        from openai import OpenAI  # type: ignore[import-untyped]
    except ImportError as exc:
        raise OpenAIClassifierUnavailable(
            "openai package is not installed"
        ) from exc

    try:
        data_uri = _image_bytes_to_jpeg_data_uri(image_bytes)
    except Exception as exc:
        raise OpenAIClassifierUnavailable(f"could not decode image for OpenAI: {exc}") from exc

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
                            "text": "Please evaluate the acne severity in this image.",
                        },
                    ],
                },
            ],
            max_tokens=150,
            temperature=0.0,
        )

        raw = response.choices[0].message.content or ""
        raw_json = _extract_json_object(raw)
        parsed = json.loads(raw_json)

        model_bucket = str(parsed["severity_bucket"]).lower()
        if model_bucket not in ("mild", "moderate", "severe"):
            raise ValueError(f"Unexpected bucket: {model_bucket}")

        score = float(parsed["severity_score_0_10"])
        score = max(0.0, min(10.0, score))

        derived_bucket = _bucket_from_score_0_10(score)
        matched = model_bucket == derived_bucket
        if not matched:
            logger.info(
                "OpenAI bucket/score reconcile: model_bucket=%s score=%.2f -> using_bucket=%s",
                model_bucket,
                score,
                derived_bucket,
            )

        logger.info("OpenAI severity => bucket=%s  score=%.1f", derived_bucket, score)
        return OpenAISeverityResult(
            severity_bucket=derived_bucket,
            severity_score_0_10=score,
            model_bucket_matched_score=matched,
        )

    except Exception as exc:
        logger.warning("OpenAI classifier failed: %s", exc, exc_info=True)
        raise OpenAIClassifierUnavailable(str(exc)) from exc
