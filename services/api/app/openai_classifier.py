"""
OpenAI Vision-based severity classifier.

Sends the uploaded skin image to GPT-4o and asks for a structured
severity assessment (bucket + numeric score).  Falls back gracefully
when OPENAI_API_KEY is missing or the request fails.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class OpenAIClassifierUnavailable(Exception):
    """Raised when the OpenAI classifier cannot be used."""


@dataclass(frozen=True)
class OpenAISeverityResult:
    severity_bucket: str        # "mild" | "moderate" | "severe"
    severity_score_0_10: float  # 0.0 – 10.0


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

Be accurate and unbiased. Do NOT default to mild.
Respond with ONLY the JSON object—no markdown fences, no extra text.
"""


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

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64_image}"

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
        # Strip potential markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        parsed = json.loads(raw)
        bucket = str(parsed["severity_bucket"]).lower()
        if bucket not in ("mild", "moderate", "severe"):
            raise ValueError(f"Unexpected bucket: {bucket}")

        score = float(parsed["severity_score_0_10"])
        score = max(0.0, min(10.0, score))

        logger.info("OpenAI severity => bucket=%s  score=%.1f", bucket, score)
        return OpenAISeverityResult(severity_bucket=bucket, severity_score_0_10=score)

    except Exception as exc:
        logger.warning("OpenAI classifier failed: %s", exc, exc_info=True)
        raise OpenAIClassifierUnavailable(str(exc)) from exc
