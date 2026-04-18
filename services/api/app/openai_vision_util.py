"""Shared helpers for OpenAI vision calls (JPEG data URIs, JSON extraction)."""

from __future__ import annotations

import base64
import io
import re

from PIL import Image

_MAX_IMAGE_SIDE = 1536
_JPEG_QUALITY = 88


def image_bytes_to_jpeg_data_uri(image_bytes: bytes) -> str:
    """Decode arbitrary image bytes, normalize to RGB JPEG, return a data URI."""
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


def extract_json_object(raw: str) -> str:
    """Strip markdown fences and isolate the outermost `{...}` JSON object."""
    text = raw.strip()
    text = re.sub(r"^\s*```[a-zA-Z0-9]*\s*", "", text)
    text = text.strip()
    if text.endswith("```"):
        text = text[: text.rfind("```")].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return text.strip()
