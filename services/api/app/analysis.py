from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

import numpy as np
from PIL import Image, ImageFilter

from .ml_model import ModelUnavailable, predict_prob_mask
from .openai_classifier import OpenAIClassifierUnavailable, classify_severity
from .openai_eczema_classifier import OpenAIEczemaClassifierUnavailable, classify_eczema
from .severity_head import SeverityHeadUnavailable, predict_severity


@dataclass(frozen=True)
class RegionScores:
    forehead: float
    left_cheek: float
    right_cheek: float
    jawline: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "forehead": float(self.forehead),
            "left_cheek": float(self.left_cheek),
            "right_cheek": float(self.right_cheek),
            "jawline": float(self.jawline),
        }


def _to_rgb_array(img: Image.Image, max_side: int = 640) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def _simple_face_regions(arr: np.ndarray) -> Dict[str, Tuple[slice, slice]]:
    """
    Heuristic regions assuming a centered selfie.
    Returns regions as [y, x] slices.
    """
    h, w, _ = arr.shape
    y0, y1 = int(0.12 * h), int(0.95 * h)
    x0, x1 = int(0.10 * w), int(0.90 * w)

    face_h = y1 - y0
    face_w = x1 - x0

    forehead = (slice(y0, y0 + int(0.22 * face_h)), slice(x0 + int(0.20 * face_w), x0 + int(0.80 * face_w)))
    left_cheek = (
        slice(y0 + int(0.32 * face_h), y0 + int(0.70 * face_h)),
        slice(x0 + int(0.10 * face_w), x0 + int(0.45 * face_w)),
    )
    right_cheek = (
        slice(y0 + int(0.32 * face_h), y0 + int(0.70 * face_h)),
        slice(x0 + int(0.55 * face_w), x0 + int(0.90 * face_w)),
    )
    jawline = (slice(y0 + int(0.70 * face_h), y0 + int(0.95 * face_h)), slice(x0 + int(0.18 * face_w), x0 + int(0.82 * face_w)))

    return {
        "forehead": forehead,
        "left_cheek": left_cheek,
        "right_cheek": right_cheek,
        "jawline": jawline,
        "full_face": (slice(y0, y1), slice(x0, x1)),
    }


def _redness_map(arr: np.ndarray) -> np.ndarray:
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    # A simple erythema-ish proxy: red relative to green/blue.
    redness = np.clip(r - 0.5 * g - 0.2 * b, 0.0, 1.0)
    return redness


def _texture_map(img: Image.Image) -> np.ndarray:
    # Edge magnitude as a cheap “lesion/texture” proxy.
    gray = img.convert("L").filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(gray).astype(np.float32) / 255.0
    return arr


def _normalize01(x: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(x, 5))
    hi = float(np.percentile(x, 95))
    if hi - lo < 1e-6:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _heatmap_png_base64(heat: np.ndarray) -> str:
    """
    heat: 2D array 0..1
    Produces a red overlay heatmap PNG (RGBA) encoded as base64.
    """
    heat_u8 = (np.clip(heat, 0.0, 1.0) * 255.0).astype(np.uint8)
    h, w = heat_u8.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 255  # red
    rgba[:, :, 1] = 0
    rgba[:, :, 2] = 0
    rgba[:, :, 3] = heat_u8  # alpha
    out = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _count_components(binary: np.ndarray) -> int:
    """
    Count connected components in a small binary mask (HxW, values {0,1}).
    4-neighborhood. Designed for small masks (e.g., 160x160).
    """
    h, w = binary.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    count = 0
    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0 or visited[y, x] == 1:
                continue
            count += 1
            stack = [(y, x)]
            visited[y, x] = 1
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and visited[ny, nx] == 0 and binary[ny, nx] == 1:
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
    return count


def analyze_image_bytes(image_bytes: bytes, *, filename: str | None = None) -> dict:
    img = Image.open(io.BytesIO(image_bytes))
    arr = _to_rgb_array(img)
    regions = _simple_face_regions(arr)

    # Try ML model first (lesion probability mask). Fall back to heuristic proxy.
    inflammation: np.ndarray
    used_model = False
    lesion_area_pct = 0.0
    lesion_count = 0
    avg_lesion_prob = 0.0

    try:
        pred = predict_prob_mask(image_bytes, size=256)
        prob = pred.prob_mask  # 256x256
        # Resize model output to match `arr` spatial resolution so region slicing
        # and intensity scoring are consistent.
        ah, aw, _ = arr.shape
        prob_img = Image.fromarray((np.clip(prob, 0.0, 1.0) * 255).astype(np.uint8), mode="L").resize(
            (aw, ah), resample=Image.BILINEAR
        )
        inflammation = (np.asarray(prob_img).astype(np.float32) / 255.0)
        used_model = True

        lesion_binary = (inflammation > 0.5).astype(np.uint8)
        lesion_area_pct = float(np.mean(lesion_binary))
        avg_lesion_prob = float(np.mean(inflammation))

        # Count blobs on downsampled mask for stability
        small = Image.fromarray((lesion_binary * 255).astype(np.uint8), mode="L").resize((160, 160), resample=Image.NEAREST)
        small_bin = (np.asarray(small) > 0).astype(np.uint8)
        lesion_count = int(_count_components(small_bin))
    except ModelUnavailable:
        redness = _normalize01(_redness_map(arr))
        texture = _normalize01(_texture_map(img.resize((arr.shape[1], arr.shape[0]))))
        inflammation = np.clip(0.75 * redness + 0.25 * texture, 0.0, 1.0)

    full = regions["full_face"]
    face_infl = inflammation[full[0], full[1]]
    intensity = float(np.mean(face_infl))

    if used_model:
        # Segmentation-derived scoring
        lesion_area_score = min(1.0, lesion_area_pct / 0.18)
        lesion_count_score = min(1.0, lesion_count / 45.0)
        intensity_score = min(1.0, intensity / 0.45)
        severity_0_10 = 10.0 * (0.45 * lesion_area_score + 0.35 * lesion_count_score + 0.20 * intensity_score)
    else:
        red_area_pct = float(np.mean(face_infl > 0.55))
        texture = _normalize01(_texture_map(img.resize((arr.shape[1], arr.shape[0]))))
        lesion_proxy = float(np.mean(texture[full[0], full[1]] > 0.65))

        lesion_score = min(1.0, lesion_proxy / 0.12)
        red_area_score = min(1.0, red_area_pct / 0.25)
        intensity_score = min(1.0, intensity / 0.55)
        severity_0_10 = 10.0 * (0.4 * lesion_score + 0.3 * red_area_score + 0.3 * intensity_score)

    severity_0_10 = float(np.clip(severity_0_10, 0.0, 10.0))

    if severity_0_10 < 3.5:
        bucket = "mild"
    elif severity_0_10 < 6.5:
        bucket = "moderate"
    else:
        bucket = "severe"

    # If this is a dataset image with a known `levleX_` label, expose it as debug components
    # so we can spot model-vs-label mismatches quickly (does not affect scoring logic).
    filename_level = float("nan")
    if filename:
        lower = filename.lower()
        j = lower.find("levle")
        if j != -1 and j + 5 < len(lower):
            ch = lower[j + 5]
            if ch.isdigit():
                lvl = int(ch)
                if 0 <= lvl <= 3:
                    filename_level = float(lvl)

    severity_probs: dict[str, float] | None = None
    severity_head_pred_0_2 = float("nan")
    try:
        # Feature vector aligned with ml/train_severity_head.py
        rgb01 = _to_rgb_array(img, max_side=512)
        red = _redness_map(rgb01)
        # bring prob/inflammation to same shape as rgb01
        ih, iw, _ = rgb01.shape
        infl_img = Image.fromarray((np.clip(inflammation, 0, 1) * 255).astype(np.uint8), mode="L").resize(
            (iw, ih), resample=Image.BILINEAR
        )
        infl = np.asarray(infl_img).astype(np.float32) / 255.0

        face = _simple_face_regions(rgb01)["full_face"]
        red_face = red[face[0], face[1]]
        infl_face = infl[face[0], face[1]]

        p = infl_face.reshape(-1)
        k = max(1, int(0.02 * p.size))
        topk = float(np.mean(np.partition(p, -k)[-k:]))
        mass = float(np.mean(infl_face))
        area_t20 = float(np.mean(infl_face > 0.20))
        area_t35 = float(np.mean(infl_face > 0.35))
        intensity_face = float(np.mean(infl_face))

        lesion_bin = (infl_face > 0.5).astype(np.uint8)
        lesion_area_bin = float(np.mean(lesion_bin))
        small = Image.fromarray((lesion_bin * 255).astype(np.uint8), mode="L").resize((160, 160), resample=Image.NEAREST)
        lesion_count_face = float(_count_components((np.asarray(small) > 0).astype(np.uint8)))

        r = red_face.reshape(-1)
        red_mean = float(np.mean(red_face))
        red_top = float(np.mean(np.partition(r, -k)[-k:]))

        # Texture features (edge magnitude) on the same face crop
        face_img = Image.fromarray((rgb01 * 255).astype(np.uint8), mode="RGB").crop((face[1].start, face[0].start, face[1].stop, face[0].stop))
        tex = _normalize01(_texture_map(face_img))
        tex_p = tex.reshape(-1)
        tex_top = float(np.mean(np.partition(tex_p, -k)[-k:]))
        tex_mean = float(np.mean(tex))

        feats = np.asarray(
            [
                mass,
                topk,
                area_t20,
                area_t35,
                red_mean,
                red_top,
                lesion_area_bin,
                lesion_count_face,
                intensity_face,
                tex_mean,
                tex_top,
                red_top * topk,
            ],
            dtype=np.float32,
        )

        sp = predict_severity(feats)
        severity_probs = sp.probs
        severity_head_pred_0_2 = float(0 if sp.bucket == "mild" else (1 if sp.bucket == "moderate" else 2))
    except SeverityHeadUnavailable:
        pass

    # Confidence-based bucket override using severity head (keeps numeric score stable).
    if severity_probs:
        p_mild = float(severity_probs.get("mild", 0.0))
        p_mod = float(severity_probs.get("moderate", 0.0))
        p_sev = float(severity_probs.get("severe", 0.0))
        # Override only when one class is clearly dominant.
        if p_sev >= 0.60:
            bucket = "severe"
        elif p_mod >= 0.60:
            bucket = "moderate"
        elif p_mild >= 0.60:
            bucket = "mild"

    # ── OpenAI Vision override ──────────────────────────────────────────────
    used_openai = False
    oai_result = None
    acne_openai_skip_reason: str | None = None
    try:
        oai_result = classify_severity(image_bytes)
        severity_0_10 = oai_result.severity_score_0_10
        bucket = oai_result.severity_bucket
        used_openai = True
    except OpenAIClassifierUnavailable as exc:
        acne_openai_skip_reason = str(exc).replace("\n", " ").strip()[:240] or "OpenAIClassifierUnavailable"

    # ── OpenAI eczema / atopic pattern assessment ───────────────────────────
    eczema_bucket = "none"
    eczema_likelihood_0_10 = 0.0
    used_openai_eczema = False
    oai_eczema = None
    eczema_openai_skip_reason: str | None = None
    try:
        oai_eczema = classify_eczema(image_bytes)
        eczema_bucket = oai_eczema.eczema_bucket
        eczema_likelihood_0_10 = oai_eczema.eczema_likelihood_0_10
        used_openai_eczema = True
    except OpenAIEczemaClassifierUnavailable as exc:
        eczema_openai_skip_reason = str(exc).replace("\n", " ").strip()[:240] or "OpenAIEczemaClassifierUnavailable"

    acne_src = "openai_gpt-4o" if used_openai else f"local_fallback ({acne_openai_skip_reason or 'unknown'})"
    eczema_src = "openai_gpt-4o" if used_openai_eczema else f"local_fallback ({eczema_openai_skip_reason or 'unknown'})"
    scoring_debug = f"acne_severity={acne_src} ; eczema={eczema_src}"
    logger.info("analyze %s", scoring_debug)

    def region_mean(name: str) -> float:
        ys, xs = regions[name]
        return float(np.mean(inflammation[ys, xs]))

    region_scores = RegionScores(
        forehead=region_mean("forehead"),
        left_cheek=region_mean("left_cheek"),
        right_cheek=region_mean("right_cheek"),
        jawline=region_mean("jawline"),
    )

    # Downsample heatmap for transport.
    heat_small = Image.fromarray((inflammation * 255.0).astype(np.uint8), mode="L").resize((160, 160))
    heat = np.asarray(heat_small).astype(np.float32) / 255.0
    heat_png_b64 = _heatmap_png_base64(heat)

    return {
        "severity_score_0_10": severity_0_10,
        "severity_bucket": bucket,
        "eczema_bucket": eczema_bucket,
        "eczema_likelihood_0_10": eczema_likelihood_0_10,
        "scoring_debug": scoring_debug,
        "components": {
            "used_model": float(1.0 if used_model else 0.0),
            "used_openai": float(1.0 if used_openai else 0.0),
            "used_openai_eczema": float(1.0 if used_openai_eczema else 0.0),
            "lesion_area_pct_0_1": float(lesion_area_pct),
            "lesion_count": float(lesion_count),
            "avg_lesion_prob_0_1": float(avg_lesion_prob),
            "inflammation_intensity_0_1": float(min(1.0, intensity / 0.55)),
            "severity_probs_mild": float(severity_probs["mild"]) if severity_probs else float("nan"),
            "severity_probs_moderate": float(severity_probs["moderate"]) if severity_probs else float("nan"),
            "severity_probs_severe": float(severity_probs["severe"]) if severity_probs else float("nan"),
            "filename_level_0_3": float(filename_level),
            "severity_head_pred_0_2": float(severity_head_pred_0_2),
            # 1.0 when heatmap/region scores and severity score share the same local pipeline;
            # 0.0 when severity came from OpenAI but heatmap is still segmentation-based.
            "severity_heatmap_same_source_as_score": float(0.0 if used_openai else 1.0),
            # 1.0 if the model's severity_bucket matched the score band; NaN if OpenAI was not used.
            "openai_model_bucket_matched_score": (
                float(1.0 if oai_result.model_bucket_matched_score else 0.0)
                if oai_result is not None
                else float("nan")
            ),
            "openai_eczema_model_bucket_matched_score": (
                float(1.0 if oai_eczema.model_bucket_matched_score else 0.0)
                if oai_eczema is not None
                else float("nan")
            ),
        },
        "region_scores_0_1": region_scores.as_dict(),
        "heatmap_png_base64": heat_png_b64,
    }

