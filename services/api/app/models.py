from __future__ import annotations

from typing import Dict, Literal

from pydantic import BaseModel, Field


SeverityBucket = Literal["mild", "moderate", "severe"]
EczemaBucket = Literal["none", "mild_eczema", "severe_eczema"]


class AnalyzeResponse(BaseModel):
    severity_score_0_10: float = Field(ge=0.0, le=10.0)
    severity_bucket: SeverityBucket
    eczema_bucket: EczemaBucket = "none"
    eczema_likelihood_0_10: float = Field(default=0.0, ge=0.0, le=10.0)
    scoring_debug: str = Field(
        default="",
        description="Whether OpenAI vision ran or local fallback was used (acne + eczema).",
    )
    components: Dict[str, float]
    region_scores_0_1: Dict[str, float]
    heatmap_png_base64: str


class RecommendationResponse(BaseModel):
    decision: Literal["otc", "derm"]
    title: str
    bullets: list[str]
    cautions: list[str] = []

