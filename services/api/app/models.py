from __future__ import annotations

from typing import Dict, Literal

from pydantic import BaseModel, Field


SeverityBucket = Literal["mild", "moderate", "severe"]


class AnalyzeResponse(BaseModel):
    severity_score_0_10: float = Field(ge=0.0, le=10.0)
    severity_bucket: SeverityBucket
    components: Dict[str, float]
    region_scores_0_1: Dict[str, float]
    heatmap_png_base64: str


class RecommendationResponse(BaseModel):
    decision: Literal["otc", "derm"]
    title: str
    bullets: list[str]
    cautions: list[str] = []

