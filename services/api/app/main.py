from __future__ import annotations

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .analysis import analyze_image_bytes
from .models import AnalyzeResponse, EczemaBucket, RecommendationResponse, SeverityBucket
from .recommendations import recommend, recommend_eczema


app = FastAPI(title="Skin Journey AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(image: UploadFile = File(...)) -> AnalyzeResponse:
    data = await image.read()
    result = analyze_image_bytes(data, filename=image.filename)
    return AnalyzeResponse(**result)


@app.get("/recommend", response_model=RecommendationResponse)
def get_recommendation(
    severity_bucket: SeverityBucket,
    worsening_streak_days: int = 0,
    no_improvement_days: int = 0,
    cystic_suspected: bool = False,
) -> RecommendationResponse:
    return recommend(
        severity_bucket=severity_bucket,
        worsening_streak_days=worsening_streak_days,
        no_improvement_days=no_improvement_days,
        cystic_suspected=cystic_suspected,
    )


@app.get("/recommend-eczema", response_model=RecommendationResponse)
def get_eczema_recommendation(eczema_bucket: EczemaBucket) -> RecommendationResponse:
    return recommend_eczema(eczema_bucket=eczema_bucket)

