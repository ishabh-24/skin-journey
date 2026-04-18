from __future__ import annotations

from typing import Literal

from .models import EczemaBucket, RecommendationResponse, SeverityBucket


def recommend(
    *,
    severity_bucket: SeverityBucket,
    worsening_streak_days: int = 0,
    no_improvement_days: int = 0,
    cystic_suspected: bool = False,
) -> RecommendationResponse:
    derm_trigger = (
        cystic_suspected
        or worsening_streak_days >= 3
        or (severity_bucket == "severe")
        or (no_improvement_days >= 14 and severity_bucket in ("moderate", "severe"))
    )

    if derm_trigger:
        return RecommendationResponse(
            decision="derm",
            title="Consider dermatology care",
            bullets=[
                "Your trend/severity suggests you may benefit from prescription options.",
                "If you have pain, scarring, or deep bumps, try to book a visit soon.",
                "If symptoms worsen rapidly, consider urgent care.",
            ],
            cautions=[
                "This is not a medical diagnosis.",
            ],
        )

    if severity_bucket == "mild":
        return RecommendationResponse(
            decision="otc",
            title="OTC routine (mild acne)",
            bullets=[
                "Gentle cleanser (fragrance-free), 1–2x/day",
                "Salicylic acid (0.5–2%) a few nights/week",
                "Spot treat with benzoyl peroxide (2.5%) if tolerated",
                "Moisturizer + sunscreen daily",
            ],
            cautions=[
                "Introduce one active at a time to reduce irritation.",
            ],
        )

    return RecommendationResponse(
        decision="otc",
        title="OTC routine (moderate acne)",
        bullets=[
            "Gentle cleanser + moisturizer + sunscreen daily",
            "Benzoyl peroxide wash (2.5–5%) in the morning if tolerated",
            "Adapalene (OTC retinoid) at night, start 2–3x/week then increase",
            "Avoid harsh scrubs; keep routine consistent for 8–12 weeks",
        ],
        cautions=[
            "Retinoids can cause dryness; use moisturizer and go slowly.",
            "Avoid retinoids if pregnant; consult a clinician.",
        ],
    )


def recommend_eczema(*, eczema_bucket: EczemaBucket) -> RecommendationResponse:
    """OTC vs dermatology style guidance for eczema-type patterns (not a diagnosis)."""
    if eczema_bucket == "severe_eczema":
        return RecommendationResponse(
            decision="derm",
            title="Eczema pattern looks significantly inflamed",
            bullets=[
                "Strong facial redness, weeping, crusting, or widespread rash merits "
                "timely medical evaluation — prescription anti-inflammatories are often needed.",
                "Until seen: gentle fragrance-free cleanser, thick bland moisturizer after "
                "lukewarm rinses, avoid scratching and new products.",
                "Seek urgent care if rapidly spreading pain, fever, or eye involvement.",
            ],
            cautions=[
                "This is not a medical diagnosis; photos can miss important context.",
            ],
        )

    if eczema_bucket == "mild_eczema":
        return RecommendationResponse(
            decision="otc",
            title="Possible mild eczema-type dryness (OTC care)",
            bullets=[
                "Use a thick fragrance-free cream or ointment moisturizer at least twice daily "
                "and after washing.",
                "Short lukewarm showers; avoid harsh soaps, scrubs, and fragranced products.",
                "Identify and reduce triggers (wind, sweat, certain cosmetics).",
                "If itching persists beyond ~2 weeks or worsens, book a clinician visit.",
            ],
            cautions=[
                "If OTC care stings or flares the skin, stop and get professional advice.",
            ],
        )

    return RecommendationResponse(
        decision="otc",
        title="No clear eczema pattern detected",
        bullets=[
            "Skin appears closer to normal or non-eczema dryness from this photo alone.",
            "Maintain gentle cleanser + daily moisturizer + sunscreen.",
            "Re-check with new photos if you develop itchy, scaly patches that persist.",
        ],
        cautions=[
            "Generic dryness and early eczema can look similar; when in doubt, ask a clinician.",
        ],
    )

