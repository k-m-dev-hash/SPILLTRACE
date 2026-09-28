import numpy as np


def validate_candidate(candidate, image_statistics, wind_speed=None):
    score = 0.0
    reasons = []

    darkness_score = candidate.get("darkness_score", 0.0)
    compactness = candidate.get("compactness", 0.0)
    aspect_ratio = candidate.get("aspect_ratio", 1.0)
    area_ratio = candidate.get("area_ratio", 0.0)

    if darkness_score >= 1.5:
        score += 30
        reasons.append("Strong SAR backscatter contrast")
    elif darkness_score >= 0.8:
        score += 20
        reasons.append("Moderate SAR backscatter contrast")
    else:
        score += 5
        reasons.append("Weak SAR backscatter contrast")

    if 1.3 <= aspect_ratio <= 8:
        score += 20
        reasons.append("Elongated slick-like geometry")
    else:
        score += 5
        reasons.append("Less characteristic geometry")

    if 0.05 <= compactness <= 0.65:
        score += 15
        reasons.append("Irregular boundary geometry")
    else:
        score += 5
        reasons.append("Compact geometry")

    if 0.005 <= area_ratio <= 0.40:
        score += 15
        reasons.append("Reasonable candidate size")
    else:
        score += 5
        reasons.append("Candidate size requires review")

    if wind_speed is not None:
        if 2 <= wind_speed <= 10:
            score += 15
            reasons.append("Wind conditions compatible with surface slick detection")
        elif wind_speed < 2:
            score -= 10
            reasons.append("Very low wind may produce SAR dark look-alikes")
        else:
            score += 5
            reasons.append("Higher wind conditions require additional validation")
    else:
        score += 5
        reasons.append("Wind data not yet available")

    score = float(np.clip(score, 0, 100))

    if score >= 70:
        classification = "PROBABLE OIL"
    elif score >= 40:
        classification = "REQUIRES REVIEW"
    else:
        classification = "LIKELY LOOK-ALIKE"

    return {
        "oil_confidence": round(score / 100, 3),
        "validation_score": round(score, 2),
        "classification": classification,
        "reasons": reasons,
        "image_statistics": image_statistics
    }