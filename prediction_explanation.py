def explain_prediction(probability, threshold=0.5):
    label = "FAKE" if probability >= threshold else "REAL"
    confidence = max(probability, 1.0 - probability)

    if confidence >= 0.85:
        strength = "high-confidence"
    elif confidence >= 0.65:
        strength = "moderate-confidence"
    else:
        strength = "low-confidence"

    if label == "FAKE":
        reason = (
            "the fused Mel and linear-frequency features matched patterns the model "
            "learned from synthetic audio. The linear spectrogram keeps higher-frequency "
            "detail that Mel compression can hide."
        )
    else:
        reason = (
            "the fused Mel and linear-frequency features stayed closer to patterns the "
            "model learned from real speech, with no strong synthetic-artifact score."
        )

    return f"{strength} {label}: {reason}"
