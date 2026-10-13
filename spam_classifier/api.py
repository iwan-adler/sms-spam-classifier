"""FastAPI prediction service. Run: uvicorn spam_classifier.api:create_app --factory"""

from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from spam_classifier import models
from spam_classifier.train import MODEL_PATH


class PredictIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=1000, examples=["WINNER!! Claim your £900 prize now, call 09061701461"])


class Feature(BaseModel):
    feature: str
    contribution: float


class PredictOut(BaseModel):
    label: str
    is_spam: bool
    score: float
    score_type: str
    threshold: float
    top_spam_features: list[Feature] = Field(
        description="Features in this message that pushed the score towards spam (exact for linear models)."
    )


def create_app(model_path: Path = MODEL_PATH) -> FastAPI:
    app = FastAPI(
        title="SMS Spam Classifier",
        description="Classify SMS messages as spam or ham with a model trained on the UCI SMS Spam Collection.",
        version="0.1.0",
    )
    bundle = joblib.load(model_path) if model_path.is_file() else None

    def require_model() -> dict:
        if bundle is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "No trained model found. Run: python scripts/download_data.py && python -m spam_classifier.train",
            )
        return bundle

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_loaded": bundle is not None}

    @app.get("/model")
    def model_info() -> dict[str, Any]:
        """Which model is serving, its threshold, and its held-out test metrics."""
        return require_model()["metadata"]

    @app.post("/predict", response_model=PredictOut)
    def predict(payload: PredictIn) -> dict[str, Any]:
        loaded = require_model()
        model, meta = loaded["model"], loaded["metadata"]
        score = float(models.scores(model, [payload.text])[0])
        is_spam = score >= meta["threshold"]
        return {
            "label": "spam" if is_spam else "ham",
            "is_spam": is_spam,
            "score": round(score, 4),
            "score_type": meta["score_type"],
            "threshold": round(meta["threshold"], 4),
            "top_spam_features": [
                {"feature": name, "contribution": value}
                for name, value in models.top_spam_features(model, payload.text)
            ],
        }

    return app
