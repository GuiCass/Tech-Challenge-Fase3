"""Carrega o modelo treinado e expõe a função de predição usada pela API."""

from functools import lru_cache
import os

import joblib

from src.ml.train import MODEL_PATH


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} não encontrado. Rode `python -m src.ml.train` primeiro."
        )
    return joblib.load(MODEL_PATH)


def predict_urgency(text: str) -> tuple[str, float]:
    if os.getenv("MODEL_BACKEND", "sklearn").lower() == "onnx":
        from src.ml.predict_onnx import predict_urgency_onnx

        return predict_urgency_onnx(text)

    model = load_model()
    label = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    confidence = float(max(proba))
    return label, confidence
