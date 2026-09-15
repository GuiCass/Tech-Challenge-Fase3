"""Inferencia do modelo exportado para ONNX Runtime."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

ONNX_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "urgency_classifier.onnx"


@lru_cache(maxsize=1)
def load_onnx_model() -> ort.InferenceSession:
    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{ONNX_MODEL_PATH} nao encontrado. Rode `python scripts/optimize_model.py` primeiro."
        )
    return ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])


def predict_urgency_onnx(text: str) -> tuple[str, float]:
    session = load_onnx_model()
    input_name = session.get_inputs()[0].name
    labels, probabilities = session.run(None, {input_name: np.array([[text]], dtype=object)})
    label = labels[0]
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    confidence = float(np.max(probabilities[0]))
    return str(label), confidence