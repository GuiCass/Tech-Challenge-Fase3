"""Compara a latencia da inferencia scikit-learn e ONNX Runtime."""

import statistics
import sys
import time
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.predict_onnx import load_onnx_model  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "urgency_classifier.joblib"

SAMPLE_TEXT = "Patient presents with acute chest pain and shortness of breath."


def measure(function, repetitions: int = 200) -> float:
    function(SAMPLE_TEXT)
    values = []
    for _ in range(repetitions):
        started = time.perf_counter()
        function(SAMPLE_TEXT)
        values.append((time.perf_counter() - started) * 1000)
    return statistics.mean(values)


def main() -> None:
    sklearn_model = joblib.load(MODEL_PATH)
    onnx_session = load_onnx_model()
    onnx_input = onnx_session.get_inputs()[0].name

    def sklearn_predict(text: str):
        sklearn_model.predict([text])

    def onnx_predict(text: str):
        onnx_session.run(None, {onnx_input: [[text]]})

    sklearn_ms = measure(sklearn_predict)
    onnx_ms = measure(onnx_predict)
    improvement = (1 - onnx_ms / sklearn_ms) * 100
    print(f"scikit-learn: {sklearn_ms:.3f} ms (media)")
    print(f"ONNX Runtime: {onnx_ms:.3f} ms (media)")
    print(f"Melhoria: {improvement:.2f}%")


if __name__ == "__main__":
    main()