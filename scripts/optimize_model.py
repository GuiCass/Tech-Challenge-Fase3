"""Exporta o pipeline scikit-learn treinado para ONNX Runtime."""

from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "urgency_classifier.joblib"
ONNX_MODEL_PATH = MODEL_PATH.with_suffix(".onnx")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} nao encontrado. Rode `python -m src.ml.train` primeiro."
        )

    pipeline = joblib.load(MODEL_PATH)
    initial_types = [("text", StringTensorType([None, 1]))]
    options = {id(pipeline.named_steps["clf"]): {"zipmap": False}}
    model_onnx = convert_sklearn(
        pipeline,
        initial_types=initial_types,
        options=options,
        target_opset=17,
    )
    ONNX_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    Path(ONNX_MODEL_PATH).write_bytes(model_onnx.SerializeToString())
    print(f"Modelo ONNX salvo em {ONNX_MODEL_PATH}")


if __name__ == "__main__":
    main()