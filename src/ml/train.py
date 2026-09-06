"""Treina o classificador de urgência (TF-IDF + LogisticRegression) e salva o artefato."""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from src.ml.data import load_test_df, load_train_df

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODEL_DIR / "urgency_classifier.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000, C=5.0)),
        ]
    )


def train() -> Pipeline:
    train_df = load_train_df()
    test_df = load_test_df()

    pipeline = build_pipeline()
    pipeline.fit(train_df["text"], train_df["urgency"])

    predictions = pipeline.predict(test_df["text"])
    print("Relatório de classificação (conjunto de teste):")
    print(classification_report(test_df["urgency"], predictions))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Modelo salvo em {MODEL_PATH}")
    return pipeline


if __name__ == "__main__":
    train()
