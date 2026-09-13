import joblib
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.ml import predict as predict_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Treina um modelo minúsculo em memória para não depender do dataset completo no CI."""
    texts = ["chest pain emergency", "routine checkup fine", "mild headache attention needed"]
    labels = ["urgente", "normal", "atenção"]
    tiny_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression()),
        ]
    )
    tiny_pipeline.fit(texts, labels)

    model_path = tmp_path / "tiny_model.joblib"
    joblib.dump(tiny_pipeline, model_path)

    monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
    predict_module.load_model.cache_clear()

    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client

    predict_module.load_model.cache_clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_classification_and_confidence(client):
    response = client.post("/predict", json={"text": "chest pain emergency"})
    assert response.status_code == 200
    body = response.json()
    assert body["classification"] in {"normal", "atenção", "urgente"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_metrics_exposes_request_volume_latency_and_predictions(client):
    prediction = client.post("/predict", json={"text": "chest pain emergency"})
    assert prediction.status_code == 200

    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "text/plain; version=0.0.4"
    )

    metric_names = {
        family.name for family in text_string_to_metric_families(response.text)
    }
    assert "triagem_http_requests" in metric_names
    assert "triagem_http_request_duration_seconds" in metric_names
    assert "triagem_predictions" in metric_names

    assert 'route="/predict"' in response.text
    assert 'status_code="200"' in response.text
    assert "triagem_http_request_duration_seconds_bucket" in response.text
    assert "triagem_predictions_total" in response.text


def test_metrics_does_not_count_its_own_scrapes(client):
    first = client.get("/metrics")
    second = client.get("/metrics")

    assert first.status_code == 200
    assert second.status_code == 200
    assert 'route="/metrics"' not in second.text
