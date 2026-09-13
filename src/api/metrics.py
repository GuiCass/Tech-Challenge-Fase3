"""Métricas Prometheus da API de triagem.

Os nomes usam um prefixo próprio para evitar colisões com métricas do runtime.
O rótulo de rota recebe o template cadastrado no FastAPI (por exemplo,
``/predict``), e não a URL bruta, evitando cardinalidade ilimitada.
"""

from prometheus_client import Counter, Histogram


HTTP_REQUESTS = Counter(
    "triagem_http_requests_total",
    "Total de requisições HTTP recebidas pela API.",
    labelnames=("method", "route", "status_code"),
)

HTTP_REQUEST_DURATION = Histogram(
    "triagem_http_request_duration_seconds",
    "Tempo de resposta das requisições HTTP em segundos.",
    labelnames=("method", "route"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PREDICTIONS = Counter(
    "triagem_predictions_total",
    "Total de classificações produzidas pelo modelo.",
    labelnames=("classification",),
)
