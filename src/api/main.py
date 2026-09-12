from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.ml.predict import load_model, predict_urgency

from .metrics import HTTP_REQUEST_DURATION, HTTP_REQUESTS, PREDICTIONS
from .schemas import HealthResponse, PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # carrega o modelo uma vez no startup, evita custo na primeira requisição
    yield


app = FastAPI(
    title="Triagem de Laudos Médicos",
    description="Classifica a urgência (normal/atenção/urgente) de um laudo médico.",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe_http_requests(request: Request, call_next):
    """Registra volume, status e latência sem criar rótulos de alta cardinalidade."""
    started_at = perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        route_path = getattr(route, "path", "unmatched")

        # O scrape do Prometheus não deve inflar os indicadores de uso da API.
        if route_path != "/metrics":
            labels = {
                "method": request.method,
                "route": route_path,
            }
            HTTP_REQUESTS.labels(
                **labels,
                status_code=str(status_code),
            ).inc()
            HTTP_REQUEST_DURATION.labels(**labels).observe(perf_counter() - started_at)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expõe as séries no formato de texto esperado pelo Prometheus."""
    return Response(
        content=generate_latest(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    classification, confidence = predict_urgency(request.text)
    PREDICTIONS.labels(classification=classification).inc()
    return PredictResponse(classification=classification, confidence=confidence)
