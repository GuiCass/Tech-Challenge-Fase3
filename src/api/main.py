from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.ml.predict import load_model, predict_urgency

from .schemas import HealthResponse, PredictRequest, PredictResponse

REQUEST_COUNT = Counter(
    "triagem_http_requests_total",
    "Total de requisicoes HTTP recebidas pela API",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "triagem_http_request_duration_seconds",
    "Duracao das requisicoes HTTP em segundos",
    ["method", "path"],
)


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
async def observe_requests(request: Request, call_next):
    started = perf_counter()
    status = "500"
    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        path = request.url.path
        REQUEST_COUNT.labels(request.method, path, status).inc()
        REQUEST_LATENCY.labels(request.method, path).observe(perf_counter() - started)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    classification, confidence = predict_urgency(request.text)
    return PredictResponse(classification=classification, confidence=confidence)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
