from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.ml.predict import load_model, predict_urgency

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


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    classification, confidence = predict_urgency(request.text)
    return PredictResponse(classification=classification, confidence=confidence)
