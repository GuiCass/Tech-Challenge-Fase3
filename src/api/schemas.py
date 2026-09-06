from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texto do laudo médico a classificar")


class PredictResponse(BaseModel):
    classification: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
