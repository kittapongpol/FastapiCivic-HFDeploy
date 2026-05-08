from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str = Field(..., examples=["FE"])
    confidence: float = Field(..., ge=0, le=1, examples=[0.87])


class PredictionResponse(BaseModel):
    filename: str
    model: str
    optimized_runtime: str
    top_prediction: Prediction
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    labels: list[str]
    model_ready: bool
