from pydantic import BaseModel
from typing import Optional, Literal

class PredictResponse(BaseModel):
    identity: str
    confidence: float
    distance: Optional[float] = None # KNN distance only
    model_used: Literal["knn", "cnn", "ensemble"]
    bbox: Optional[list] = None

class RegisterRequest(BaseModel):
    label: str # me or person name

class HealthResponse(BaseModel):
    status: str
    knn_loaded: bool
    cnn_loaded: bool
    known_identities: list[str]
    threshold: float