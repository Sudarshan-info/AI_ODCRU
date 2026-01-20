from pydantic import BaseModel


class DetectionMeta(BaseModel):
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
