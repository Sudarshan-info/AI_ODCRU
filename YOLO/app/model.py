from ultralytics import YOLO
from app.config import settings


def load_model():
    return YOLO(settings.model_path)
