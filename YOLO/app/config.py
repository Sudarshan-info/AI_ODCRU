from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "models/license.pt"
    image_format: str = ".jpg"
    confidence_threshold: float = 0.4

    class Config:
        env_file = ".env"


settings = Settings()
