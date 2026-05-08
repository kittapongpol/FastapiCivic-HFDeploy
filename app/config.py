from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "Honda Civic Generation Classifier"
    model_id: str = "google/vit-base-patch16-224"
    model_task: str = "image-classification"
    model_path: Path = Path("models/civic_gen_quantized.onnx")
    labels: tuple[str, ...] = ("EG", "EK", "ES", "FD", "FB", "FC", "FE", "FK")
    max_upload_bytes: int = 5 * 1024 * 1024
    worker_count: int = 2


settings = Settings()
