from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import settings
from app.model import InvalidImageError, classify_image
from app.schemas import HealthResponse, PredictionResponse

app = FastAPI(title=settings.app_name, version="1.0.0")
executor = ProcessPoolExecutor(max_workers=settings.worker_count)


@app.on_event("shutdown")
def shutdown_executor() -> None:
    executor.shutdown(wait=False, cancel_futures=True)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        labels=list(settings.labels),
        model_ready=settings.model_path.exists(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are accepted.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Uploaded image is larger than 5 MB.")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(executor, classify_image, payload, settings.model_path)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    predictions = result["predictions"]
    return PredictionResponse(
        filename=file.filename or "upload",
        model=settings.model_id,
        optimized_runtime=result["runtime"],
        top_prediction=predictions[0],
        predictions=predictions,
    )
