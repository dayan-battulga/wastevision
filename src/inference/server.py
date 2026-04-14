"""FastAPI inference server for DyrtyVision."""

from __future__ import annotations

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI(
    title="DyrtyVision API",
    description="YOLOv8 waste classification inference endpoint",
    version="0.1.0",
)


class Detection(BaseModel):
    """Single detection result."""

    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]


class PredictionResponse(BaseModel):
    """Response from the /predict endpoint."""

    detections: list[Detection]
    image_width: int
    image_height: int


@app.on_event("startup")
async def load_model() -> None:
    """Load the YOLOv8 model into memory on server startup."""
    # TODO: Implement — load model from config, store in app.state
    pass


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Run waste detection on an uploaded image.

    Args:
        file: Uploaded image file (JPEG/PNG).

    Returns:
        Detection results with bounding boxes, classes, and confidence scores.
    """
    # TODO: Implement — read image, preprocess, run inference, postprocess
    raise NotImplementedError
