"""
app/schemas.py
==============
Pydantic request / response models for the Waste Detection API.
These are import-time safe (no FastAPI dependency).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── sub-models ────────────────────────────────────────────────────────────────

class ImageSize(BaseModel):
    width: int = Field(..., description="Original image width in pixels")
    height: int = Field(..., description="Original image height in pixels")


class BoundingBox(BaseModel):
    x1: float = Field(..., description="Left edge (original image coords)")
    y1: float = Field(..., description="Top edge (original image coords)")
    x2: float = Field(..., description="Right edge (original image coords)")
    y2: float = Field(..., description="Bottom edge (original image coords)")


class Detection(BaseModel):
    bbox: BoundingBox
    material: str = Field(..., description="Predicted waste material class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    biodegradable: bool = Field(..., description="Derived from rules.json")
    recyclable: bool = Field(..., description="Derived from rules.json")
    category: str | None = Field(default=None, description="General material classification")
    bin: str | None = Field(default=None, description="Recommended disposal bin")
    instructions: str | None = Field(default=None, description="Handling or recycling guidelines")
    polygon: list[list[float]] | None = Field(default=None, description="Instance segmentation mask polygon points [[x, y], ...]")


# ── request / response schemas ────────────────────────────────────────────────

class PredictBase64Request(BaseModel):
    image: str = Field(..., description="Base64-encoded image (with or without data URL prefix)")
    conf_thres: float | None = Field(default=None, ge=0.05, le=1.0, description="Optional override for confidence threshold")
    iou_thres: float | None = Field(default=None, ge=0.05, le=1.0, description="Optional override for NMS IoU threshold")


class PredictResponse(BaseModel):
    image_size: ImageSize
    count: int = Field(..., description="Number of detections returned")
    detections: list[Detection]
    inference_ms: float = Field(..., description="Wall-clock inference time in ms")


# ── health response ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field("ok", description="Always 'ok' when the server is running")
    model_loaded: bool = Field(..., description="True when the ONNX session is ready")
    model_file: str = Field(..., description="Basename of the model file in use")
    classes: list[str] = Field(default_factory=list, description="List of recognized material classes")
