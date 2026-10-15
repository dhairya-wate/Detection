"""
app/main.py
===========
FastAPI application: routes, middleware, and lifespan management.

Start-up order (inside lifespan):
  1. Load RulesEngine (validates rules.json)
  2. Load WasteDetector (opens ONNX session)
  3. Yield → server is ready
  4. Shutdown: detector reference is released

Routes
------
  GET  /              → Live Camera Web Interface (HTML)
  GET  /camera        → Live Camera Web Interface (HTML)
  GET  /health        → HealthResponse
  GET  /rules         → Waste categorization & recycling rules
  POST /predict       → PredictResponse (multipart file upload)
  POST /predict_frame → PredictResponse (base64 camera frame)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import (
    FastAPI,
    File,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

# Ensure project root is in sys.path so modules are resolvable in all execution modes
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from app.config import settings
    from app.inference import WasteDetector
    from app.rules import MODEL_CLASSES, RulesEngine
    from app.schemas import (
        BoundingBox,
        Detection,
        HealthResponse,
        ImageSize,
        PredictBase64Request,
        PredictResponse,
    )
except (ImportError, ModuleNotFoundError):
    from config import settings  # type: ignore[no-redef]
    from inference import WasteDetector  # type: ignore[no-redef]
    from rules import MODEL_CLASSES, RulesEngine  # type: ignore[no-redef]
    from schemas import (  # type: ignore[no-redef]
        BoundingBox,
        Detection,
        HealthResponse,
        ImageSize,
        PredictBase64Request,
        PredictResponse,
    )


# ── JSON logging ──────────────────────────────────────────────────────────────

def _configure_logging() -> None:
    """Set up structured JSON logging via the standard logging module."""

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload: dict[str, Any] = {
                "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                payload["exc"] = self.formatException(record.exc_info)
            return json.dumps(payload)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


_configure_logging()
logger = logging.getLogger(__name__)

# ── application state (lifespan-managed) ─────────────────────────────────────

_detector: WasteDetector | None = None
_rules: RulesEngine | None = None


def _resolve_path(raw_path: str | Path) -> Path:
    """Resolve path against CWD or project root."""
    p = Path(raw_path)
    if p.is_absolute() or p.exists():
        return p
    candidate = _PROJECT_ROOT / p
    if candidate.exists():
        return candidate
    return p


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load resources on startup; release on shutdown."""
    global _detector, _rules

    rules_path = _resolve_path(settings.RULES_PATH)
    logger.info("Starting up – loading rules from %s", rules_path)
    _rules = RulesEngine(rules_path, MODEL_CLASSES)

    model_path = _resolve_path(settings.MODEL_PATH)
    logger.info("Loading ONNX model from %s", model_path)
    try:
        _detector = WasteDetector(
            model_path=model_path,
            class_names=MODEL_CLASSES,
            conf_thres=settings.CONF_THRES,
            iou_thres=settings.IOU_THRES,
            intra_op_threads=settings.ONNX_THREADS,
        )
        logger.info("Startup complete. Model: %s", _detector.model_file)
    except FileNotFoundError:
        logger.warning(
            "Model file not found at %s. Server starting in model-pending mode "
            "(GET /health will report model_loaded=False; place best.onnx in models/ to enable inference).",
            model_path,
        )
        _detector = None
    except Exception as exc:
        logger.warning("Failed to initialize ONNX model: %s", exc)
        _detector = None

    yield

    logger.info("Shutting down.")
    _detector = None
    _rules = None


# ── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(
    title="Waste Detection API",
    description="Detects and classifies waste materials using a YOLOv8 TACO ONNX model.",
    version="1.1.0",
    lifespan=lifespan,
)

# Static files for Camera Web Interface
_static_dir = _PROJECT_ROOT / "app" / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# CORS
if settings.cors_origins_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Allow all origins locally for camera web clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ── Request-ID + latency logging middleware ───────────────────────────────────

@app.middleware("http")
async def request_logging_middleware(
    request: Request, call_next: Callable[[Request], Any]
) -> Response:
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - t0) * 1000
    # Avoid spamming console with high-frequency frame predictions
    if not request.url.path.startswith("/predict_frame"):
        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": round(latency_ms, 2),
                }
            )
        )
    response.headers["X-Request-ID"] = request_id
    return response


# ── helpers ───────────────────────────────────────────────────────────────────

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _check_api_key(x_api_key: str | None) -> None:
    """Raise 401 if API_KEY is configured and the header doesn't match."""
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


def _detector_required() -> WasteDetector:
    global _detector
    if _detector is None:
        model_path = _resolve_path(settings.MODEL_PATH)
        if model_path.exists():
            try:
                _detector = WasteDetector(
                    model_path=model_path,
                    class_names=MODEL_CLASSES,
                    conf_thres=settings.CONF_THRES,
                    iou_thres=settings.IOU_THRES,
                    intra_op_threads=settings.ONNX_THREADS,
                )
                logger.info("Model loaded on-demand: %s", _detector.model_file)
                return _detector
            except Exception as exc:
                logger.error("Failed to load model on-demand: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please place best.onnx in the models directory.",
        )
    return _detector


def _rules_required() -> RulesEngine:
    global _rules
    if _rules is None:
        rules_path = _resolve_path(settings.RULES_PATH)
        if rules_path.exists():
            try:
                _rules = RulesEngine(rules_path, MODEL_CLASSES)
                logger.info("Rules loaded on-demand: %s", rules_path)
                return _rules
            except Exception as exc:
                logger.error("Failed to load rules on-demand: %s", exc)
        raise HTTPException(status_code=503, detail="Rules not loaded.")
    return _rules


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["ui"], include_in_schema=False)
@app.get("/camera", tags=["ui"])
async def camera_page() -> FileResponse:
    """Serve the real-time camera waste detection web interface."""
    index_file = _PROJECT_ROOT / "app" / "static" / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web UI not found.")
    return FileResponse(index_file)


@app.get("/rules", tags=["inference"])
async def rules() -> dict[str, Any]:
    """Return all waste classification rules, recyclability info, and disposal instructions."""
    rules_engine = _rules_required()
    return {
        "classes": rules_engine.classes,
        "rules": rules_engine.all_rules,
    }


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Return service health status."""
    global _detector
    if _detector is None:
        model_path = _resolve_path(settings.MODEL_PATH)
        if model_path.exists():
            try:
                _detector = WasteDetector(
                    model_path=model_path,
                    class_names=MODEL_CLASSES,
                    conf_thres=settings.CONF_THRES,
                    iou_thres=settings.IOU_THRES,
                    intra_op_threads=settings.ONNX_THREADS,
                )
            except Exception:
                pass
    loaded = _detector is not None
    model_file = _detector.model_file if loaded else Path(settings.MODEL_PATH).name
    classes = _detector.class_names if loaded else MODEL_CLASSES
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_file=model_file,
        classes=classes,
    )


@app.post("/predict_frame", response_model=PredictResponse, tags=["inference"])
async def predict_frame(
    payload: PredictBase64Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> PredictResponse:
    """
    Detect and classify waste items from a base64-encoded camera frame.
    Optimized for high-FPS live webcam streaming.
    """
    _check_api_key(x_api_key)
    detector = _detector_required()
    rules = _rules_required()

    raw_b64 = payload.image
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        data = base64.b64decode(raw_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 payload.") from exc

    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Frame exceeds MAX_UPLOAD_MB={settings.MAX_UPLOAD_MB} MB limit.",
        )

    try:
        image = Image.open(io.BytesIO(data))
        image = image.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Image frame could not be decoded.") from exc

    orig_w, orig_h = image.size

    t0 = time.perf_counter()
    detections_raw, _ = await run_in_threadpool(
        detector.detect,
        image,
        payload.conf_thres,
        payload.iou_thres,
        payload.max_area_ratio,
        None,
        None,
        payload.roi,
    )
    inference_ms = (time.perf_counter() - t0) * 1000.0

    detections: list[Detection] = []
    for det in detections_raw:
        rule = rules.get(det.material)
        detections.append(
            Detection(
                bbox=BoundingBox(**det.bbox),
                material=det.material,
                confidence=round(det.confidence, 4),
                biodegradable=rule["biodegradable"],
                recyclable=rule["recyclable"],
                category=rule.get("category"),
                bin=rule.get("bin"),
                instructions=rule.get("instructions"),
                polygon=det.polygon,
            )
        )

    return PredictResponse(
        image_size=ImageSize(width=orig_w, height=orig_h),
        count=len(detections),
        detections=detections,
        inference_ms=round(inference_ms, 2),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    file: UploadFile = File(..., description="JPEG, PNG, or WebP image"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> PredictResponse:
    """
    Detect and classify waste items in the uploaded image.

    - **file**: image file (JPEG / PNG / WebP), max size controlled by MAX_UPLOAD_MB
    """
    _check_api_key(x_api_key)
    detector = _detector_required()
    rules = _rules_required()

    # ── content-type validation ───────────────────────────────────────────
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{content_type}'. "
                   f"Accepted: {sorted(_ALLOWED_CONTENT_TYPES)}",
        )

    # ── size check ────────────────────────────────────────────────────────
    max_bytes = settings.max_upload_bytes
    chunk_size = 64 * 1024  # 64 KB chunks
    data = bytearray()
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds MAX_UPLOAD_MB={settings.MAX_UPLOAD_MB} MB limit.",
            )

    # ── decode image ──────────────────────────────────────────────────────
    try:
        image = Image.open(io.BytesIO(bytes(data)))
        image.verify()                          # detect corrupt files early
        image = Image.open(io.BytesIO(bytes(data)))  # reopen after verify()
        image = image.convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        logger.warning("Failed to decode uploaded image: %s", exc)
        raise HTTPException(status_code=400, detail="Image could not be decoded.") from exc

    orig_w, orig_h = image.size

    # ── inference (off the event loop) ───────────────────────────────────
    t0 = time.perf_counter()
    detections_raw, _ = await run_in_threadpool(detector.detect, image)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    # ── enrich with rules ─────────────────────────────────────────────────
    detections: list[Detection] = []
    for det in detections_raw:
        rule = rules.get(det.material)
        detections.append(
            Detection(
                bbox=BoundingBox(**det.bbox),
                material=det.material,
                confidence=round(det.confidence, 4),
                biodegradable=rule["biodegradable"],
                recyclable=rule["recyclable"],
                category=rule.get("category"),
                bin=rule.get("bin"),
                instructions=rule.get("instructions"),
                polygon=det.polygon,
            )
        )

    return PredictResponse(
        image_size=ImageSize(width=orig_w, height=orig_h),
        count=len(detections),
        detections=detections,
        inference_ms=round(inference_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
