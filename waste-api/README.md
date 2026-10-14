# Waste Detection API

A production-ready REST API that accepts an image, runs a YOLOv8 ONNX model on CPU to detect waste items, classifies each item's material, and enriches each detection with biodegradability and recyclability flags derived from a configuration file.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure](#project-structure)
4. [Placing the Model](#placing-the-model)
5. [Local Development Setup](#local-development-setup)
6. [Running Locally](#running-locally)
7. [Running Tests](#running-tests)
8. [Docker Build & Run](#docker-build--run)
9. [API Reference](#api-reference)
10. [Customising the Rules (No Rebuild Required)](#customising-the-rules-no-rebuild-required)
11. [Performance Tuning](#performance-tuning)
12. [Configuration Reference](#configuration-reference)

---

## Architecture Overview

```
POST /predict
     │
     ├─ Content-type check (JPEG / PNG / WebP)
     ├─ Size check (MAX_UPLOAD_MB)
     ├─ Pillow decode  →  RGB ndarray
     │
     ├─ WasteDetector.detect()  [run_in_threadpool – non-blocking]
     │       ├─ letterbox resize to 640×640
     │       ├─ ONNX Runtime (CPUExecutionProvider)
     │       └─ postprocess: sigmoid, filter, NMS (NumPy)
     │
     └─ RulesEngine.get(material)
             └─ biodegradable + recyclable from rules.json
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11 |
| pip | ≥ 23 |
| Docker + Docker Compose | any recent version (optional) |

---

## Project Structure

```
waste-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, routes, startup/shutdown
│   ├── config.py        # Settings loaded from environment variables
│   ├── inference.py     # WasteDetector: preprocess, ONNX run, postprocess, NMS
│   ├── rules.py         # Loads config/rules.json, maps material → properties
│   └── schemas.py       # Pydantic models for responses
├── models/
│   └── .gitkeep         # Place best.onnx here
├── config/
│   └── rules.json       # Material → {biodegradable, recyclable}
├── tests/
│   ├── test_inference.py
│   ├── test_rules.py
│   └── test_api.py
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Placing the Model

The repository ships **without** a model file.  
Copy your exported YOLOv8 ONNX model into the `models/` directory:

```bash
cp /path/to/your/best.onnx models/best.onnx
```

The model must have been exported from YOLOv8 in ONNX format:

```bash
# From your training environment:
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=17
```

Expected output shape: `(1, 4 + num_classes, num_anchors)` — standard YOLOv8 ONNX export.

> **Class order matters.** The class indices in the model must match the order in `app/rules.py`:  
> `paper, cardboard, organic, glass, metal, plastic, textile, ewaste`

---

## Local Development Setup

```bash
# 1. Clone / enter the project directory
cd waste-api

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install runtime + dev dependencies
pip install -r requirements-dev.txt

# 4. (Optional) create a .env file for local overrides
cp .env.example .env
```

---

## Running Locally

```bash
# From the waste-api/ directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs are available at: http://localhost:8000/docs

> **Note:** The server will fail to start if `models/best.onnx` does not exist.

---

## Running Tests

```bash
# Unit tests only (no model file required)
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=app --cov-report=term-missing

# The integration test in test_api.py is automatically skipped when
# models/best.onnx is absent.
```

---

## Docker Build & Run

### Build

```bash
docker build -t waste-api:latest .
```

### Run with Docker Compose (recommended)

```bash
# 1. Create .env from the example
cp .env.example .env

# 2. Place your model
cp /path/to/best.onnx models/best.onnx

# 3. Start the service
docker compose up -d

# 4. Check health
curl http://localhost:8000/health
```

The `models/` and `config/` directories are mounted **read-only** into the container, so you can swap `best.onnx` or update `rules.json` and just restart the container — no rebuild needed:

```bash
cp /path/to/new_model.onnx models/best.onnx
docker compose restart
```

### Run without Compose

```bash
docker run -d \
  --name waste-api \
  -p 8000:8000 \
  -v "$(pwd)/models:/app/models:ro" \
  -v "$(pwd)/config:/app/config:ro" \
  -e CONF_THRES=0.4 \
  -e IOU_THRES=0.5 \
  -e ONNX_THREADS=2 \
  waste-api:latest
```

---

## API Reference

### `GET /health`

Returns service status and model information.

**Response 200**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_file": "best.onnx"
}
```

---

### `POST /predict`

Detect and classify waste items in an uploaded image.

**Headers (optional)**
- `X-API-Key: <your-key>` — required only when `API_KEY` is configured

**Request**
```
Content-Type: multipart/form-data
Field: file — JPEG, PNG, or WebP image (max MAX_UPLOAD_MB)
```

**Example curl request**
```bash
curl -X POST http://localhost:8000/predict \
  -H "X-API-Key: mysecret" \
  -F "file=@/path/to/waste_image.jpg"
```

**Sample JSON response**
```json
{
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "count": 3,
  "detections": [
    {
      "bbox": {
        "x1": 120.5,
        "y1": 340.2,
        "x2": 310.8,
        "y2": 590.1
      },
      "material": "plastic",
      "confidence": 0.8731,
      "biodegradable": false,
      "recyclable": true
    },
    {
      "bbox": {
        "x1": 450.0,
        "y1": 100.0,
        "x2": 680.0,
        "y2": 350.0
      },
      "material": "paper",
      "confidence": 0.9102,
      "biodegradable": true,
      "recyclable": true
    },
    {
      "bbox": {
        "x1": 800.0,
        "y1": 200.0,
        "x2": 1050.0,
        "y2": 500.0
      },
      "material": "ewaste",
      "confidence": 0.7654,
      "biodegradable": false,
      "recyclable": false
    }
  ],
  "inference_ms": 48.32
}
```

**Error responses**

| Status | Condition |
|--------|-----------|
| 400 | Unsupported content type, corrupt / undecodable image |
| 401 | Missing or incorrect `X-API-Key` (when `API_KEY` is set) |
| 413 | Upload exceeds `MAX_UPLOAD_MB` |
| 503 | Model not yet loaded (startup still in progress) |

---

## Customising the Rules (No Rebuild Required)

The `config/rules.json` file controls which materials are considered biodegradable and/or recyclable.  
**No code change or Docker rebuild is needed** — just edit the file and restart:

```json
{
  "paper":     {"biodegradable": true,  "recyclable": true},
  "cardboard": {"biodegradable": true,  "recyclable": true},
  "organic":   {"biodegradable": true,  "recyclable": false},
  "glass":     {"biodegradable": false, "recyclable": true},
  "metal":     {"biodegradable": false, "recyclable": true},
  "plastic":   {"biodegradable": false, "recyclable": true},
  "textile":   {"biodegradable": true,  "recyclable": false},
  "ewaste":    {"biodegradable": false, "recyclable": false}
}
```

> **Constraint:** Every class that the model can predict must have an entry.  
> Removing a class or misspelling a key will cause the server to refuse to start with a clear error message.

When running with Docker Compose, `config/` is a read-only bind mount.  
To apply a rules change:

```bash
# Edit config/rules.json on the host, then:
docker compose restart
```

---

## Performance Tuning

### `ONNX_THREADS`

Controls `intra_op_num_threads` in ONNX Runtime — the number of threads used for parallelising operators within a single inference call.

| Server CPU cores | Recommended `ONNX_THREADS` | Recommended `--workers` |
|------------------|-----------------------------|--------------------------|
| 2                | 2                           | 1                        |
| 4                | 2                           | 2                        |
| 8                | 4                           | 2–4                      |
| 16+              | 4–8                         | 4–8                      |

**Rule of thumb:**  
`ONNX_THREADS × workers ≤ total CPU cores`

### Uvicorn workers

Set via `WEB_CONCURRENCY` env var (default `2`) or `--workers` flag.  
Each worker loads its own ONNX session, so memory usage scales linearly.

### Example `.env` for a 4-core server

```env
ONNX_THREADS=2
WEB_CONCURRENCY=2
CONF_THRES=0.4
IOU_THRES=0.5
MAX_UPLOAD_MB=10
```

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/best.onnx` | Path to the ONNX model file |
| `RULES_PATH` | `config/rules.json` | Path to the material rules file |
| `CONF_THRES` | `0.4` | Minimum detection confidence (0–1) |
| `IOU_THRES` | `0.5` | IoU threshold for NMS |
| `MAX_UPLOAD_MB` | `10` | Maximum upload size in megabytes |
| `ONNX_THREADS` | `2` | ONNX Runtime intra-op thread count |
| `API_KEY` | _(empty)_ | If set, `X-API-Key` header is required |
| `CORS_ORIGINS` | _(empty)_ | Comma-separated allowed CORS origins |
| `WEB_CONCURRENCY` | `2` | Number of Uvicorn worker processes |
