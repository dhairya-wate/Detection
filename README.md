# EcoVision AI — TACO Dataset Real-Time Camera Waste Detector

Real-time waste detection and environmental classification system powered by **YOLOv8** and the **TACO (Trash Annotations in Context)** dataset.

The system detects waste items directly via **Camera (Webcam)**, classifies their materials, determines whether each item is **Biodegradable** and **Recyclable**, and provides recommended disposal bins and handling instructions.

---

## 🚀 Quick Start: Camera Detection

You can run camera detection in **two convenient ways**:

### 1. Modern Web Camera Interface (Recommended)

Run from the repository root:
```bash
python run.py
```
Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your browser:
- 📹 **Live Camera Feed**: Instant webcam streaming with bounding boxes and glowing HUD
- 🏷️ **Real-Time Classification**: Instant identification of waste materials
- 🌱 **Eco Badges**: Biodegradable vs. Non-biodegradable and Recyclable vs. Landfill
- 📥 **Smart Disposal Bins**: Specific municipal bin recommendations for each item
- 📸 **Snapshot & Sound**: Download annotated detection frames and hear audio feedback
- 📁 **Test Image Mode**: Drag & drop test images directly into the browser

### 2. Desktop OpenCV Camera Window

Run the standalone desktop camera detection script:
```bash
python detect_camera.py
```
Or via the runner:
```bash
python run.py --desktop
```

#### Keyboard Shortcuts:
| Key | Action |
|---|---|
| `q` or `ESC` | Exit camera window |
| `s` | Save annotated snapshot image |
| `space` | Pause / Freeze detection frame |
| `+` / `-` | Increase / Decrease confidence threshold |
| `c` | Switch camera device index (0, 1, ...) |
| `h` | Toggle HUD telemetry overlay |

---

## 🏷️ TACO Dataset Classes & Rules

Trained on the **TACO (Trash Annotations in Context)** dataset with 10 waste categories:

| # | Class Name | Material Category | Biodegradable | Recyclable | Recommended Disposal Bin |
|---|---|---|---|---|---|
| 0 | `battery` | Hazardous / E-Waste | 🚫 No | ♻️ Yes | Battery & E-Waste Drop-off Hub |
| 1 | `can` | Metal (Aluminum/Steel) | 🚫 No | ♻️ Yes | Metal & Cans Recycling Bin |
| 2 | `cardboard` | Paper / Cardboard | 🌱 Yes | ♻️ Yes | Paper & Cardboard Recycling Bin |
| 3 | `drink carton` | Composite Carton | 🚫 No | ♻️ Yes | Cartons / Mixed Recycling Bin |
| 4 | `glass bottle` | Glass | 🚫 No | ♻️ Yes | Glass Recycling Bin |
| 5 | `paper` | Paper | 🌱 Yes | ♻️ Yes | Paper & Cardboard Recycling Bin |
| 6 | `plastic bag` | Soft Plastic Film | 🚫 No | 🚫 Landfill* | Supermarket Soft Plastics Drop-off |
| 7 | `plastic bottle` | Rigid Plastic (PET/HDPE) | 🚫 No | ♻️ Yes | Plastics & Containers Recycling Bin |
| 8 | `plastic bottle cap`| Rigid Plastic (PP/HDPE) | 🚫 No | ♻️ Yes | Plastics Bin (Attached to bottle) |
| 9 | `pop tab` | Metal (Aluminum) | 🚫 No | ♻️ Yes | Metal & Cans Recycling Bin |

*\*Soft plastics jam mechanical recycling sorters; drop them at dedicated collection points.*

---

## 🛠️ REST API Endpoints

The FastAPI server provides endpoints for client apps and camera streaming:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` or `/camera` | Live Web Camera Detection Interface |
| `GET` | `/health` | System health and model status check |
| `GET` | `/rules` | All TACO waste classification and recycling rules |
| `POST` | `/predict` | Multipart image file upload (`JPEG`, `PNG`, `WebP`) |
| `POST` | `/predict_frame` | Base64-encoded camera frame prediction (high FPS) |
| `GET` | `/docs` | Interactive Swagger API documentation |

### Example Base64 Frame Request
```bash
POST /predict_frame
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,...",
  "conf_thres": 0.40
}
```

---

## 🧪 Running Automated Tests

Run the complete test suite:
```bash
pytest waste-api/tests
```
All 59 unit and integration tests validate:
- Preprocessing & letterbox affine transforms
- ONNX inference and pure NumPy NMS postprocessing
- Full TACO rules completeness and mapping
- Web camera interface routes and base64 prediction payloads
- API authentication and image validation
