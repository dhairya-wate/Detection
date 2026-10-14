"""
detect_camera.py
================
Live Desktop Camera Waste Detection using OpenCV and the TACO Dataset model.

Features:
  - Real-time webcam feed with YOLOv8 ONNX inference
  - Bounding box rendering with class-specific colors
  - Biodegradability & Recyclability tagging on detected items
  - Real-time FPS, latency, and item telemetry HUD
  - Disposal bin recommendations displayed on screen
  - Keyboard shortcuts:
      'q' or ESC : Exit
      's'        : Save annotated snapshot to disk
      'space'    : Pause / Resume detection
      '+' / '-'  : Increase / decrease confidence threshold
      'c'        : Cycle camera device index (0, 1, ...)
      'h'        : Toggle HUD telemetry

Usage:
    python detect_camera.py
    python detect_camera.py --camera 0 --conf 0.45
    python detect_camera.py --model waste-api/models/best.onnx
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Ensure waste-api is on sys.path
BASE_DIR = Path(__file__).resolve().parent
API_DIR = BASE_DIR / "waste-api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

try:
    from app.inference import WasteDetector  # type: ignore
    from app.rules import MODEL_CLASSES, RulesEngine  # type: ignore
except ImportError:
    from waste_api.app.inference import WasteDetector  # type: ignore
    from waste_api.app.rules import MODEL_CLASSES, RulesEngine  # type: ignore

# ── Color Palette per Material ────────────────────────────────────────────────
COLORS_BGR = {
    "battery": (94, 63, 244),           # Rose / Red
    "can": (11, 158, 245),              # Amber
    "cardboard": (129, 185, 16),        # Emerald Green
    "drink carton": (212, 182, 6),      # Cyan
    "glass bottle": (246, 130, 59),     # Sky Blue
    "paper": (246, 92, 139),            # Violet
    "plastic bag": (153, 72, 236),      # Pink
    "plastic bottle": (233, 165, 14),   # Electric Blue
    "plastic bottle cap": (166, 184, 20),# Teal
    "pop tab": (8, 179, 234),           # Gold
}
DEFAULT_COLOR_BGR = (129, 185, 16)


def draw_hud(
    frame: np.ndarray,
    detections: list,
    rules: RulesEngine,
    fps: float,
    inference_ms: float,
    conf_thres: float,
    paused: bool,
    show_hud: bool = True,
) -> np.ndarray:
    """Render bounding boxes, tags, and telemetry overlay on the frame."""
    h, w = frame.shape[:2]
    annotated = frame.copy()

    # Draw Detections
    primary_item = None
    max_conf = -1.0

    for det in detections:
        bx1 = int(det.bbox["x1"])
        by1 = int(det.bbox["y1"])
        bx2 = int(det.bbox["x2"])
        by2 = int(det.bbox["y2"])
        mat = det.material
        conf = det.confidence

        if conf > max_conf:
            max_conf = conf
            primary_item = det

        rule = rules.get(mat)
        bio = rule.get("biodegradable", False)
        rec = rule.get("recyclable", False)
        color = COLORS_BGR.get(mat.lower(), DEFAULT_COLOR_BGR)

        # 1. Instance Segmentation Mask (if available)
        if hasattr(det, "polygon") and det.polygon and len(det.polygon) >= 3:
            pts = np.array(det.polygon, np.int32).reshape((-1, 1, 2))
            mask_layer = annotated.copy()
            cv2.fillPoly(mask_layer, [pts], color)
            cv2.addWeighted(mask_layer, 0.25, annotated, 0.75, 0, annotated)
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

        # 2. Bounding Box over detected object
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 2, cv2.LINE_AA)

        # Corner brackets for clear target visualization
        corner_len = min(18, max(6, (bx2 - bx1) // 5), max(6, (by2 - by1) // 5))
        # Top-left
        cv2.line(annotated, (bx1, by1), (bx1 + corner_len, by1), color, 3, cv2.LINE_AA)
        cv2.line(annotated, (bx1, by1), (bx1, by1 + corner_len), color, 3, cv2.LINE_AA)
        # Top-right
        cv2.line(annotated, (bx2, by1), (bx2 - corner_len, by1), color, 3, cv2.LINE_AA)
        cv2.line(annotated, (bx2, by1), (bx2, by1 + corner_len), color, 3, cv2.LINE_AA)
        # Bottom-left
        cv2.line(annotated, (bx1, by2), (bx1 + corner_len, by2), color, 3, cv2.LINE_AA)
        cv2.line(annotated, (bx1, by2), (bx1, by2 - corner_len), color, 3, cv2.LINE_AA)
        # Bottom-right
        cv2.line(annotated, (bx2, by2), (bx2 - corner_len, by2), color, 3, cv2.LINE_AA)
        cv2.line(annotated, (bx2, by2), (bx2, by2 - corner_len), color, 3, cv2.LINE_AA)

        # 2. Header Label Pill
        label = f"{mat.upper()} {int(conf * 100)}%"
        (lbl_w, lbl_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        pill_y1 = max(by1 - lbl_h - 10, 0)
        pill_y2 = pill_y1 + lbl_h + 10
        pill_x2 = min(bx1 + lbl_w + 12, w)

        cv2.rectangle(annotated, (bx1, pill_y1), (pill_x2, pill_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (bx1 + 6, pill_y2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 3. Badges tag (BIO / RECYCLE)
        tag_str = f"{'BIO' if bio else 'NON-BIO'} | {'RECYCLE' if rec else 'LANDFILL'}"
        (tw, th), _ = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y1 = min(by2 + 4, h - th - 6)
        if by2 - by1 > 60:
            cv2.rectangle(annotated, (bx1, tag_y1 - 2), (bx1 + tw + 8, tag_y1 + th + 4), (20, 24, 34), -1)
            cv2.rectangle(annotated, (bx1, tag_y1 - 2), (bx1 + tw + 8, tag_y1 + th + 4), color, 1)
            cv2.putText(
                annotated,
                tag_str,
                (bx1 + 4, tag_y1 + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 200) if rec else (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

    if not show_hud:
        return annotated

    # Top Status Bar (Transparent dark strip)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (10, 14, 23), -1)
    cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0, annotated)

    # Telemetry text
    status_text = f"FPS: {fps:4.1f} | Latency: {inference_ms:4.1f}ms | Conf: {int(conf_thres*100)}% | Items: {len(detections)}"
    if paused:
        status_text += " [PAUSED]"
    cv2.putText(annotated, "TACO AI WASTE DETECTOR", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 185, 129), 2, cv2.LINE_AA)
    cv2.putText(annotated, status_text, (w - 480, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 225, 230), 1, cv2.LINE_AA)

    # Bottom Recycling Guidance Banner (if waste is detected)
    if primary_item:
        p_rule = rules.get(primary_item.material)
        bin_name = p_rule.get("bin", "Recycling Bin")
        instructions = p_rule.get("instructions", "Dispose responsibly.")

        overlay_bot = annotated.copy()
        cv2.rectangle(overlay_bot, (0, h - 55), (w, h), (10, 14, 23), -1)
        cv2.addWeighted(overlay_bot, 0.85, annotated, 0.15, 0, annotated)

        guide_line1 = f"Disposal: {bin_name}"
        guide_line2 = f"Tip: {instructions}"
        cv2.putText(annotated, guide_line1, (16, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2, cv2.LINE_AA)
        cv2.putText(annotated, guide_line2[:95], (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 175, 190), 1, cv2.LINE_AA)
    else:
        # Subtle hint
        cv2.putText(annotated, "Point camera at waste item | Press 's' to save snapshot | 'q' to quit", (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 116, 139), 1, cv2.LINE_AA)

    return annotated


def run_camera(
    camera_id: int = 0,
    model_path: str = "waste-api/models/best.onnx",
    rules_path: str = "waste-api/config/rules.json",
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> None:
    """Main camera detection loop."""
    print("=" * 65)
    print("       TACO Dataset Real-Time Camera Waste Detector       ")
    print("=" * 65)

    # Check files
    m_path = Path(model_path)
    if not m_path.exists():
        # Check relative to API_DIR
        candidate = API_DIR / model_path
        if candidate.exists():
            m_path = candidate
        else:
            print(f"[!] ERROR: Model file not found at '{model_path}'")
            sys.exit(1)

    r_path = Path(rules_path)
    if not r_path.exists():
        candidate = API_DIR / rules_path
        if candidate.exists():
            r_path = candidate

    print(f"[*] Loading Rules from: {r_path}")
    rules_engine = RulesEngine(r_path, MODEL_CLASSES)

    print(f"[*] Initializing ONNX Detector from: {m_path}")
    detector = WasteDetector(
        model_path=m_path,
        class_names=MODEL_CLASSES,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
    )
    print(f"[+] Loaded {len(detector.class_names)} classes: {detector.class_names}")

    # Open Camera
    print(f"[*] Opening camera device index {camera_id}...")
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"[!] Fallback: Trying default capture...")
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"[!] ERROR: Could not open camera {camera_id}. Check camera connection.")
        sys.exit(1)

    # Request high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[+] Camera opened successfully: {actual_w}x{actual_h}")
    print("\nControls:")
    print("  'q' or ESC : Exit")
    print("  's'        : Save annotated snapshot")
    print("  'space'    : Pause / Resume detection")
    print("  '+' / '-'  : Change confidence threshold")
    print("  'h'        : Toggle HUD")
    print("-" * 65)

    window_name = "EcoVision TACO Waste Detector (Camera)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1080, 640)

    fps = 0.0
    frame_count = 0
    t_start = time.perf_counter()
    paused = False
    show_hud = True
    last_detections = []
    last_inference_ms = 0.0

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[!] Failed to grab frame from camera.")
                    break

                frame_count += 1
                if frame_count % 5 == 0:
                    t_now = time.perf_counter()
                    fps = 5.0 / (t_now - t_start)
                    t_start = t_now

                # Run inference
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                detections, last_inference_ms = detector.detect(
                    pil_image,
                    conf_thres=conf_thres,
                    iou_thres=iou_thres,
                )
                last_detections = detections

            # Draw HUD
            display_frame = draw_hud(
                frame,
                last_detections,
                rules_engine,
                fps=fps,
                inference_ms=last_inference_ms,
                conf_thres=conf_thres,
                paused=paused,
                show_hud=show_hud,
            )

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                print("[*] Quitting camera detector...")
                break
            elif key == ord(' '):
                paused = not paused
                print(f"[*] Detection {'PAUSED' if paused else 'RESUMED'}")
            elif key in (ord('s'), ord('S')):
                snap_path = BASE_DIR / f"waste_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(str(snap_path), display_frame)
                print(f"[+] Saved snapshot to: {snap_path}")
            elif key in (ord('+'), ord('=')):
                conf_thres = min(0.95, round(conf_thres + 0.05, 2))
                print(f"[*] Confidence threshold increased to: {conf_thres:.2f}")
            elif key in (ord('-'), ord('_')):
                conf_thres = max(0.10, round(conf_thres - 0.05, 2))
                print(f"[*] Confidence threshold decreased to: {conf_thres:.2f}")
            elif key in (ord('h'), ord('H')):
                show_hud = not show_hud
            elif key in (ord('c'), ord('C')):
                # Cycle camera
                cap.release()
                camera_id = 1 if camera_id == 0 else 0
                print(f"[*] Switching to camera {camera_id}...")
                cap = cv2.VideoCapture(camera_id)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[+] Camera released. Clean exit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Camera Waste Detector (TACO Dataset)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--model", type=str, default="waste-api/models/best.onnx", help="Path to ONNX model")
    parser.add_argument("--rules", type=str, default="waste-api/config/rules.json", help="Path to rules.json")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")

    args = parser.parse_args()
    run_camera(
        camera_id=args.camera,
        model_path=args.model,
        rules_path=args.rules,
        conf_thres=args.conf,
        iou_thres=args.iou,
    )
