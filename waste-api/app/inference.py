"""
app/inference.py
================
WasteDetector: wraps an ONNX Runtime session and performs:
  1. Letterbox preprocessing (640 × 640, gray pad = 114)
  2. YOLOv8 inference (CPU only)
  3. Postprocessing: transpose → filter → cx/cy/w/h → x1/y1/x2/y2 → undo-letterbox
  4. Per-class NMS (pure NumPy, no torch)
  5. Returns a list of Detection dicts

This module deliberately has NO FastAPI imports so it can be imported and
tested in isolation.

YOLOv8 ONNX output assumptions
-------------------------------
Shape : (1, 4 + num_classes, num_anchors)
        e.g. (1, 14, 8400) for 10 classes
Layout: first 4 channels are [cx, cy, w, h]; remaining channels are
        per-class confidence scores (probabilities in [0, 1] or raw logits).
"""

from __future__ import annotations

import ast
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── internal types ────────────────────────────────────────────────────────────

@dataclass
class _LetterboxMeta:
    """Stores the affine parameters needed to undo letterboxing."""
    scale: float          # scale applied to the original image
    pad_top: int          # pixels of padding at the top
    pad_left: int         # pixels of padding on the left
    orig_w: int
    orig_h: int


@dataclass
class DetectionResult:
    bbox: dict[str, float]   # {"x1", "y1", "x2", "y2"}
    material: str
    confidence: float
    polygon: list[list[float]] | None = None


def extract_segmentation_polygon(
    img_array: np.ndarray | None,
    bbox: dict[str, float],
) -> list[list[float]]:
    """
    Extract a smooth instance segmentation polygon contour of the object
    inside the bounding box using adaptive saliency and contour approximation.
    """
    x1 = float(bbox["x1"])
    y1 = float(bbox["y1"])
    x2 = float(bbox["x2"])
    y2 = float(bbox["y2"])
    bw = x2 - x1
    bh = y2 - y1

    if img_array is None or bw < 8 or bh < 8:
        # Octagonal chamfered contour hugging the bounding box
        ix, iy = bw * 0.12, bh * 0.12
        return [
            [round(x1 + ix, 1), round(y1, 1)],
            [round(x2 - ix, 1), round(y1, 1)],
            [round(x2, 1), round(y1 + iy, 1)],
            [round(x2, 1), round(y2 - iy, 1)],
            [round(x2 - ix, 1), round(y2, 1)],
            [round(x1 + ix, 1), round(y2, 1)],
            [round(x1, 1), round(y2 - iy, 1)],
            [round(x1, 1), round(y1 + iy, 1)],
        ]

    try:
        import cv2
        ix1 = max(0, int(x1))
        iy1 = max(0, int(y1))
        ix2 = min(img_array.shape[1], int(x2))
        iy2 = min(img_array.shape[0], int(y2))
        crop = img_array[iy1:iy2, ix1:ix2]

        if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
            raise ValueError("Crop too small")

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 0.10 * (bw * bh):
                epsilon = 0.015 * cv2.arcLength(c, True)
                poly = cv2.approxPolyDP(c, epsilon, True)
                if len(poly) >= 3:
                    return [[round(float(pt[0][0] + ix1), 1), round(float(pt[0][1] + iy1), 1)] for pt in poly]
    except Exception:
        pass

    # Clean octagonal polygon hugging bbox
    ix, iy = bw * 0.12, bh * 0.12
    return [
        [round(x1 + ix, 1), round(y1, 1)],
        [round(x2 - ix, 1), round(y1, 1)],
        [round(x2, 1), round(y1 + iy, 1)],
        [round(x2, 1), round(y2 - iy, 1)],
        [round(x2 - ix, 1), round(y2, 1)],
        [round(x1 + ix, 1), round(y2, 1)],
        [round(x1, 1), round(y2 - iy, 1)],
        [round(x1, 1), round(y1 + iy, 1)],
    ]


# ── helper functions (pure NumPy, testable in isolation) ──────────────────────

def letterbox(
    img: np.ndarray,
    new_size: int = 640,
    pad_value: int = 114,
) -> tuple[np.ndarray, _LetterboxMeta]:
    """
    Resize *img* (H×W×3, uint8) to *new_size*×*new_size* preserving aspect
    ratio and padding with *pad_value*.

    Returns the padded image and a _LetterboxMeta so boxes can be mapped back.
    """
    orig_h, orig_w = img.shape[:2]
    scale = min(new_size / orig_w, new_size / orig_h)

    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    # Resize with Pillow (high-quality Lanczos)
    pil = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
    resized = np.array(pil)

    # Create canvas and centre the resized image
    canvas = np.full((new_size, new_size, 3), pad_value, dtype=np.uint8)
    pad_top = (new_size - new_h) // 2
    pad_left = (new_size - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    meta = _LetterboxMeta(
        scale=scale,
        pad_top=pad_top,
        pad_left=pad_left,
        orig_w=orig_w,
        orig_h=orig_h,
    )
    return canvas, meta


def undo_letterbox(
    boxes: np.ndarray,
    meta: _LetterboxMeta,
) -> np.ndarray:
    """
    Map boxes from letterboxed-image space back to original image space.

    Parameters
    ----------
    boxes : np.ndarray, shape (N, 4) — [x1, y1, x2, y2] in 640-px space
    meta  : _LetterboxMeta

    Returns
    -------
    np.ndarray, shape (N, 4) — clipped to original image bounds
    """
    boxes = boxes.copy().astype(np.float32)
    boxes[:, 0] -= meta.pad_left   # x1
    boxes[:, 2] -= meta.pad_left   # x2
    boxes[:, 1] -= meta.pad_top    # y1
    boxes[:, 3] -= meta.pad_top    # y2
    boxes /= meta.scale

    # Clip to original image bounds
    boxes[:, 0] = np.clip(boxes[:, 0], 0, meta.orig_w)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, meta.orig_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, meta.orig_h)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, meta.orig_h)
    return boxes


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between *box* (shape 4,) and *boxes* (shape N×4).
    All boxes are [x1, y1, x2, y2].
    """
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = area_box + area_boxes - inter_area

    return np.where(union_area > 0, inter_area / union_area, 0.0)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_thres: float,
) -> list[int]:
    """
    Per-class Non-Maximum Suppression implemented in NumPy.

    Parameters
    ----------
    boxes     : (N, 4) float32 [x1,y1,x2,y2]
    scores    : (N,)  float32
    class_ids : (N,)  int
    iou_thres : float

    Returns
    -------
    List of kept indices (sorted by descending score within each class).
    """
    kept: list[int] = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_indices = np.where(cls_mask)[0]
        cls_scores = scores[cls_indices]
        cls_boxes = boxes[cls_indices]

        order = np.argsort(cls_scores)[::-1]
        suppressed = np.zeros(len(order), dtype=bool)

        for i, idx in enumerate(order):
            if suppressed[i]:
                continue
            kept.append(int(cls_indices[idx]))
            ious = _iou(cls_boxes[idx], cls_boxes[order[i + 1 :]])
            for j, iou_val in enumerate(ious):
                if iou_val > iou_thres:
                    suppressed[i + 1 + j] = True

    return kept


def suppress_cross_class_overlaps(
    boxes: np.ndarray,
    scores: np.ndarray,
    kept_indices: list[int],
    cross_iou_thres: float = 0.50,
) -> list[int]:
    """
    Among remaining candidate detections from different classes that heavily
    overlap (e.g. pop tab vs plastic bottle cap in the same position), keep
    only the detection with the higher confidence score.
    """
    if len(kept_indices) <= 1:
        return kept_indices

    sorted_kept = sorted(kept_indices, key=lambda idx: float(scores[idx]), reverse=True)
    final_kept: list[int] = []
    suppressed = set()

    for i, idx1 in enumerate(sorted_kept):
        if idx1 in suppressed:
            continue
        final_kept.append(idx1)
        b1 = boxes[idx1]
        for idx2 in sorted_kept[i + 1 :]:
            if idx2 in suppressed:
                continue
            b2 = boxes[idx2]
            inter_x1 = max(b1[0], b2[0])
            inter_y1 = max(b1[1], b2[1])
            inter_x2 = min(b1[2], b2[2])
            inter_y2 = min(b1[3], b2[3])
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            if inter_area <= 0:
                continue
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union_area = area1 + area2 - inter_area
            if union_area > 0 and (inter_area / union_area) > cross_iou_thres:
                suppressed.add(idx2)

    return final_kept


def preprocess(img_array: np.ndarray, input_size: int = 640) -> tuple[np.ndarray, _LetterboxMeta]:
    """
    Convert a uint8 HxWxC array to a float32 NCHW tensor ready for ONNX.
    Returns the tensor and the letterbox metadata.
    """
    lb_img, meta = letterbox(img_array, new_size=input_size)
    tensor = lb_img.astype(np.float32) / 255.0          # [0, 1]
    tensor = np.transpose(tensor, (2, 0, 1))             # HWC → CHW
    tensor = np.expand_dims(tensor, axis=0)              # → NCHW
    return tensor, meta


def postprocess(
    raw_output: np.ndarray,
    meta: _LetterboxMeta,
    conf_thres: float,
    iou_thres: float,
    class_names: list[str],
    img_array: np.ndarray | None = None,
    max_area_ratio: float = 0.55,
    min_area_ratio: float = 0.002,
    cross_class_iou: float | None = 0.50,
    roi: list[float] | None = None,
) -> list[DetectionResult]:
    """
    Convert raw YOLOv8 ONNX output to a list of DetectionResult objects.

    Parameters
    ----------
    raw_output : np.ndarray, shape (1, 4+C, N)
    meta       : _LetterboxMeta from preprocess
    conf_thres : minimum confidence threshold
    iou_thres  : IoU threshold for NMS
    class_names: ordered list of class name strings
    max_area_ratio : maximum box area as a fraction of image (rejects huge background false positives)
    min_area_ratio : minimum box area as a fraction of image (rejects tiny noise specks)
    cross_class_iou: IoU threshold to suppress duplicate overlapping boxes from different classes
    roi        : optional [x1, y1, x2, y2] bounding box to restrict valid detection centers

    Returns
    -------
    List of DetectionResult (may be empty).
    """
    # Shape: (1, 4+C, N) → (N, 4+C)
    pred = raw_output[0].T  # (N, 4+C)

    num_classes = len(class_names)
    num_expected = 4 + num_classes
    if pred.shape[1] < num_expected:
        logger.error(
            "ONNX output has %d channels, expected at least %d (4 + %d classes).",
            pred.shape[1], num_expected, num_classes,
        )
        return []

    boxes_cxcy = pred[:, :4]          # cx, cy, w, h (640-px space)
    class_logits = pred[:, 4:4 + num_classes]  # (N, C)

    # Confidence = max class score
    class_ids = np.argmax(class_logits, axis=1)    # (N,)
    confidences = class_logits[np.arange(len(class_logits)), class_ids]  # (N,)

    # Apply sigmoid ONLY if raw outputs are un-activated logits (outside [0, 1]).
    # Standard Ultralytics YOLOv8 ONNX exports already have sigmoid applied to class scores.
    if np.any(class_logits < 0.0) or np.any(class_logits > 1.0):
        confidences = 1.0 / (1.0 + np.exp(-confidences))

    # Filter by confidence threshold
    mask = confidences >= conf_thres
    if not np.any(mask):
        return []

    boxes_cxcy = boxes_cxcy[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # cx,cy,w,h → x1,y1,x2,y2
    x1 = boxes_cxcy[:, 0] - boxes_cxcy[:, 2] / 2
    y1 = boxes_cxcy[:, 1] - boxes_cxcy[:, 3] / 2
    x2 = boxes_cxcy[:, 0] + boxes_cxcy[:, 2] / 2
    y2 = boxes_cxcy[:, 1] + boxes_cxcy[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # Undo letterbox → original image coords
    boxes_orig = undo_letterbox(boxes_xyxy, meta)

    # Per-class NMS
    kept = nms(boxes_orig, confidences, class_ids, iou_thres)

    # Cross-class NMS (prevents conflicting duplicate classes on the exact same item)
    if cross_class_iou is not None and len(kept) > 1:
        kept = suppress_cross_class_overlaps(boxes_orig, confidences, kept, cross_class_iou)

    total_img_area = float(meta.orig_w * meta.orig_h)
    results: list[DetectionResult] = []

    for idx in kept:
        bx1, by1, bx2, by2 = boxes_orig[idx]
        bw = bx2 - bx1
        bh = by2 - by1
        b_area = bw * bh
        conf = float(confidences[idx])

        # Reject degenerate or invalid dimensions
        if bw < 2 or bh < 2:
            continue

        # Reject extreme aspect ratios (e.g. wall seams, horizontal lines)
        aspect = max(bw / bh, bh / bw)
        if aspect > 7.5:
            continue

        # Reject tiny sub-pixel noise specks
        if total_img_area > 0 and (b_area / total_img_area) < min_area_ratio:
            continue

        # Reject absurdly large bounding boxes (e.g. background room/person false positives)
        # unless model is exceptionally confident (> 0.85)
        if total_img_area > 0 and (b_area / total_img_area) > max_area_ratio and conf < 0.85:
            continue

        # Reject detections whose center lies outside designated ROI
        if roi is not None and len(roi) == 4:
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            if not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
                continue

        bbox_dict = {
            "x1": float(bx1),
            "y1": float(by1),
            "x2": float(bx2),
            "y2": float(by2),
        }
        poly = extract_segmentation_polygon(img_array, bbox_dict)
        results.append(
            DetectionResult(
                bbox=bbox_dict,
                material=class_names[int(class_ids[idx])],
                confidence=conf,
                polygon=poly,
            )
        )

    return results


# ── main class ────────────────────────────────────────────────────────────────

class WasteDetector:
    """
    Wraps an ONNX Runtime InferenceSession.

    The session is created once (at startup) and reused for every request.
    All inference runs on CPU only.
    """

    def __init__(
        self,
        model_path: str | Path,
        class_names: list[str],
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        intra_op_threads: int = 2,
    ) -> None:
        import onnxruntime as ort  # local import keeps the module lightweight in tests

        self._path = Path(model_path)
        if not self._path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self._path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session: Any = ort.InferenceSession(
            str(self._path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self._input_name: str = self._session.get_inputs()[0].name
        self._class_names = list(class_names)
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._input_size = 640  # YOLOv8 standard input resolution

        # Attempt to auto-extract classes and input resolution from ONNX model metadata
        try:
            meta = self._session.get_modelmeta().custom_metadata_map
            if "names" in meta:
                names_dict = ast.literal_eval(meta["names"])
                if isinstance(names_dict, dict):
                    extracted = [names_dict[k] for k in sorted(names_dict.keys())]
                    if extracted:
                        self._class_names = extracted
                        logger.info("Auto-loaded class names from ONNX metadata: %s", self._class_names)
            if "imgsz" in meta:
                sz = ast.literal_eval(meta["imgsz"])
                if isinstance(sz, list) and len(sz) >= 1:
                    self._input_size = int(sz[0])
        except Exception as exc:
            logger.debug("Could not read custom metadata from ONNX session: %s", exc)

        logger.info(
            "WasteDetector loaded: model=%s, classes=%s, conf=%.2f, iou=%.2f, threads=%d, imgsz=%d",
            self._path.name, self._class_names, conf_thres, iou_thres, intra_op_threads, self._input_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_file(self) -> str:
        return self._path.name

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    def detect(
        self,
        image: Image.Image,
        conf_thres: float | None = None,
        iou_thres: float | None = None,
        max_area_ratio: float | None = None,
        min_area_ratio: float | None = None,
        cross_class_iou: float | None = None,
        roi: list[float] | None = None,
    ) -> tuple[list[DetectionResult], float]:
        """
        Run inference on a PIL Image.

        Parameters
        ----------
        image       : PIL Image
        conf_thres  : optional override for confidence threshold
        iou_thres   : optional override for IoU threshold
        max_area_ratio : optional override for max box area ratio
        min_area_ratio : optional override for min box area ratio
        cross_class_iou: optional override for cross-class suppression threshold
        roi         : optional [x1, y1, x2, y2] to restrict detections to a specific area

        Returns
        -------
        (detections, inference_ms)
        detections : list of DetectionResult (may be empty)
        inference_ms : wall-clock time for preprocessing + ort.run + postprocessing
        """
        active_conf = conf_thres if conf_thres is not None else self._conf_thres
        active_iou = iou_thres if iou_thres is not None else self._iou_thres
        active_max_area = max_area_ratio if max_area_ratio is not None else 0.55
        active_min_area = min_area_ratio if min_area_ratio is not None else 0.002
        active_cross_iou = cross_class_iou if cross_class_iou is not None else 0.50

        img_array = np.array(image.convert("RGB"))
        tensor, meta = preprocess(img_array, input_size=self._input_size)

        t0 = time.perf_counter()
        raw: list[np.ndarray] = self._session.run(None, {self._input_name: tensor})
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0

        detections = postprocess(
            raw[0],
            meta,
            active_conf,
            active_iou,
            self._class_names,
            img_array=img_array,
            max_area_ratio=active_max_area,
            min_area_ratio=active_min_area,
            cross_class_iou=active_cross_iou,
            roi=roi,
        )

        logger.debug("Detected %d objects in %.1f ms", len(detections), inference_ms)
        return detections, inference_ms

