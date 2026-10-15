"""
tests/test_inference.py
=======================
Unit tests for the pure-NumPy inference helpers in app.inference.
All tests mock the ONNX session – no real model file is required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.inference import (
    _LetterboxMeta,
    DetectionResult,
    WasteDetector,
    _iou,
    letterbox,
    nms,
    postprocess,
    preprocess,
    undo_letterbox,
)

# ── class fixtures ────────────────────────────────────────────────────────────

CLASSES = [
    "paper", "cardboard", "organic", "glass",
    "metal", "plastic", "textile", "ewaste",
]
NUM_CLASSES = len(CLASSES)


# ── letterbox round-trip ──────────────────────────────────────────────────────

class TestLetterbox:
    def _make_image(self, w: int, h: int) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_square_image_no_padding(self) -> None:
        img = self._make_image(640, 640)
        out, meta = letterbox(img, new_size=640)
        assert out.shape == (640, 640, 3)
        assert meta.pad_top == 0
        assert meta.pad_left == 0
        assert abs(meta.scale - 1.0) < 1e-6

    def test_wide_image_pads_top_bottom(self) -> None:
        """Landscape 1280×640 → scale=0.5 → resized 640×320 → pad 160 top/bottom."""
        img = self._make_image(1280, 640)
        out, meta = letterbox(img, new_size=640)
        assert out.shape == (640, 640, 3)
        assert abs(meta.scale - 0.5) < 1e-6
        assert meta.pad_left == 0
        assert meta.pad_top == (640 - 320) // 2

    def test_tall_image_pads_left_right(self) -> None:
        """Portrait 640×1280 → scale=0.5 → resized 320×640 → pad 160 left/right."""
        img = self._make_image(640, 1280)
        out, meta = letterbox(img, new_size=640)
        assert out.shape == (640, 640, 3)
        assert abs(meta.scale - 0.5) < 1e-6
        assert meta.pad_top == 0
        assert meta.pad_left == (640 - 320) // 2


class TestUndoLetterbox:
    def test_round_trip_wide_image(self) -> None:
        """
        Boxes in 640-space should map back to original image coords.
        Wide 1280×640 image: scale=0.5, pad_top=160, pad_left=0.
        A box at [0, 160, 640, 480] in letterbox space should map
        back to [0, 0, 1280, 640] in original space.
        """
        meta = _LetterboxMeta(
            scale=0.5, pad_top=160, pad_left=0, orig_w=1280, orig_h=640
        )
        boxes = np.array([[0.0, 160.0, 640.0, 480.0]])
        result = undo_letterbox(boxes, meta)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 1280.0, 640.0], atol=1.0)

    def test_round_trip_tall_image(self) -> None:
        """Portrait 640×1280: scale=0.5, pad_top=0, pad_left=160."""
        meta = _LetterboxMeta(
            scale=0.5, pad_top=0, pad_left=160, orig_w=640, orig_h=1280
        )
        boxes = np.array([[160.0, 0.0, 480.0, 640.0]])
        result = undo_letterbox(boxes, meta)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 640.0, 1280.0], atol=1.0)

    def test_clipping(self) -> None:
        """Boxes outside the original image bounds should be clipped."""
        meta = _LetterboxMeta(
            scale=1.0, pad_top=0, pad_left=0, orig_w=100, orig_h=100
        )
        boxes = np.array([[-10.0, -10.0, 150.0, 150.0]])
        result = undo_letterbox(boxes, meta)
        assert result[0, 0] == 0.0
        assert result[0, 1] == 0.0
        assert result[0, 2] == 100.0
        assert result[0, 3] == 100.0


# ── IoU ───────────────────────────────────────────────────────────────────────

class TestIou:
    def test_identical_boxes_iou_is_one(self) -> None:
        box = np.array([0.0, 0.0, 10.0, 10.0])
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        assert abs(_iou(box, boxes)[0] - 1.0) < 1e-6

    def test_non_overlapping_boxes_iou_is_zero(self) -> None:
        box = np.array([0.0, 0.0, 5.0, 5.0])
        boxes = np.array([[10.0, 10.0, 20.0, 20.0]])
        assert _iou(box, boxes)[0] == 0.0

    def test_half_overlap(self) -> None:
        # box A: [0,0,10,10] area=100; box B: [5,0,15,10] area=100; overlap=[5,0,10,10]=50
        # IoU = 50 / (100+100-50) = 50/150 ≈ 0.333
        box = np.array([0.0, 0.0, 10.0, 10.0])
        boxes = np.array([[5.0, 0.0, 15.0, 10.0]])
        iou_val = _iou(box, boxes)[0]
        assert abs(iou_val - 50 / 150) < 1e-5


# ── NMS ───────────────────────────────────────────────────────────────────────

class TestNms:
    def test_removes_overlapping_same_class(self) -> None:
        """Two highly overlapping boxes of the same class → keep only highest-scored."""
        boxes = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],  # high overlap with box 0
        ])
        scores = np.array([0.9, 0.8])
        class_ids = np.array([0, 0])
        kept = nms(boxes, scores, class_ids, iou_thres=0.5)
        assert len(kept) == 1
        assert 0 in kept

    def test_keeps_different_class_overlapping_boxes(self) -> None:
        """Overlapping boxes of different classes must both survive NMS."""
        boxes = np.array([
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],  # same position, different class
        ])
        scores = np.array([0.9, 0.85])
        class_ids = np.array([0, 1])
        kept = nms(boxes, scores, class_ids, iou_thres=0.5)
        assert len(kept) == 2

    def test_empty_input(self) -> None:
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)
        class_ids = np.zeros(0, dtype=int)
        kept = nms(boxes, scores, class_ids, iou_thres=0.5)
        assert kept == []

    def test_single_box_always_kept(self) -> None:
        boxes = np.array([[0.0, 0.0, 5.0, 5.0]])
        scores = np.array([0.7])
        class_ids = np.array([2])
        kept = nms(boxes, scores, class_ids, iou_thres=0.5)
        assert len(kept) == 1


# ── postprocess ───────────────────────────────────────────────────────────────

def _make_raw_output(
    cx: float, cy: float, w: float, h: float,
    cls_idx: int, logit: float, num_classes: int = NUM_CLASSES, num_boxes: int = 1
) -> np.ndarray:
    """
    Build a synthetic YOLOv8 raw output tensor (1, 4+C, N).
    Only the first box slot is filled.

    All class channels default to -100.0 (sigmoid ≈ 0) so that only the
    explicitly-set cls_idx has a meaningful score.
    """
    total_channels = 4 + num_classes
    # Start with very negative logits for all class channels
    raw = np.full((1, total_channels, num_boxes), -100.0, dtype=np.float32)
    # Box coordinates (not logits, so zero them out)
    raw[0, 0, 0] = cx
    raw[0, 1, 0] = cy
    raw[0, 2, 0] = w
    raw[0, 3, 0] = h
    # Set the target class logit
    raw[0, 4 + cls_idx, 0] = logit
    return raw


class TestPostprocess:
    def _meta(self) -> _LetterboxMeta:
        return _LetterboxMeta(scale=1.0, pad_top=0, pad_left=0, orig_w=640, orig_h=640)

    def test_single_detection_above_threshold(self) -> None:
        # logit=5.0 → sigmoid≈0.993
        raw = _make_raw_output(cx=100, cy=200, w=50, h=80, cls_idx=0, logit=5.0)
        meta = self._meta()
        results = postprocess(raw, meta, conf_thres=0.4, iou_thres=0.5, class_names=CLASSES)
        assert len(results) == 1
        assert results[0].material == "paper"
        assert results[0].confidence > 0.4

    def test_detection_below_threshold_returns_empty(self) -> None:
        # logit=-2.0 → sigmoid≈0.119, below conf_thres=0.4
        # All other class logits are -100 (sigmoid≈0), so no class wins above threshold
        raw = _make_raw_output(cx=100, cy=200, w=50, h=80, cls_idx=0, logit=-2.0)
        meta = self._meta()
        results = postprocess(raw, meta, conf_thres=0.4, iou_thres=0.5, class_names=CLASSES)
        assert results == []

    def test_empty_output_returns_empty(self) -> None:
        raw = np.zeros((1, 4 + NUM_CLASSES, 0), dtype=np.float32)
        meta = self._meta()
        results = postprocess(raw, meta, conf_thres=0.4, iou_thres=0.5, class_names=CLASSES)
        assert results == []

    def test_bbox_coordinates_correct(self) -> None:
        """cx=320,cy=320,w=640,h=640 → box covers full 640×640 image."""
        raw = _make_raw_output(cx=320, cy=320, w=640, h=640, cls_idx=1, logit=6.0)
        meta = self._meta()
        results = postprocess(raw, meta, conf_thres=0.4, iou_thres=0.5, class_names=CLASSES)
        assert len(results) == 1
        bbox = results[0].bbox
        assert bbox["x1"] == pytest.approx(0.0, abs=1.0)
        assert bbox["y1"] == pytest.approx(0.0, abs=1.0)
        assert bbox["x2"] == pytest.approx(640.0, abs=1.0)
        assert bbox["y2"] == pytest.approx(640.0, abs=1.0)

    def test_oversized_moderate_confidence_box_rejected(self) -> None:
        """A huge box (>55% of image) with moderate confidence (<0.85) is suppressed as room/background noise."""
        # logit=0.2 -> sigmoid ~ 0.55 (above conf_thres=0.4, but < 0.85 and area=100%)
        raw = _make_raw_output(cx=320, cy=320, w=600, h=600, cls_idx=1, logit=0.2)
        meta = self._meta()
        results = postprocess(raw, meta, conf_thres=0.4, iou_thres=0.5, class_names=CLASSES, max_area_ratio=0.55)
        assert results == []

    def test_roi_filters_outside_detection(self) -> None:
        """A box centered outside the designated target ROI is suppressed."""
        # Box centered at (100, 100)
        raw = _make_raw_output(cx=100, cy=100, w=50, h=50, cls_idx=0, logit=6.0)
        meta = self._meta()
        # ROI is in the center [200, 200, 440, 440]
        results = postprocess(raw, meta, conf_thres=0.4, iou_thres=0.5, class_names=CLASSES, roi=[200.0, 200.0, 440.0, 440.0])
        assert results == []

    def test_roi_keeps_inside_detection(self) -> None:
        """A box centered inside the designated target ROI is preserved."""
        # Box centered at (320, 320)
        raw = _make_raw_output(cx=320, cy=320, w=60, h=60, cls_idx=0, logit=6.0)
        meta = self._meta()
        results = postprocess(raw, meta, conf_thres=0.4, iou_thres=0.5, class_names=CLASSES, roi=[200.0, 200.0, 440.0, 440.0])
        assert len(results) == 1
        assert results[0].material == "paper"


# ── WasteDetector (mocked session) ───────────────────────────────────────────

def _build_mock_session(raw_output: np.ndarray) -> MagicMock:
    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [MagicMock(name="images")]
    mock_session.run.return_value = [raw_output]
    return mock_session


def _build_detector_with_mock_session(tmp_path, raw_output: np.ndarray) -> WasteDetector:
    """
    Construct a WasteDetector without touching the ONNX runtime:
    - patch onnxruntime so __init__ doesn't try to open a real model
    - then inject a mock session directly onto the object
    """
    import onnxruntime as ort_module  # noqa: F401  (imported only to get the real module path)

    model_path = tmp_path / "dummy.onnx"
    model_path.write_bytes(b"\x00" * 16)

    mock_session = _build_mock_session(raw_output)
    mock_opts = MagicMock()

    with patch("onnxruntime.SessionOptions", return_value=mock_opts), \
         patch("onnxruntime.GraphOptimizationLevel") as mock_gol, \
         patch("onnxruntime.InferenceSession", return_value=mock_session):
        mock_gol.ORT_ENABLE_ALL = 99
        detector = WasteDetector(
            model_path=model_path,
            class_names=CLASSES,
            conf_thres=0.4,
            iou_thres=0.5,
            intra_op_threads=1,
        )
    # Ensure the mock session is active even after __init__ (defensive)
    detector._session = mock_session
    return detector


class TestWasteDetector:
    def test_detect_returns_empty_for_below_threshold(self, tmp_path) -> None:
        # logit=-5 → sigmoid ≈ 0.0067, well below conf_thres=0.4
        raw = _make_raw_output(cx=100, cy=200, w=50, h=80, cls_idx=0, logit=-5.0)
        det = _build_detector_with_mock_session(tmp_path, raw)
        image = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        results, ms = det.detect(image)
        assert results == []
        assert ms >= 0.0

    def test_detect_returns_detection_above_threshold(self, tmp_path) -> None:
        # logit=8.0 → sigmoid ≈ 0.9997, well above threshold
        raw = _make_raw_output(cx=320, cy=320, w=100, h=100, cls_idx=2, logit=8.0)
        det = _build_detector_with_mock_session(tmp_path, raw)
        image = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        results, ms = det.detect(image)
        assert len(results) == 1
        assert results[0].material == "organic"

    def test_detect_returns_inference_time_positive(self, tmp_path) -> None:
        raw = _make_raw_output(cx=320, cy=320, w=100, h=100, cls_idx=0, logit=8.0)
        det = _build_detector_with_mock_session(tmp_path, raw)
        image = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        _, ms = det.detect(image)
        assert ms >= 0.0

    def test_model_file_property(self, tmp_path) -> None:
        raw = np.zeros((1, 4 + NUM_CLASSES, 1), dtype=np.float32)
        det = _build_detector_with_mock_session(tmp_path, raw)
        assert det.model_file == "dummy.onnx"
