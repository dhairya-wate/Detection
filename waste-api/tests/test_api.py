"""
tests/test_api.py
=================
API-level tests using FastAPI TestClient.
The WasteDetector is monkeypatched so no real model is required.

Strategy
--------
We bypass the FastAPI lifespan entirely by directly injecting _detector and
_rules into app.main's module globals *before* the TestClient is created.
The TestClient is created with raise_server_exceptions=False so that
expected error responses (400, 401, 413) come back as normal HTTP responses.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    """Return the bytes of a minimal JPEG image."""
    img = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    img = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── fake detector ─────────────────────────────────────────────────────────────

def _make_fake_detector() -> MagicMock:
    """Return a mock WasteDetector that returns zero detections."""
    fake = MagicMock()
    fake.model_file = "best.onnx"
    fake.detect.return_value = ([], 5.0)  # (detections, inference_ms)
    return fake


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path) -> Generator[TestClient, None, None]:
    """
    Build a TestClient with:
      - A real rules.json loaded from config/rules.json
      - _detector and _rules injected directly, bypassing the lifespan
    """
    from app.rules import MODEL_CLASSES, RulesEngine
    import app.main as main_module

    rules_path = Path(__file__).parent.parent / "config" / "rules.json"
    fake_detector = _make_fake_detector()
    rules_engine = RulesEngine(rules_path, MODEL_CLASSES)

    # Inject directly - bypasses the lifespan model-loading
    main_module._detector = fake_detector
    main_module._rules = rules_engine

    # Use a no-op lifespan so TestClient doesn't try to load the real model
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop_lifespan(app):
        yield

    original_lifespan = main_module.app.router.lifespan_context
    main_module.app.router.lifespan_context = _noop_lifespan

    try:
        with TestClient(main_module.app, raise_server_exceptions=True) as c:
            yield c
    finally:
        main_module.app.router.lifespan_context = original_lifespan
        main_module._detector = None
        main_module._rules = None


@pytest.fixture()
def client_with_key(tmp_path) -> Generator[TestClient, None, None]:
    """Client where API_KEY='secret' is required."""
    from app.rules import MODEL_CLASSES, RulesEngine
    from app.config import settings
    import app.main as main_module

    rules_path = Path(__file__).parent.parent / "config" / "rules.json"
    fake_detector = _make_fake_detector()
    rules_engine = RulesEngine(rules_path, MODEL_CLASSES)

    main_module._detector = fake_detector
    main_module._rules = rules_engine

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop_lifespan(app):
        yield

    original_lifespan = main_module.app.router.lifespan_context
    main_module.app.router.lifespan_context = _noop_lifespan

    try:
        with patch.object(settings, "API_KEY", "secret"):
            with TestClient(main_module.app, raise_server_exceptions=True) as c:
                yield c
    finally:
        main_module.app.router.lifespan_context = original_lifespan
        main_module._detector = None
        main_module._rules = None


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_body_has_required_fields(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert "status" in body
        assert "model_loaded" in body
        assert "model_file" in body

    def test_health_status_is_ok(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_with_valid_jpeg_returns_200(self, client: TestClient) -> None:
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_predict_response_schema(self, client: TestClient) -> None:
        body = client.post(
            "/predict",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        ).json()
        assert "image_size" in body
        assert "count" in body
        assert "detections" in body
        assert "inference_ms" in body
        assert body["count"] == 0  # fake detector returns empty
        assert body["detections"] == []

    def test_predict_rejects_non_image_content_type(self, client: TestClient) -> None:
        resp = client.post(
            "/predict",
            files={"file": ("data.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 400

    def test_predict_rejects_pdf_content_type(self, client: TestClient) -> None:
        resp = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert resp.status_code == 400

    def test_predict_returns_400_for_corrupt_image(self, client: TestClient) -> None:
        """Send bytes that look like a JPEG but are actually corrupt."""
        resp = client.post(
            "/predict",
            files={"file": ("bad.jpg", b"\xff\xd8\xff\x00corrupt garbage", "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_predict_rejects_oversize_upload(self, client: TestClient, monkeypatch) -> None:
        """MAX_UPLOAD_MB=0 → max_upload_bytes=0, any upload triggers 413."""
        from app.config import settings
        monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 0)

        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 413

    def test_predict_accepts_png(self, client: TestClient) -> None:
        resp = client.post(
            "/predict",
            files={"file": ("test.png", _make_png_bytes(), "image/png")},
        )
        assert resp.status_code == 200


class TestApiKey:
    def test_missing_key_returns_401(self, client_with_key: TestClient) -> None:
        resp = client_with_key.post(
            "/predict",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, client_with_key: TestClient) -> None:
        resp = client_with_key.post(
            "/predict",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
            headers={"X-API-Key": "wrongkey"},
        )
        assert resp.status_code == 401

    def test_correct_key_returns_200(self, client_with_key: TestClient) -> None:
        resp = client_with_key.post(
            "/predict",
            files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
            headers={"X-API-Key": "secret"},
        )
        assert resp.status_code == 200


# ── integration test (skipped if model absent) ────────────────────────────────

_MODEL_PATH = Path(__file__).parent.parent / "models" / "best.onnx"


@pytest.mark.skipif(
    not _MODEL_PATH.exists(),
    reason="models/best.onnx not present – skipping integration test",
)
class TestIntegration:
    """Runs only when a real ONNX model is available."""

    def test_predict_with_real_model(self, tmp_path) -> None:
        """
        End-to-end: spin up the real app with the real model and
        send a synthetic image. Verify the response schema.
        """
        from app.config import settings
        rules_path = Path(__file__).parent.parent / "config" / "rules.json"

        with patch.object(settings, "MODEL_PATH", str(_MODEL_PATH)), \
             patch.object(settings, "RULES_PATH", str(rules_path)):

            import app.main as main_module

            with TestClient(main_module.app) as c:
                img_bytes = _make_jpeg_bytes(640, 640)
                resp = c.post(
                    "/predict",
                    files={"file": ("test.jpg", img_bytes, "image/jpeg")},
                )
                assert resp.status_code == 200
                body = resp.json()
                assert "count" in body
                assert isinstance(body["detections"], list)


class TestCameraAndRules:
    """Tests for the live camera UI, rules endpoint, and predict_frame base64."""

    def test_camera_ui_returns_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "html" in resp.headers.get("content-type", "").lower()
        assert "TACO" in resp.text

    def test_camera_page_route(self, client: TestClient) -> None:
        resp = client.get("/camera")
        assert resp.status_code == 200
        assert "html" in resp.headers.get("content-type", "").lower()

    def test_rules_endpoint_returns_all_rules(self, client: TestClient) -> None:
        resp = client.get("/rules")
        assert resp.status_code == 200
        data = resp.json()
        assert "classes" in data
        assert "rules" in data
        assert "battery" in data["rules"]
        assert "plastic bottle" in data["rules"]

    def test_predict_frame_base64_returns_200(self, client: TestClient) -> None:
        import base64
        img_bytes = _make_jpeg_bytes(64, 64)
        b64_str = base64.b64encode(img_bytes).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64_str}"

        resp = client.post(
            "/predict_frame",
            json={"image": data_url, "conf_thres": 0.4},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert "detections" in data
        assert "inference_ms" in data

    def test_predict_frame_invalid_base64_returns_400(self, client: TestClient) -> None:
        resp = client.post(
            "/predict_frame",
            json={"image": "not_valid_base64_at_all!!@@#"},
        )
        assert resp.status_code == 400

