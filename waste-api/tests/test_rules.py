"""
tests/test_rules.py
===================
Unit tests for app.rules.RulesEngine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rules import MODEL_CLASSES, RulesEngine

# Load the real complete rules file for robust test fixtures
RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "rules.json"
with RULES_PATH.open("r", encoding="utf-8") as fh:
    FULL_RULES = json.load(fh)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def rules_file(tmp_path):
    """Write a complete rules.json to a temp directory and return its path."""
    p = tmp_path / "rules.json"
    p.write_text(json.dumps(FULL_RULES), encoding="utf-8")
    return p


@pytest.fixture()
def engine(rules_file) -> RulesEngine:
    return RulesEngine(rules_file, MODEL_CLASSES)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestRulesEngineLoading:
    def test_loads_all_classes(self, engine: RulesEngine) -> None:
        assert engine.classes == MODEL_CLASSES

    def test_file_not_found_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            RulesEngine(tmp_path / "missing.json", MODEL_CLASSES)

    def test_missing_class_raises_runtime_error(self, tmp_path) -> None:
        """Omit a required class – engine must raise RuntimeError at load time."""
        target_missing = MODEL_CLASSES[0]
        incomplete = {k: v for k, v in FULL_RULES.items() if k != target_missing}
        p = tmp_path / "incomplete.json"
        p.write_text(json.dumps(incomplete), encoding="utf-8")
        with pytest.raises(RuntimeError, match=target_missing):
            RulesEngine(p, MODEL_CLASSES)

    def test_malformed_entry_raises_value_error(self, tmp_path) -> None:
        """Entry missing 'recyclable' key should raise ValueError."""
        bad = dict(FULL_RULES)
        bad["paper"] = {"biodegradable": True}           # missing recyclable
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="recyclable"):
            RulesEngine(p, MODEL_CLASSES)


class TestRulesEngineMapping:
    def test_paper_is_biodegradable_and_recyclable(self, engine: RulesEngine) -> None:
        assert engine.is_biodegradable("paper") is True
        assert engine.is_recyclable("paper") is True

    def test_organic_is_not_recyclable(self, engine: RulesEngine) -> None:
        assert engine.is_biodegradable("organic") is True
        assert engine.is_recyclable("organic") is False

    def test_glass_is_not_biodegradable(self, engine: RulesEngine) -> None:
        assert engine.is_biodegradable("glass") is False
        assert engine.is_recyclable("glass") is True

    def test_ewaste_neither(self, engine: RulesEngine) -> None:
        assert engine.is_biodegradable("ewaste") is False
        assert engine.is_recyclable("ewaste") is False

    def test_unknown_material_returns_conservative_defaults(self, engine: RulesEngine) -> None:
        rule = engine.get("unknown_material")
        assert rule["biodegradable"] is False
        assert rule["recyclable"] is False

    @pytest.mark.parametrize("material", MODEL_CLASSES)
    def test_all_classes_resolve(self, engine: RulesEngine, material: str) -> None:
        rule = engine.get(material)
        assert isinstance(rule["biodegradable"], bool)
        assert isinstance(rule["recyclable"], bool)
