"""
app/rules.py
============
Loads config/rules.json and maps each detected material to its
biodegradability and recyclability properties, disposal bin recommendation,
and handling instructions.

The module validates on load that every class known to the model has an
entry in the rules file; if not it raises a RuntimeError so the process
fails at startup with a clear message.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)


class MaterialRule(TypedDict, total=False):
    biodegradable: bool
    recyclable: bool
    category: str
    bin: str
    instructions: str


# ── Canonical Class Definitions ───────────────────────────────────────────────

# TACO (Trash Annotations in Context) / Roboflow Trash Detections dataset classes
# matches the trained YOLOv8 model weights in models/best.onnx (10 classes):
TACO_CLASSES: list[str] = [
    "battery",
    "can",
    "cardboard",
    "drink carton",
    "glass bottle",
    "paper",
    "plastic bag",
    "plastic bottle",
    "plastic bottle cap",
    "pop tab",
]

# Legacy generic 8-class fallback list:
GENERIC_CLASSES: list[str] = [
    "paper",
    "cardboard",
    "organic",
    "glass",
    "metal",
    "plastic",
    "textile",
    "ewaste",
]

# Default classes to use when model metadata is absent
MODEL_CLASSES: list[str] = TACO_CLASSES


class RulesEngine:
    """Loads rules.json and answers property queries for a given material."""

    def __init__(self, rules_path: str | Path, model_classes: list[str] | None = None) -> None:
        self._rules: dict[str, MaterialRule] = {}
        self._classes = model_classes if model_classes is not None else MODEL_CLASSES
        self._load(Path(rules_path))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Rules file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw: dict[str, dict[str, object]] = json.load(fh)

        # Structural validation & ingestion
        for material, props in raw.items():
            if "biodegradable" not in props or "recyclable" not in props:
                raise ValueError(
                    f"rules.json entry for '{material}' is missing "
                    "'biodegradable' and/or 'recyclable' keys."
                )
            self._rules[material] = MaterialRule(
                biodegradable=bool(props["biodegradable"]),
                recyclable=bool(props["recyclable"]),
                category=str(props.get("category", material.title())),
                bin=str(props.get("bin", "General Recycling / Waste Bin")),
                instructions=str(props.get("instructions", "Dispose responsibly according to local guidelines.")),
            )

        # Completeness check: every active model class must appear in the rules
        missing = [cls for cls in self._classes if cls not in self._rules]
        if missing:
            raise RuntimeError(
                f"rules.json is missing entries for model classes: {missing}. "
                "Add them to config/rules.json before starting the server."
            )

        logger.info("Loaded %d material rules from %s", len(self._rules), path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, material: str) -> MaterialRule:
        """Return the rule for *material*, or a safe default if unknown."""
        key = material.lower().strip()
        if key in self._rules:
            return self._rules[key]

        # Fuzzy alias matching for common item names
        if "bottle" in key:
            candidate = self._rules.get("plastic bottle") or self._rules.get("bottle")
            if candidate:
                return candidate
        if "carton" in key or "tetra" in key:
            candidate = self._rules.get("drink carton") or self._rules.get("cardboard")
            if candidate:
                return candidate
        if "bag" in key or "wrapper" in key or "film" in key:
            candidate = self._rules.get("plastic bag") or self._rules.get("plastic")
            if candidate:
                return candidate
        if "phone" in key or "battery" in key or "laptop" in key or "electronic" in key:
            candidate = self._rules.get("battery") or self._rules.get("ewaste")
            if candidate:
                return candidate
        if "box" in key or "cardboard" in key:
            candidate = self._rules.get("cardboard")
            if candidate:
                return candidate
        if "can" in key or "tin" in key:
            candidate = self._rules.get("can") or self._rules.get("metal")
            if candidate:
                return candidate
        if "paper" in key or "book" in key:
            candidate = self._rules.get("paper")
            if candidate:
                return candidate

        logger.warning("Unknown material '%s'; returning conservative defaults.", material)
        return MaterialRule(
            biodegradable=False,
            recyclable=False,
            category="General Waste",
            bin="General Waste Bin",
            instructions="Dispose in general waste bin or consult local council guidelines.",
        )

    def is_biodegradable(self, material: str) -> bool:
        return self.get(material)["biodegradable"]

    def is_recyclable(self, material: str) -> bool:
        return self.get(material)["recyclable"]

    @property
    def classes(self) -> list[str]:
        """Ordered list of model class names."""
        return list(self._classes)

    @property
    def all_rules(self) -> dict[str, MaterialRule]:
        """Copy of all loaded rules."""
        return dict(self._rules)
