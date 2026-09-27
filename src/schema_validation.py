"""Validate prediction features against the model schema."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real
from typing import Any


def validate_features(
    features: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> list[str]:
    """Return schema violations for one feature dictionary."""
    definitions = {
        definition["name"]: definition
        for definition in schema.get("features", [])
    }
    errors = []

    unknown = sorted(set(features) - set(definitions))
    if unknown:
        errors.append(f"Unknown feature(s): {', '.join(unknown)}")

    for name, definition in definitions.items():
        if name not in features:
            if definition.get("required", False):
                errors.append(f"Missing required feature: {name}")
            continue

        value = features[name]
        if value is None:
            if definition.get("required", False):
                errors.append(f"Feature {name} cannot be null")
            continue

        if definition["type"] == "number":
            if isinstance(value, bool) or not isinstance(value, Real):
                errors.append(f"Feature {name} must be numeric")
                continue
            if not isfinite(float(value)):
                errors.append(f"Feature {name} must be finite")
                continue
            if value < definition["min"] or value > definition["max"]:
                errors.append(
                    f"Feature {name} must be between "
                    f"{definition['min']} and {definition['max']}"
                )
        elif not isinstance(value, str):
            errors.append(f"Feature {name} must be a string")
        elif definition.get("values") and value not in definition["values"]:
            errors.append(
                f"Feature {name} must be one of: "
                f"{', '.join(definition['values'])}"
            )

    return errors