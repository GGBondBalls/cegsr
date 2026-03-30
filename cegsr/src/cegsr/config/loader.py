from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from cegsr.utils.io import read_yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two config dictionaries."""
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config and optionally apply KEY=VALUE overrides."""
    config = read_yaml(path) or {}
    config = _expand_env(config)
    overrides = overrides or []
    for item in overrides:
        key, raw_value = item.split("=", 1)
        target = config
        parts = key.split(".")
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        if raw_value.lower() in {"true", "false"}:
            parsed: Any = raw_value.lower() == "true"
        else:
            try:
                parsed = int(raw_value)
            except ValueError:
                try:
                    parsed = float(raw_value)
                except ValueError:
                    parsed = raw_value
        target[parts[-1]] = parsed
    return config
