from __future__ import annotations

from pathlib import Path
from typing import Any

from cegsr.utils.io import read_json, write_json


def register_checkpoint(registry_path: str, alias: str, path: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    registry_file = Path(registry_path)
    if registry_file.exists():
        registry = read_json(registry_file)
    else:
        registry = {}
    registry[alias] = {"path": path, "metadata": metadata or {}}
    write_json(registry_file, registry)
    return registry


def resolve_checkpoint(registry_path: str, alias: str) -> str | None:
    registry_file = Path(registry_path)
    if not registry_file.exists():
        return None
    registry = read_json(registry_file)
    item = registry.get(alias)
    return item.get("path") if item else None
