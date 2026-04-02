from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _is_model_dir(path: Path) -> bool:
    return (path / 'config.json').exists() and ((path / 'tokenizer_config.json').exists() or (path / 'tokenizer.json').exists())


def _iter_candidate_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    candidates: list[Path] = []
    if _is_model_dir(root):
        candidates.append(root)
    snapshots = root / 'snapshots'
    if snapshots.exists():
        for child in sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if child.is_dir() and _is_model_dir(child):
                candidates.append(child)
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []:
        if child.is_dir() and _is_model_dir(child):
            candidates.append(child)
    seen: set[str] = set()
    uniq: list[Path] = []
    for item in candidates:
        key = str(item.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(item)
    return uniq


def render_model_path_template(model_name_or_path: str, model_size_hint: str | None = None) -> str:
    raw = os.path.expandvars(model_name_or_path)
    size = model_size_hint or os.environ.get('CEGSR_MODEL_SIZE', '7B')
    if 'X.XB' in raw:
        raw = raw.replace('X.XB', size)
    if '{model_size}' in raw:
        raw = raw.format(model_size=size)
    return raw


def resolve_local_model_path(model_name_or_path: str, model_size_hint: str | None = None) -> str:
    """
    Resolve a local HuggingFace model path robustly.

    Supports:
    1) a direct model directory
    2) a HuggingFace cache repo root (it will descend into snapshots/*)
    3) a template path containing ``X.XB`` or ``{model_size}``
    4) a non-local identifier such as ``Qwen/Qwen2.5-7B-Instruct`` (returned unchanged)
    """
    raw = render_model_path_template(model_name_or_path, model_size_hint=model_size_hint)
    size = model_size_hint or os.environ.get('CEGSR_MODEL_SIZE', '7B')

    path = Path(raw)
    if not path.exists():
        return raw

    candidates = list(_iter_candidate_dirs(path))
    if candidates:
        return str(candidates[0])

    if path.name.startswith('models--'):
        parent = path.parent
        pattern = path.name.replace(size, '*') if size and size in path.name else path.name
        matches = sorted(parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        for match in matches:
            nested = list(_iter_candidate_dirs(match))
            if nested:
                return str(nested[0])

    return str(path)
