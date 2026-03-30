from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

JSONValue = Any
JSONDict = Dict[str, JSONValue]
Messages = List[Dict[str, str]]
ConfigDict = Dict[str, Any]

__all__ = ["JSONValue", "JSONDict", "Messages", "ConfigDict"]
