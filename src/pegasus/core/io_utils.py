"""Core I/O and JSON utilities."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping


def _compact(value: Any) -> str:
    """Serialize a value to a compact JSON string."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _load_json(path: Path | str) -> dict[str, Any]:
    """Load JSON from a file. Returns an empty dict if it fails or doesn't exist."""
    path_obj = Path(path)
    if not path_obj.exists():
        return {}
    try:
        payload = json.loads(path_obj.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path | str, payload: Mapping[str, Any] | list[Any]) -> Path:
    """Atomically write JSON payload to a file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    tmp_path = path_obj.with_suffix(f".tmp.{uuid.uuid4().hex[:8]}")
    text = json.dumps(
        dict(payload) if isinstance(payload, Mapping) else payload, 
        ensure_ascii=False, 
        sort_keys=True, 
        indent=2, 
        default=str
    )
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path_obj)
    return path_obj


def _hash_payload(payload: Mapping[str, Any] | list[Any] | str) -> str:
    """Generate SHA-256 hex digest for a JSON-serializable payload."""
    if isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        val = dict(payload) if isinstance(payload, Mapping) else payload
        raw = json.dumps(val, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _safe_id(value: Any) -> str:
    """Convert an arbitrary value to a filesystem-safe identifier."""
    raw = str(value or "field").lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")
    return out or "field"
