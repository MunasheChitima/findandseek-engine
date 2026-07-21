"""Organize settings, persisted to ~/.findandseek/organize.json.

Currently the opt-in Finder-tag sync flag (off by default — design open-question
#2). Override the path with FINDANDSEEK_ORGANIZE_SETTINGS (tests redirect it).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "finder_sync": False,
    "synced_kinds": ["type", "year", "party", "status"],
}


def _path() -> Path:
    override = os.environ.get("FINDANDSEEK_ORGANIZE_SETTINGS")
    return Path(override) if override else Path.home() / ".findandseek" / "organize.json"


def get_settings() -> dict[str, Any]:
    p = _path()
    data = dict(DEFAULTS)
    if p.exists():
        try:
            data.update(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 — corrupt file → defaults
            pass
    return data


def update_settings(**changes: Any) -> dict[str, Any]:
    data = get_settings()
    data.update({k: v for k, v in changes.items() if v is not None})
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
