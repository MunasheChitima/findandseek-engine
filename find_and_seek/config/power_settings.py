"""User-facing power/performance preference, persisted to ~/.findandseek/power.json.

The ingest worker is sustained heavy CPU/GPU work. Run flat-out and it spins the
fans up ("jet engine") and — on laptops with a smaller adapter — can draw more
watts than the charger supplies, so the battery *discharges even while plugged
in*. That is never the intent.

So the product has exactly two modes, a single box the user ticks:

  • "smooth"  (default) — be a good citizen. On battery: don't index (protect the
    battery). On AC: index *gently* — small batches with a cool-down between them
    so average draw stays low and the fans stay quiet, and yield to the user when
    the machine is actively busy. Indexing still completes; it just sips power.

  • "full"    (user ticked "Full power") — blitz. Maximise throughput for the
    machine's tier, ignore power state. The user explicitly opted into the heat
    and the battery cost.

This file is the single source of truth, hot-read by the worker each batch (so a
toggle in the app takes effect on the next cycle, no restart). ``FINDANDSEEK_
PERFORMANCE_MODE`` overrides the file for power users / tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

MODES = ("smooth", "full")
DEFAULT_MODE = "smooth"


def _path() -> Path:
    override = os.environ.get("FINDANDSEEK_POWER_SETTINGS")
    return Path(override) if override else Path.home() / ".findandseek" / "power.json"


def get_mode() -> str:
    """Resolve the active performance mode. Env var wins, then the JSON file, then
    the smooth default. Anything unrecognised falls back to smooth (the safe one)."""
    env = os.environ.get("FINDANDSEEK_PERFORMANCE_MODE")
    if env:
        env = env.strip().lower()
        return env if env in MODES else DEFAULT_MODE
    p = _path()
    if p.exists():
        try:
            mode = str(json.loads(p.read_text(encoding="utf-8")).get("mode", "")).lower()
            if mode in MODES:
                return mode
        except Exception:  # noqa: BLE001 — corrupt/partial file → safe default
            pass
    return DEFAULT_MODE


def get_settings() -> dict[str, Any]:
    return {"mode": get_mode()}


def set_mode(mode: str) -> dict[str, Any]:
    """Persist the performance mode. Unknown values clamp to the smooth default."""
    mode = (mode or "").strip().lower()
    if mode not in MODES:
        mode = DEFAULT_MODE
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"mode": mode}, indent=2), encoding="utf-8")
    return {"mode": mode}
