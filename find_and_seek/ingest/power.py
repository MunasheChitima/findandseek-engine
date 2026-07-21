"""Power-state helpers for power-aware ingest (D45: battery drain under backlog).

A deep ingest backlog is sustained heavy CPU/GPU work. Run unconditionally on
battery and it drains the machine; in Low-Power Mode it also fights the OS's own
throttling. These helpers let the worker back off on battery / Low-Power and run
full speed on AC.

Both results are cached for a few seconds — power state changes on the order of
seconds-to-minutes, not per batch, so polling every batch is wasteful (and the
`pmset` shell-out in particular shouldn't run in a tight loop).
"""

from __future__ import annotations

import subprocess
import time

import psutil

# Cache window: long enough to avoid per-batch polling, short enough that an
# unplug/plug or Low-Power toggle is noticed within a few seconds.
_CACHE_TTL = 10.0

_ac_cache: tuple[float, bool] | None = None
_lpm_cache: tuple[float, bool] | None = None


def on_ac_power() -> bool:
    """True if the machine is on AC (or has no battery — desktop Macs).

    ``psutil.sensors_battery()`` returns None on hardware without a battery; we
    treat that as AC so ingest runs full speed there. Cached for a few seconds.
    """
    global _ac_cache
    now = time.monotonic()
    if _ac_cache is not None and now - _ac_cache[0] < _CACHE_TTL:
        return _ac_cache[1]
    try:
        batt = psutil.sensors_battery()
        # None ⇒ no battery (desktop) ⇒ treat as plugged in.
        on_ac = True if batt is None else bool(batt.power_plugged)
    except Exception:  # noqa: BLE001 — sensor unavailable ⇒ assume AC (don't stall ingest)
        on_ac = True
    _ac_cache = (now, on_ac)
    return on_ac


def low_power_mode() -> bool:
    """Best-effort: True if macOS Low-Power Mode is on. False on any error.

    Parsed from ``pmset -g`` (`lowpowermode 1`). This is a soft signal — we never
    hard-depend on it, so a missing/failed shell-out just reports False (no LPM).
    Cached for a few seconds; the subprocess has a short timeout.
    """
    global _lpm_cache
    now = time.monotonic()
    if _lpm_cache is not None and now - _lpm_cache[0] < _CACHE_TTL:
        return _lpm_cache[1]
    lpm = False
    try:
        out = subprocess.run(
            ["pmset", "-g"],
            capture_output=True, text=True, timeout=2,
        ).stdout
        for line in out.splitlines():
            if "lowpowermode" in line:
                # e.g. " lowpowermode         1"
                lpm = line.split()[-1].strip() == "1"
                break
    except Exception:  # noqa: BLE001 — pmset missing/slow/odd output ⇒ assume off
        lpm = False
    _lpm_cache = (now, lpm)
    return lpm
