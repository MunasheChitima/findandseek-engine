"""Hardware → model tier detection."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import psutil

from find_and_seek.config.models import resolve_models

PROFILES: dict[str, dict[str, Any]] = {
    # Per-tier ingest behaviour. The actual model ids are MLX repos supplied by
    # resolve_models() (same across tiers; the summary-size split lives in
    # config/models.mlx_summary_candidates). Only the ingest pacing differs here:
    # low-RAM machines ingest idle-only so heavy models are freed between batches.
    "low": {
        "ingest_mode": "idle_only",
        "max_resident_heavy_models": 1,
    },
    "mid": {
        "ingest_mode": "background",
        "max_resident_heavy_models": 1,
    },
    "mid_gpu": {
        "ingest_mode": "background",
        "max_resident_heavy_models": 1,
    },
    "high": {
        "ingest_mode": "background",
        "max_resident_heavy_models": 1,
    },
}

PROFILE_PATH = Path.home() / ".findandseek" / "profile.json"
import os as _os
EMBED_DIM = int(_os.environ.get("FINDANDSEEK_EMBED_DIM", "768"))


def _has_metal() -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return "Apple" in out.stdout or "Metal" in out.stdout
    except (OSError, subprocess.TimeoutExpired):
        return platform.machine() == "arm64"


def detect_profile() -> str:
    ram_gb = psutil.virtual_memory().total / (1024**3)
    metal = _has_metal()
    if ram_gb >= 28:
        return "high"
    if ram_gb >= 14:
        return "mid_gpu" if metal else "mid"
    return "low"


def get_profile(force_detect: bool = False) -> dict[str, Any]:
    import os

    if os.environ.get("FINDANDSEEK_TEST") == "1":
        base = {"name": "mid", **PROFILES["mid"]}
        return {**base, **resolve_models("mid")}

    if PROFILE_PATH.exists() and not force_detect:
        data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        name = data.get("name", "mid")
        base = {"name": name, **PROFILES.get(name, PROFILES["mid"])}
        return {**base, **resolve_models(name)}

    name = detect_profile()
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps({"name": name}), encoding="utf-8")
    base = {"name": name, **PROFILES[name]}
    return {**base, **resolve_models(name)}
