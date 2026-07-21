"""Local-only diagnostics — the minimum operational data to debug a user issue.

Design rule (data minimisation): capture exactly what makes a problem
diagnosable and **nothing more**. Specifically we record:
  • hardware/tier (RAM, chip, macOS), library + app versions,
  • which models are configured + whether OCR is on,
  • index counts (how many files/queue), DB size,
  • operational events: ingest throughput, memory-guard pauses, error *types*.

We deliberately do NOT record: document contents, file names or paths, search
queries, or any personal data. Error events store the exception *type* and the
pipeline stage only — never the message (messages can embed paths). Everything
stays on disk under ~/.findandseek/logs; nothing is ever sent anywhere. The user
exports it explicitly (an in-app button) when they want to share it.
"""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import time
from pathlib import Path
from typing import Any

LOG_DIR = Path.home() / ".findandseek" / "logs"
METRICS_PATH = LOG_DIR / "metrics.jsonl"
_MAX_BYTES = 2 * 1024 * 1024   # cap the rolling metrics log at ~2 MB
_KEEP_LINES = 1000             # trim to the most recent N lines when over cap

# Only these keys are ever written in an event payload — a hard allow-list so a
# careless caller can't leak a path or content into the diagnostics log.
_ALLOWED_FIELDS = {
    "files", "seconds", "free_gb", "threshold", "model", "load_seconds",
    "stage", "error_type", "ext", "tier", "count",
    # Power-aware ingest (battery/AC/Low-Power) and skip-reason events. The tray
    # reads these to surface why ingest paused/throttled. All are safe scalars —
    # never a path or content (skip-reason events carry only a reason + ext).
    "reason", "power_mode", "on_ac", "low_power", "size_mb",
}


def record(kind: str, **fields: Any) -> None:
    """Append one operational event. Non-allow-listed or non-scalar fields are
    dropped, so only safe, structured signal is ever persisted."""
    safe = {
        k: v for k, v in fields.items()
        if k in _ALLOWED_FIELDS and isinstance(v, (int, float, str, bool))
    }
    event = {"ts": round(time.time(), 1), "kind": str(kind)[:40], **safe}
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with METRICS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        _rotate_if_needed()
    except OSError:
        pass  # diagnostics must never break ingest


def record_error(stage: str, exc: BaseException, ext: str | None = None) -> None:
    """Record an error as type + stage only (never the message — it can hold a
    path). `ext` is the file extension, which is safe and useful."""
    record("error", stage=stage, error_type=type(exc).__name__,
           ext=(ext or "")[:12])


def _rotate_if_needed() -> None:
    try:
        if METRICS_PATH.stat().st_size <= _MAX_BYTES:
            return
        lines = METRICS_PATH.read_text(encoding="utf-8").splitlines()[-_KEEP_LINES:]
        METRICS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return "unknown"


def _app_version() -> str:
    return _pkg_version("find-and-seek")


def system_snapshot() -> dict[str, Any]:
    """Hardware, OS, versions, and the active model/tier config. No user data."""
    import psutil

    from find_and_seek.config import settings
    from find_and_seek.config.models import MLX_MODELS, mlx_summary_candidates

    vm = psutil.virtual_memory()
    snap: dict[str, Any] = {
        "app_version": _app_version(),
        "os": f"macOS {platform.mac_ver()[0]}" if platform.system() == "Darwin" else platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(vm.total / 1e9, 1),
        "ram_available_gb": round(vm.available / 1e9, 1),
        "low_ram_tier": settings.LOW_RAM,
        "min_free_gb": settings.default_min_free_gb(),
        "batch_size": settings.default_batch_size(),
        "ocr_enabled": settings.ocr_enabled_default(),
        "summary_model": mlx_summary_candidates()[0],
        "embed_model": MLX_MODELS["embed_model"],
        "vision_model": MLX_MODELS["vision_model"],
        "packages": {p: _pkg_version(p) for p in
                     ("mlx", "mlx-lm", "mlx-vlm", "numpy", "onnxruntime",
                      "huggingface-hub", "fastapi")},
    }
    return snap


def runtime_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Index counts + DB size. Counts only — no file identities."""
    def _count(sql: str) -> int:
        try:
            return int(conn.execute(sql).fetchone()[0])
        except sqlite3.Error:
            return -1

    db_bytes = -1
    try:
        for _, _, path in conn.execute("PRAGMA database_list").fetchall():
            if path and Path(path).exists():
                db_bytes = Path(path).stat().st_size
                break
    except (sqlite3.Error, OSError):
        pass

    return {
        "files_indexed": _count("SELECT COUNT(*) FROM files WHERE status='indexed'"),
        "files_failed": _count("SELECT COUNT(*) FROM files WHERE status='failed'"),
        "queue_pending": _count("SELECT COUNT(*) FROM ingest_queue WHERE status='pending'"),
        "queue_failed": _count("SELECT COUNT(*) FROM ingest_queue WHERE status='failed'"),
        "db_size_mb": round(db_bytes / 1e6, 1) if db_bytes >= 0 else -1,
    }


def recent_events(limit: int = 300) -> list[dict[str, Any]]:
    """The tail of the structured metrics log (already path/content-free)."""
    try:
        lines = METRICS_PATH.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def collect(conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    """The full export payload: snapshot + counts + recent operational events.
    Built only from curated, path-free sources — never raw log files."""
    stats: dict[str, Any] = {}
    if conn is not None:
        stats = runtime_stats(conn)
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "schema": "find-and-seek-diagnostics/1",
        "system": system_snapshot(),
        "runtime": stats,
        "events": recent_events(),
    }


def export(conn: sqlite3.Connection | None, dest_dir: Path | str | None = None) -> Path:
    """Write a diagnostics bundle (zip) to dest_dir and return its path.

    Bundle contents:
      diagnostics.json  — structured payload (system + counts + events)
      findandseek.log   — rotating sidecar log (last ~5 MB, privacy-safe)
      app-boot.log      — Swift app boot / sidecar-launch trace
      metrics.jsonl     — structured operational events (same as events in JSON)

    All sources are already path/content-free by construction; the bundle is
    safe to share as-is when reporting a bug.
    """
    import zipfile

    dest = Path(dest_dir) if dest_dir else (Path.home() / "Desktop")
    dest.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = dest / f"FindAndSeek-diagnostics-{stamp}.zip"

    payload = json.dumps(collect(conn), indent=2).encode("utf-8")

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"diagnostics-{stamp}/diagnostics.json", payload)
        for fname in ("findandseek.log", "app-boot.log", "metrics.jsonl"):
            src = LOG_DIR / fname
            if src.exists():
                zf.write(src, arcname=f"diagnostics-{stamp}/{fname}")

    return out
