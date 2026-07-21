"""Runtime settings — ports, hosts."""

from __future__ import annotations

import os
import sys

import psutil

IS_MAC = sys.platform == "darwin"

# ── Hardware-aware RAM tier (lets the app run on 8 GB Macs) ──────────
# Detected once at import. Drives a smaller summary model, smaller ingest
# batches, a lower memory-pause threshold, and OCR-off-by-default on low-RAM
# machines. Everything below is still overridable via the matching env var.
TOTAL_RAM_GB: float = psutil.virtual_memory().total / (1024 ** 3)
# 8 GB Macs report ~8.0; a little slack covers binary-vs-marketing GB.
LOW_RAM: bool = TOTAL_RAM_GB <= 9.0


def default_min_free_gb(total_gb: float | None = None) -> float:
    """Memory-pause threshold scaled to installed RAM. A flat 5 GB would block
    ingest *forever* on an 8 GB Mac (it never has 5 GB free)."""
    gb = TOTAL_RAM_GB if total_gb is None else total_gb
    if gb <= 9.0:
        return 1.5
    if gb <= 17.0:
        return 2.5
    return 5.0


def default_batch_size(total_gb: float | None = None) -> int:
    """Smaller batches bound transient memory on low-RAM machines."""
    gb = TOTAL_RAM_GB if total_gb is None else total_gb
    return 4 if gb <= 9.0 else 16


def ocr_enabled_default(low_ram: bool | None = None) -> bool:
    """OCR loads the ~2.9 GB vision model — off by default on 8 GB (opt-in via
    FINDANDSEEK_ENABLE_OCR), on everywhere else."""
    return not (LOW_RAM if low_ram is None else low_ram)

API_HOST = os.environ.get("FINDANDSEEK_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("FINDANDSEEK_API_PORT", "8775"))

MCP_HOST = os.environ.get("FINDANDSEEK_MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.environ.get("FINDANDSEEK_MCP_PORT", "8776"))

# ── Inference backend selection ──────────────────────────────────────
# "mlx" (default, the only production backend: in-process Apple-Silicon inference,
# no external daemon). Per-role overrides (FINDANDSEEK_EMBED_BACKEND / _SUMMARY_BACKEND
# / _VISION_BACKEND) take precedence over the global FINDANDSEEK_BACKEND. In test
# mode every role uses the lightweight degraded path (pseudo embeddings, heuristic
# summaries, no vision) so the suite is fast and hermetic. Any unrecognised value
# is treated as MLX, so a stray override degrades cleanly rather than crashing.
# NB: resolved live (not cached at import) so pytest's per-test monkeypatch of
# FINDANDSEEK_TEST / FINDANDSEEK_BACKEND takes effect regardless of import order.
def _afm_available() -> bool:
    """True if Apple Foundation Models are accessible on this machine.

    Requires macOS 26+, Apple Intelligence enabled, and apple-fm-sdk installed.
    Result is cached after the first call — availability doesn't change mid-process.
    """
    if not hasattr(_afm_available, "_cache"):
        try:
            import apple_fm_sdk as fm  # noqa: PLC0415
            ok, _ = fm.SystemLanguageModel().is_available()
            _afm_available._cache = bool(ok)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            _afm_available._cache = False  # type: ignore[attr-defined]
    return _afm_available._cache  # type: ignore[attr-defined]


def role_backend(role: str) -> str:
    """Resolve the backend for a role: 'embed' | 'summary' | 'vision'."""
    override = os.environ.get(f"FINDANDSEEK_{role.upper()}_BACKEND")
    if os.environ.get("FINDANDSEEK_TEST") == "1" and not override:
        return "test"
    if role == "vision":
        # macOS Vision OCR is the ONLY production OCR path on Macs (owner
        # decision 2026-06-11): measured ~60× faster than the VLM per scanned
        # page AND recovers more text verbatim — there is no case where the VLM
        # wins on OCR. Not overridable there; a missing ocrtool helper fails
        # loudly in the dispatcher rather than degrading to a slower, worse
        # engine. Off-Mac there is no Vision framework, so OCR goes through
        # tesseract (overridable via FINDANDSEEK_VISION_BACKEND).
        if os.environ.get("FINDANDSEEK_TEST") == "1":
            return "test"
        if IS_MAC:
            return "macos"
        return (override or "tesseract").lower()
    if override:
        return override.lower()
    # MLX Qwen3 IS the production summary backend on every Mac (owner decision
    # 2026-07-17: AFM dropped from the summary path — slower, fallback-prone,
    # and Apple swaps its weights under us on OS updates). RAM tiering picks
    # 4B vs 1.7B in config.models; a machine that can't load either degrades
    # to the marked heuristic card, never silently.
    # Off-Mac there is no MLX; the default is a local Ollama daemon serving the
    # same model family (qwen3:4b / embeddinggemma — see the *_ollama modules).
    # Force another backend with: FINDANDSEEK_SUMMARY_BACKEND=<name>
    if role == "summary":
        return "mlx" if IS_MAC else "ollama"
    return os.environ.get("FINDANDSEEK_BACKEND", "mlx" if IS_MAC else "ollama").lower()

# Summariser input cap (chars). The summariser is prefill-bound — its cost is
# dominated by processing this input, not generating output — so this cap is the
# single biggest ingest-speed lever (summarisation is ~88% of ingest time).
# Measured on the eval corpus: 8000→3000 chars is ~1.9x faster with no loss in
# document_type / anchor / key_facts quality (they live in the document head).
# Raise for richer prose summaries; lower (e.g. 1500 ≈ 2.5x) if eval holds.
SUMMARY_MAX_CHARS = int(os.environ.get("FINDANDSEEK_SUMMARY_MAX_CHARS", "3000"))

# ── Ingest safety: max per-file size for extraction ──────────────────
# Extraction loads the whole file into RAM (e.g. text._read_text does
# path.read_bytes()), and the memory guard only checks BETWEEN batches — so a
# single multi-GB file (a log, DB dump, or export sitting in a watched folder)
# can OOM the worker *mid-parse*, before any guard fires. Files larger than this
# cap are skipped gracefully (left out of the index, not dead-lettered — they're
# not poison). sha256 streaming is unaffected; only extraction is guarded.
# Raise via FINDANDSEEK_MAX_FILE_MB if you index large corpora on a big-RAM Mac.
MAX_FILE_MB = int(os.environ.get("FINDANDSEEK_MAX_FILE_MB", "256"))

# ── Power-aware ingest ───────────────────────────────────────────────
# SUPERSEDED by the smooth/full performance mode in config/power_settings.py,
# which the worker reads directly (the user-facing "Full power" toggle). This
# legacy knob is retained only so an existing FINDANDSEEK_PAUSE_ON_BATTERY in
# someone's environment doesn't error; it no longer drives the worker.
_pause_on_battery = os.environ.get("FINDANDSEEK_PAUSE_ON_BATTERY", "throttle").lower()
PAUSE_ON_BATTERY = _pause_on_battery if _pause_on_battery in ("off", "throttle", "pause") else "throttle"

# ── Search / ranking tunables (override via env without code edits) ──
RRF_K = int(os.environ.get("FINDANDSEEK_RRF_K", "60"))
VEC_WEIGHT = float(os.environ.get("FINDANDSEEK_VEC_WEIGHT", "0.7"))
KW_WEIGHT = float(os.environ.get("FINDANDSEEK_KW_WEIGHT", "0.3"))
SUMMARY_BOOST = float(os.environ.get("FINDANDSEEK_SUMMARY_BOOST", "0.5"))
FILENAME_BOOST = float(os.environ.get("FINDANDSEEK_FILENAME_BOOST", "0.15"))
MAX_CHUNKS_PER_FILE = int(os.environ.get("FINDANDSEEK_MAX_CHUNKS_PER_FILE", "3"))
