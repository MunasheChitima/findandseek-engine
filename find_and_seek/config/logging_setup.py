"""Configure rotating file logging to ~/.findandseek/logs/findandseek.log.

Call configure() once at process startup (API server, worker). All
logging.* calls in the codebase will then write to the file as well as
stderr. The file uses a fixed, human-readable format:

    2026-06-12T14:30:01 INFO  find_and_seek.config.settings | role_backend(summary) = afm
    2026-06-12T14:30:02 WARN  find_and_seek.ingest.worker   | [worker] memory pressure...

Privacy: the format is deliberately fixed. Callers are responsible for
not passing paths, filenames, or document content to logging.*() — the
same rule as diagnostics.py. This module never sanitises post-hoc; the
discipline lives at the call sites, not in a regexp here.

Rotation: 5 MB × 4 files = up to 20 MB on disk. Old logs are not sent
anywhere; they are user-accessible under ~/.findandseek/logs/.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


LOG_DIR = Path.home() / ".findandseek" / "logs"
LOG_FILE = LOG_DIR / "findandseek.log"

_FORMAT = "%(asctime)s %(levelname)-5s %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%dT%H:%M:%S"

_configured = False


def configure(level: int = logging.INFO) -> None:
    """Attach a RotatingFileHandler to the root logger.

    Idempotent — safe to call from both the API process and worker processes;
    only the first call installs the handler. Each process writes its own
    rotation sequence (they share the log dir but not the file handle, which
    is fine because launchd runs them as separate PIDs and RotatingFileHandler
    uses rename-based rotation that is atomic on macOS).
    """
    global _configured
    if _configured:
        return
    _configured = True

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,   # 5 MB per file
        backupCount=3,               # keep findandseek.log + .1 .2 .3
        encoding="utf-8",
        delay=False,
    )
    handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FMT))

    root = logging.getLogger()
    # Don't double-add if somehow called twice across import cycles.
    already = any(
        isinstance(h, logging.handlers.RotatingFileHandler)
        and getattr(h, "baseFilename", None) == str(LOG_FILE)
        for h in root.handlers
    )
    if not already:
        root.addHandler(handler)

    # Raise the root level if it is currently unset or coarser than requested.
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
