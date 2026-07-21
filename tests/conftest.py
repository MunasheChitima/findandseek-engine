"""Shared pytest config for the engine suite.

Force ``FINDANDSEEK_PERFORMANCE_MODE=full`` for the whole engine test run.

Why this exists: ``worker.process_batch`` is power-aware (D45). In the default
"smooth" mode ``_pacing()`` *pauses* a batch — returning 0 without processing
anything — when the machine is on battery, in Low-Power Mode, or simply busy
(``machine_is_busy()``). Tests like ``test_phases.TestPhase2.test_index_and_search``
drive a synchronous drain loop that only exits once the queue is empty:

    while True:
        n = process_batch(conn)
        if n == 0 and pending == 0:
            break

If the power gate pauses, ``process_batch`` keeps returning 0 while items stay
``pending`` → the loop never terminates → the batch just ``time.sleep(5)``s
forever at 0% CPU. That is host-state-dependent: it passes on an idle machine on
AC and hangs the moment the machine is busy (locally or on a loaded CI runner) —
exactly the intermittent full-suite hang we chased down via a thread dump.

The power gate is a production feature, not something a deterministic ingest test
should be subject to. ``power_settings`` documents ``FINDANDSEEK_PERFORMANCE_MODE``
as the override "for power users / tests"; forcing "full" makes ``_pacing()``
never pause, so ingest tests are deterministic regardless of battery/AC/CPU load.

Tests that specifically exercise the power settings (``test_power_settings.py``)
use ``monkeypatch`` to set/delete this var and restore it afterwards, so they are
unaffected by this default.
"""

from __future__ import annotations

import os
from pathlib import Path

# Set before any test module imports the worker so every batch reads "full".
os.environ["FINDANDSEEK_PERFORMANCE_MODE"] = "full"


# ── the test corpus is generated, never committed ─────────────────
# tests/corpus holds binary fixtures (PDF, docx, xlsx, pptx) that
# generate_corpus.py reproduces in ~0.3s from libraries that are already project
# dependencies. They are gitignored, for one reason that matters more than the
# ~87 KB: a committed artifact can drift from its generator. That is not
# hypothetical here — the generator was changed to bill a fictional customer,
# the PDFs were never rebuilt, and a real person's name survived inside a binary
# fixture where no text search could see it. Artifacts that are never committed
# cannot drift.
#
# Generated at import time rather than in a fixture: test modules read the
# corpus directory at module scope, so it has to exist before collection.
_CORPUS = Path(__file__).parent / "corpus"
_SENTINEL = _CORPUS / "invoice_plumber_march.pdf"


def _ensure_corpus() -> None:
    if _SENTINEL.exists():
        return
    import importlib.util

    gen = Path(__file__).parent / "generate_corpus.py"
    spec = importlib.util.spec_from_file_location("_generate_corpus", gen)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


_ensure_corpus()
