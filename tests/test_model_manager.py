"""ModelManager residency — a failed load must not be recorded as success.

The defect: `_load` swallowed its exception and `use` set `resident = name`
regardless, so one transient OOM left the manager believing a model was loaded
that never was. Every later `use()` of that model short-circuited the load and
never retried — summaries or embeddings silently stopped working for the life of
the worker, with nothing in the logs after the first failure.
"""

from __future__ import annotations

import pytest

from find_and_seek.ingest.model_manager import _LOAD_LOG_MAX, ModelManager


@pytest.fixture
def mgr(monkeypatch):
    m = ModelManager()
    monkeypatch.setattr(m, "_backend_for", lambda name: "mlx")
    return m


class _Backend:
    """Stands in for an MLX backend module; fails `fail_times` loads first."""

    def __init__(self, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.loads = 0
        self.unloads = 0

    def load(self) -> None:
        self.loads += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("simulated OOM")

    def unload(self) -> None:
        self.unloads += 1


def test_failed_load_is_not_recorded_as_resident(mgr, monkeypatch):
    backend = _Backend(fail_times=1)
    monkeypatch.setattr(mgr, "_mlx_module", lambda name: backend)

    with mgr.use("summary"):
        pass

    assert mgr.resident is None, "a model that failed to load must not be marked resident"


def test_a_transient_load_failure_is_retried_next_time(mgr, monkeypatch):
    """The consequence that mattered: without this, the second use() short-
    circuits and the model is never loaded again."""
    backend = _Backend(fail_times=1)
    monkeypatch.setattr(mgr, "_mlx_module", lambda name: backend)

    with mgr.use("summary"):
        pass
    assert backend.loads == 1

    with mgr.use("summary"):  # must try again, not short-circuit
        pass
    assert backend.loads == 2, "a failed load was never retried"
    assert mgr.resident == "summary", "the retry succeeded, so it should be resident"


def test_successful_load_short_circuits_on_reuse(mgr, monkeypatch):
    """The optimisation the bug was hiding inside must still work."""
    backend = _Backend()
    monkeypatch.setattr(mgr, "_mlx_module", lambda name: backend)

    for _ in range(3):
        with mgr.use("summary"):
            pass

    assert backend.loads == 1, "a resident model should not be reloaded"
    assert mgr.resident == "summary"


def test_switching_models_unloads_the_previous_one(mgr, monkeypatch):
    backends = {"summary": _Backend(), "florence": _Backend()}
    monkeypatch.setattr(mgr, "_mlx_module", lambda name: backends[name])

    with mgr.use("summary"):
        pass
    with mgr.use("florence"):
        pass

    assert backends["summary"].unloads == 1
    assert mgr.resident == "florence"


def test_load_log_is_bounded(mgr, monkeypatch):
    """It lives on a worker that runs for weeks and swaps models per batch."""
    backends = {"summary": _Backend(), "florence": _Backend()}
    monkeypatch.setattr(mgr, "_mlx_module", lambda name: backends[name])

    for i in range(_LOAD_LOG_MAX * 2):
        with mgr.use("summary" if i % 2 else "florence"):
            pass

    assert len(mgr.load_log) <= _LOAD_LOG_MAX
