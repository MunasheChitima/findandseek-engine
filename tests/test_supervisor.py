"""Supervisor child lifecycle and fast-fail backoff.

The backoff regression these guard against was silent: ``started`` was sampled
inside the poll loop rather than at launch, so ``ran_for()`` was always ~0,
every restart counted as a fast failure, ``fast_fails`` never reset, and the
delay pinned at the 30s ceiling forever — including for a service that had been
healthy for hours. Nothing crashed; restarts just got slower and slower.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from find_and_seek.service import supervisor
from find_and_seek.service.supervisor import _BACKOFF_MAX, _FAST_FAIL_SECONDS, SERVICES, _Child, main


def backoff_for(fast_fails: int) -> float:
    """Mirror of the delay expression in the supervisor's restart branch."""
    return min(_BACKOFF_MAX, 2.0 ** min(fast_fails, 5)) if fast_fails else 0.0


# ── ran_for: the actual regression ────────────────────────────────


def test_ran_for_is_zero_before_start():
    assert _Child("worker", "x.y").ran_for() == 0.0


def test_ran_for_measures_from_launch_not_from_poll(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: _FakeProc())

    c = _Child("worker", "x.y")
    c.start()
    clock["t"] += 3600.0  # child stayed up an hour
    assert c.ran_for() == pytest.approx(3600.0)


def test_restarting_restamps_the_clock(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: _FakeProc())

    c = _Child("worker", "x.y")
    c.start()
    clock["t"] += 500.0
    c.start()
    clock["t"] += 2.0
    assert c.ran_for() == pytest.approx(2.0)


def test_a_long_lived_child_is_not_a_fast_failure(monkeypatch):
    """The bug in one assertion: a child up for an hour must reset the counter."""
    clock = {"t": 0.0}
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: _FakeProc())

    c = _Child("worker", "x.y")
    c.fast_fails = 5  # previously crash-looping, then stabilised
    c.start()
    clock["t"] += 3600.0

    if c.ran_for() < _FAST_FAIL_SECONDS:
        c.fast_fails += 1
    else:
        c.fast_fails = 0

    assert c.fast_fails == 0
    assert backoff_for(c.fast_fails) == 0.0


def test_a_crash_looping_child_backs_off(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **k: _FakeProc())

    c = _Child("worker", "x.y")
    delays = []
    for _ in range(8):
        c.start()
        clock["t"] += 0.2  # dies immediately, every time
        if c.ran_for() < _FAST_FAIL_SECONDS:
            c.fast_fails += 1
        else:
            c.fast_fails = 0
        delays.append(backoff_for(c.fast_fails))

    assert delays[0] < delays[1] < delays[2]  # actually escalates
    assert delays[-1] == _BACKOFF_MAX  # and caps
    assert max(delays) <= _BACKOFF_MAX


# ── command construction ──────────────────────────────────────────


class _FakeProc:
    pid = 4242

    def poll(self):
        return None


def test_cmd_invokes_the_module_main_via_this_interpreter():
    cmd = _Child("worker", "find_and_seek.ingest.worker")._cmd()
    assert cmd[0] == sys.executable
    assert cmd[1] == "-c"
    assert "from find_and_seek.ingest.worker import main; main()" == cmd[2]


@pytest.mark.parametrize("name,module", sorted(SERVICES.items()))
def test_every_service_module_is_importable_and_has_a_main(name, module):
    """SERVICES entries are launched as subprocesses, so a typo here surfaces
    only at runtime, in a child, as a restart loop."""
    mod = __import__(module, fromlist=["main"])
    assert callable(getattr(mod, "main", None)), f"{module} has no callable main()"


# ── CLI surface ───────────────────────────────────────────────────


def test_only_flag_rejects_unknown_services(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["findandseek-up", "--only", "worker,nope"])
    with pytest.raises(SystemExit):
        main()
    assert "unknown service" in capsys.readouterr().err


def test_no_api_runs_ingest_only(monkeypatch):
    seen = {}
    monkeypatch.setattr(supervisor, "_run", lambda names: seen.setdefault("names", names) or 0)
    monkeypatch.setattr(sys, "argv", ["findandseek-up", "--no-api"])
    main()
    assert seen["names"] == ["worker", "watch"]


def test_default_runs_all_three(monkeypatch):
    seen = {}
    monkeypatch.setattr(supervisor, "_run", lambda names: seen.setdefault("names", names) or 0)
    monkeypatch.setattr(sys, "argv", ["findandseek-up"])
    main()
    assert seen["names"] == ["worker", "watch", "api"]


def test_child_start_is_wired_to_popen(monkeypatch):
    """Guards the cmd/Popen seam the other tests fake out."""
    captured = {}

    def fake_popen(cmd, *a, **k):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    c = _Child("api", "find_and_seek.api.server")
    c.start()
    assert captured["cmd"] == c._cmd()
    assert c.started_at is not None
