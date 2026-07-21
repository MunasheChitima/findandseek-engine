"""Debouncer: coalescing, serialisation, and bounded thread count.

The implementation this replaced used one ``threading.Timer`` per path, which
failed two ways that only appear under volume: a thread per file, and every one
of those threads calling ``settle()`` concurrently against the single shared
sqlite connection. These tests pin both properties down.
"""

from __future__ import annotations

import threading
import time

from find_and_seek.watch.watcher import Debouncer

# Short enough to keep the suite fast, long enough that a coalescing window
# actually exists on a loaded CI runner.
INTERVAL = 0.05
SETTLE_TIMEOUT = 5.0


class Recorder:
    """Records settle calls and flags any overlap between them."""

    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list[tuple[str, str]] = []
        self.overlapped = False
        self.max_concurrent = 0
        self._active = 0
        self._lock = threading.Lock()
        self._delay = delay
        self._seen = threading.Event()

    def __call__(self, path: str, kind: str) -> None:
        with self._lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
            if self._active > 1:
                self.overlapped = True
        if self._delay:
            time.sleep(self._delay)
        with self._lock:
            self._active -= 1
            self.calls.append((path, kind))
            self._seen.set()

    def wait_for(self, n: int, timeout: float = SETTLE_TIMEOUT) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.calls) >= n:
                    return True
            time.sleep(0.01)
        return False


def test_rapid_events_for_one_path_collapse_to_one_settle():
    rec = Recorder()
    d = Debouncer(rec, interval=INTERVAL)
    try:
        for _ in range(20):
            d.debounce("/x/a.pdf", "modified")
        assert rec.wait_for(1)
        time.sleep(INTERVAL * 3)  # let any stragglers land
        assert rec.calls == [("/x/a.pdf", "modified")]
    finally:
        d.stop()


def test_last_event_kind_wins():
    rec = Recorder()
    d = Debouncer(rec, interval=INTERVAL)
    try:
        d.debounce("/x/a.pdf", "created")
        d.debounce("/x/a.pdf", "modified")
        d.debounce("/x/a.pdf", "deleted")
        assert rec.wait_for(1)
        assert rec.calls == [("/x/a.pdf", "deleted")]
    finally:
        d.stop()


def test_distinct_paths_all_settle():
    rec = Recorder()
    d = Debouncer(rec, interval=INTERVAL)
    try:
        paths = [f"/x/{i}.pdf" for i in range(50)]
        for p in paths:
            d.debounce(p, "created")
        assert rec.wait_for(50)
        assert {p for p, _ in rec.calls} == set(paths)
    finally:
        d.stop()


def test_settles_never_overlap():
    """The whole point: settle() is a read-then-write against one shared sqlite
    connection, so two of them must never be in flight at once."""
    rec = Recorder(delay=0.02)
    d = Debouncer(rec, interval=INTERVAL)
    try:
        for i in range(30):
            d.debounce(f"/x/{i}.pdf", "created")
        assert rec.wait_for(30, timeout=10.0)
        assert not rec.overlapped
        assert rec.max_concurrent == 1
    finally:
        d.stop()


def test_thread_count_is_constant_in_the_number_of_files():
    """A 2000-file drop must not spawn 2000 threads."""
    rec = Recorder()
    before = threading.active_count()
    d = Debouncer(rec, interval=INTERVAL)
    try:
        for i in range(2000):
            d.debounce(f"/x/{i}.pdf", "created")
        # One worker thread, regardless of how many paths are pending.
        assert threading.active_count() - before <= 2
        assert rec.wait_for(2000, timeout=20.0)
    finally:
        d.stop()


def test_a_failing_settle_does_not_kill_the_worker():
    calls: list[str] = []

    def flaky(path: str, kind: str) -> None:
        calls.append(path)
        if path == "/x/bad.pdf":
            raise RuntimeError("boom")

    d = Debouncer(flaky, interval=INTERVAL)
    try:
        d.debounce("/x/bad.pdf", "created")
        time.sleep(INTERVAL * 4)
        d.debounce("/x/good.pdf", "created")
        deadline = time.monotonic() + SETTLE_TIMEOUT
        while "/x/good.pdf" not in calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "/x/good.pdf" in calls
    finally:
        d.stop()


def test_stop_joins_the_worker_and_ignores_late_events():
    rec = Recorder()
    d = Debouncer(rec, interval=INTERVAL)
    d.stop()
    assert not d._thread.is_alive()
    d.debounce("/x/late.pdf", "created")  # must not raise or resurrect the worker
    time.sleep(INTERVAL * 3)
    assert rec.calls == []
