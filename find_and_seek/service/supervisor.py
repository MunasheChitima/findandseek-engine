"""Cross-platform background supervisor (the launchd replacement off-Mac).

Runs the long-lived engine processes together and keeps them alive:

  • worker  — dequeues the ingest queue and indexes files (the actual ingest)
  • watch   — watches folders and enqueues changes (background ingest trigger)
  • api     — the local HTTP control/query server (optional)

On macOS the production supervisor is launchd (``findandseek-service``); this
is for Linux and Windows, where there is no launchd. It supervises children,
restarts a crashed one with capped backoff, and shuts them all down cleanly on
Ctrl-C / SIGTERM.

    findandseek-up                # worker + watch + api
    findandseek-up --no-api       # background ingest only (worker + watch)
    findandseek-up --only worker,watch

For start-on-boot, wrap this in a systemd user service (Linux — see
docs) or Task Scheduler (Windows). This process is the single thing those
supervisors need to launch.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import threading
import time

# name → module path (each module exposes a no-arg main())
SERVICES: dict[str, str] = {
    "worker": "find_and_seek.ingest.worker",
    "watch": "find_and_seek.watch.watcher",
    "api": "find_and_seek.api.server",
}

# A child that exits within this many seconds of starting counts as a "fast
# failure"; consecutive fast failures back off so a misconfigured service
# (e.g. missing Ollama) doesn't spin in a tight restart loop.
_FAST_FAIL_SECONDS = 10.0
_BACKOFF_MAX = 30.0


class _Child:
    def __init__(self, name: str, module: str) -> None:
        self.name = name
        self.module = module
        self.proc: subprocess.Popen | None = None
        self.fast_fails = 0
        self.started_at: float | None = None

    def _cmd(self) -> list[str]:
        # Run the module's main() via the current interpreter — no dependency on
        # console scripts being on PATH, works identically on all platforms.
        return [sys.executable, "-c", f"from {self.module} import main; main()"]

    def start(self) -> None:
        self.proc = subprocess.Popen(self._cmd())
        # Stamped here, not at poll time: the fast-fail window is measured from
        # when *this child* launched. Reading the clock inside the poll loop
        # instead makes every restart look instantaneous, so fast_fails never
        # resets and the backoff below pins at its ceiling forever.
        self.started_at = time.monotonic()
        print(f"[supervisor] started {self.name} (pid {self.proc.pid})", flush=True)

    def ran_for(self) -> float:
        """Seconds this child stayed up, or 0.0 if it was never started."""
        return 0.0 if self.started_at is None else time.monotonic() - self.started_at


def _run(names: list[str]) -> int:
    children = [_Child(n, SERVICES[n]) for n in names]
    stopping = threading.Event()

    def _shutdown(signum, _frame) -> None:  # noqa: ANN001
        if stopping.is_set():
            return
        stopping.set()
        print(f"\n[supervisor] signal {signum} — stopping {len(children)} service(s)…", flush=True)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    for c in children:
        c.start()

    try:
        while not stopping.is_set():
            for c in children:
                if c.proc is None:
                    continue
                rc = c.proc.poll()
                if rc is None:
                    continue
                if stopping.is_set():
                    continue
                # Child exited on its own — restart it, backing off on fast failures.
                if c.ran_for() < _FAST_FAIL_SECONDS:
                    c.fast_fails += 1
                else:
                    c.fast_fails = 0
                backoff = min(_BACKOFF_MAX, 2.0 ** min(c.fast_fails, 5)) if c.fast_fails else 0.0
                print(f"[supervisor] {c.name} exited (code {rc}); "
                      f"restarting in {backoff:.0f}s", flush=True)
                if backoff:
                    if stopping.wait(backoff):
                        break
                if not stopping.is_set():
                    c.start()
            stopping.wait(1.0)
    finally:
        # Terminate children, escalate to kill if they don't go quietly.
        for c in children:
            if c.proc and c.proc.poll() is None:
                c.proc.terminate()
        deadline = time.monotonic() + 10.0
        for c in children:
            if not c.proc:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            try:
                c.proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print(f"[supervisor] {c.name} did not stop — killing", flush=True)
                c.proc.kill()
        print("[supervisor] all services stopped", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="findandseek-up",
        description="Run the engine's background services together (worker + watch + api).",
    )
    ap.add_argument("--no-api", action="store_true",
                    help="skip the HTTP API — run background ingest only (worker + watch)")
    ap.add_argument("--only", default="",
                    help="comma-separated subset to run (worker,watch,api)")
    args = ap.parse_args()

    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in names if n not in SERVICES]
        if unknown:
            ap.error(f"unknown service(s): {', '.join(unknown)} — choose from {', '.join(SERVICES)}")
    else:
        names = ["worker", "watch"] + ([] if args.no_api else ["api"])

    print(f"[supervisor] launching: {', '.join(names)}", flush=True)
    return _run(names)


if __name__ == "__main__":
    raise SystemExit(main())
