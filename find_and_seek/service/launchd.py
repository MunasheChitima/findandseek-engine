"""launchd LaunchAgents for the Find and Seek services (macOS).

Installs per-user agents for the API, ingest worker, and file watcher so they
**start on login** and are **restarted on crash** by launchd — production service
supervision without us writing a babysitter. `install()` writes the plists to
``~/Library/LaunchAgents`` and bootstraps them; `uninstall()` reverses it.

CLI: ``findandseek-service install | uninstall | status``.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

LABEL_PREFIX = "com.findandseek"

# The long-running services launchd supervises. Each runs as a sub-command of the
# single consolidated sidecar binary (``findandseek-sidecar <name>``) — see
# find_and_seek/sidecar.py. (``setup`` is a one-shot fetch, not a supervised service.)
SERVICES = ("api", "worker", "watch")

LOG_DIR = Path.home() / ".findandseek" / "logs"
AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"

# Conservative production defaults (memory-bounded MLX). The code already defaults
# to the MLX backend; these just keep RSS in check and give subprocesses a PATH.
# NB: FINDANDSEEK_MIN_FREE_GB is intentionally NOT set here — the worker scales it
# to installed RAM (a flat 5 GB would block ingest forever on an 8 GB Mac).
_DEFAULT_ENV = {
    "FINDANDSEEK_MLX_CACHE_MB": "512",
    "PATH": ":".join([
        str(Path(sys.executable).parent),
        "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin",
    ]),
}


def label(name: str) -> str:
    return f"{LABEL_PREFIX}.{name}"


def plist_path(name: str) -> Path:
    return AGENTS_DIR / f"{label(name)}.plist"


def _sidecar_binary() -> str:
    """Absolute path to the consolidated ``findandseek-sidecar`` binary.

    In a **frozen .app** this code *is* that binary (``findandseek-sidecar service
    install`` brought us here), so ``sys.executable`` is exactly it. In **dev**
    it's the console script next to the interpreter, or on PATH.
    """
    if getattr(sys, "frozen", False):
        return sys.executable
    candidate = Path(sys.executable).parent / "findandseek-sidecar"
    if candidate.exists():
        return str(candidate)
    from shutil import which
    return which("findandseek-sidecar") or "findandseek-sidecar"


# Env vars worth baking into the plists at install time. launchd starts the
# agents directly (it does NOT source the app's sidecar-env.sh), so the offline
# model paths must travel via the plist EnvironmentVariables or the frozen
# services would try to fetch weights — breaking the zero-egress promise.
_FORWARDED_ENV_KEYS = (
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "FINDANDSEEK_RERANK_DIR",
    "FINDANDSEEK_DB_PATH",
    "FINDANDSEEK_MLX_CACHE_MB",
    "FINDANDSEEK_MIN_FREE_GB",
)


def _forwarded_env() -> dict[str, str]:
    """Curated subset of the current environment to bake into the agents."""
    return {k: os.environ[k] for k in _FORWARDED_ENV_KEYS if k in os.environ}


def plist_dict(name: str, env: dict[str, str] | None = None) -> dict:
    """The launchd job definition for one service."""
    return {
        "Label": label(name),
        "ProgramArguments": [_sidecar_binary(), name],
        "RunAtLoad": True,                       # start on login
        "KeepAlive": {"Crashed": True, "SuccessfulExit": False},  # restart on crash
        "ProcessType": "Background",
        "EnvironmentVariables": {**_DEFAULT_ENV, **(env or {})},
        "StandardOutPath": str(LOG_DIR / f"{name}.out.log"),
        "StandardErrorPath": str(LOG_DIR / f"{name}.err.log"),
        "WorkingDirectory": str(Path.home()),
    }


def write_plist(name: str, env: dict[str, str] | None = None) -> Path:
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = plist_path(name)
    with path.open("wb") as f:
        plistlib.dump(plist_dict(name, env), f)
    return path


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _bootstrap(path: Path) -> None:
    # `bootstrap` is the modern load; fall back to legacy `load -w` if unavailable.
    r = subprocess.run(["launchctl", "bootstrap", _domain(), str(path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        # `bootstrap` returns non-zero if the job is already loaded (errno 37/EALREADY)
        # — that's fine. For anything else, fall back to legacy `load -w` and, if that
        # also fails, surface stderr to our log so a stuck install is diagnosable
        # (grok P0.1: "logs are not prominently surfaced for why the engine isn't up").
        r2 = subprocess.run(["launchctl", "load", "-w", str(path)],
                            capture_output=True, text=True)
        if r2.returncode != 0:
            print(f"  ! bootstrap {path.name}: {r.stderr.strip() or r.returncode}; "
                  f"load: {r2.stderr.strip() or r2.returncode}", file=sys.stderr, flush=True)


def _bootout(name: str, path: Path) -> None:
    subprocess.run(["launchctl", "bootout", f"{_domain()}/{label(name)}"],
                   capture_output=True, text=True)
    if path.exists():
        subprocess.run(["launchctl", "unload", "-w", str(path)],
                       capture_output=True, text=True)


def install(env: dict[str, str] | None = None) -> list[str]:
    """Write + bootstrap all service agents, then verify they actually loaded.

    Returns the labels that launchd confirms are loaded. The desktop app treats a
    non-zero exit from ``service install`` as "engine failed to start", so we must
    not report success when bootstrap silently failed — otherwise the app waits
    90 s for a health endpoint that was never going to come up. A missing binary
    path or a bad plist shows up here as a label that never appears in
    ``launchctl list``.
    """
    installed = []
    for name in SERVICES:
        # Reinstall cleanly so an updated plist takes effect.
        _bootout(name, plist_path(name))
        path = write_plist(name, env)
        _bootstrap(path)
    loaded = status()
    for name in SERVICES:
        if loaded.get(label(name)):
            installed.append(label(name))
        else:
            print(f"  ! agent did not load: {label(name)} "
                  f"(plist {plist_path(name)})", file=sys.stderr, flush=True)
    return installed


def uninstall() -> list[str]:
    """Bootout + remove all service agents. Returns the labels removed."""
    removed = []
    for name in SERVICES:
        path = plist_path(name)
        _bootout(name, path)
        if path.exists():
            path.unlink()
        removed.append(label(name))
    return removed


def status() -> dict[str, bool]:
    """Map each label → whether launchd currently has it loaded."""
    r = subprocess.run(["launchctl", "list"], capture_output=True, text=True)
    loaded = r.stdout if r.returncode == 0 else ""
    return {label(name): (label(name) in loaded) for name in SERVICES}


def main() -> None:
    if sys.platform != "darwin":
        print("findandseek-service manages launchd agents and is macOS-only.\n"
              "Off-Mac, run the services directly (e.g. `findandseek-api`, "
              "`findandseek-worker`) or under systemd.", file=sys.stderr)
        sys.exit(2)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "install":
        # Forward offline-model + tuning env from the caller (the app sets these
        # from the bundle) into the plists so launchd-started services match.
        loaded = install(_forwarded_env())
        print("Installed launchd agents:")
        for lbl in loaded:
            print(f"  ✓ {lbl}")
        print(f"\nServices start on login and restart on crash. Logs: {LOG_DIR}")
        # Exit non-zero if any agent failed to load so the app sees the failure
        # (and shows it) instead of waiting on a health check that never goes green.
        if len(loaded) != len(SERVICES):
            print(f"  ✗ {len(SERVICES) - len(loaded)} of {len(SERVICES)} agents "
                  f"failed to load — see {LOG_DIR}", file=sys.stderr, flush=True)
            sys.exit(1)
    elif cmd == "uninstall":
        print("Removed launchd agents:")
        for lbl in uninstall():
            print(f"  ✗ {lbl}")
    elif cmd == "status":
        for lbl, up in status().items():
            print(f"  {'● running' if up else '○ stopped'}  {lbl}")
    else:
        print("usage: findandseek-service [install|uninstall|status]")
        sys.exit(2)


if __name__ == "__main__":
    main()
