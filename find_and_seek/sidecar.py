"""Single dispatcher entry point for the packaged sidecar.

The shipped app freezes **one** binary (`findandseek-sidecar`) instead of four, so
PyInstaller ships a single copy of the heavy shared libraries (MLX, spaCy,
onnxruntime) rather than duplicating them per entry point — roughly a 4× size
cut. launchd's agents and the desktop app invoke it as:

    findandseek-sidecar api | worker | watch | service | setup | mcp   [args…]

Each sub-command runs the corresponding module exactly as its own console script
would (``run_name='__main__'``), so behaviour is identical to the unbundled CLIs.

``mcp`` runs the Model Context Protocol server over stdio. Unlike api/worker/watch
it is **not** a launchd-supervised service — an MCP client (e.g. Claude Desktop)
spawns ``findandseek-sidecar mcp`` on demand and talks to it over stdin/stdout.
Exposing it here is what lets the shipped binary be that command.
"""

from __future__ import annotations

import runpy
import sys

TARGETS = {
    "api": "find_and_seek.api.server",
    "worker": "find_and_seek.ingest.worker",
    "watch": "find_and_seek.watch.watcher",
    "service": "find_and_seek.service.launchd",
    "setup": "find_and_seek.setup_weights",
    "mcp": "find_and_seek.mcp.server",
}

# Targets are imported lazily — `runpy.run_module` below pulls in only the one
# we dispatch to, so a sidecar invocation doesn't pay the import cost of every
# other entry point. PyInstaller still bundles all of them via
# `--collect-all find_and_seek` in packaging/bundle_sidecar.sh, so bundling is
# decoupled from runtime import.


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in TARGETS:
        valid = "|".join(TARGETS)
        print(f"usage: findandseek-sidecar {{{valid}}} [args…]", file=sys.stderr)
        return 2
    name = sys.argv[1]
    module = TARGETS[name]
    # Present each sub-command as its own program to the module it runs.
    sys.argv = [f"findandseek-{name}", *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
