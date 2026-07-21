"""File watcher with debounce and hash guard."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from find_and_seek.db.connection import init_db
from find_and_seek.db.store import enqueue, file_with_hash, get_stored_hash, sha256_file, update_path
from find_and_seek.ingest.extract.router import WATCHED_EXTENSIONS
from find_and_seek.watch.exclusions import is_user_excluded

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 3.0

EXCLUDED_PREFIXES = (
    "/System",
    "/Library",
    "/usr",
    "/bin",
    "/tmp",
    "/var",
    "/.Trash",
)

# Directory names to skip anywhere in the tree — dependency/build/cache dirs that
# would flood the index with junk when a dev workspace (e.g. ~/Coding) is watched.
EXCLUDED_DIRS = {
    "node_modules", ".git", ".venv", "venv", "env", "__pycache__", "dist", "build",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "site-packages", ".tox",
    ".next", ".cache", "target", ".idea", ".gradle", ".terraform", "vendor",
}


def is_excluded_dir(parts: tuple[str, ...]) -> bool:
    """True if any path component is a dependency/build/cache dir to skip.

    Matches the exact names in EXCLUDED_DIRS, plus two patterns: ``*.egg-info``
    build metadata and any ``node_modules*`` variant. The prefix match matters
    because renaming a ``node_modules`` folder (e.g. to
    ``node_modules node_modules_IGNORE`` so *npm* ignores it) otherwise defeats the
    exact-name skip and floods the queue with thousands of dependency files that
    then wedge at the retry ceiling."""
    return any(
        p in EXCLUDED_DIRS or p.endswith(".egg-info") or p.startswith("node_modules")
        for p in parts
    )

DEFAULT_ROOTS = [
    Path.home() / "Documents",
    Path.home() / "Desktop",
    Path.home() / "Downloads",
]


class Debouncer:
    """Coalesce rapid events per path, then settle them on one worker thread.

    The obvious implementation — a ``threading.Timer`` per path — is wrong
    twice over, and both failures need volume to show up:

      • Thread count is O(files). Dropping a 10k-file folder into a watched
        root spawns 10k live timer threads.
      • Every one of those threads calls ``settle()`` concurrently against the
        single shared sqlite connection. ``settle()`` is a read-then-write
        (``get_stored_hash`` → ``file_with_hash`` → ``update_path`` → commit),
        so concurrent callers interleave, and one thread's ``commit()`` lands
        another thread's half-finished transaction.

    Keeping a deadline per path and draining from a single thread fixes both:
    ``settle()`` is serialised by construction, and the thread count is O(1).
    """

    def __init__(self, settle_fn, interval: float = DEBOUNCE_SECONDS) -> None:
        self._settle_fn = settle_fn
        self._interval = interval
        self._pending: dict[str, tuple[float, str]] = {}  # path → (due_at, kind)
        self._cv = threading.Condition()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, name="fs-debounce", daemon=True)
        self._thread.start()

    def debounce(self, path: str, kind: str) -> None:
        """(Re)arm *path*; the last event within the window wins."""
        with self._cv:
            if self._stopped:
                return
            self._pending[path] = (time.monotonic() + self._interval, kind)
            self._cv.notify()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop draining and join the worker. Pending events are dropped —
        they are re-discovered by the startup scan on the next run."""
        with self._cv:
            self._stopped = True
            self._cv.notify()
        self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            with self._cv:
                if self._stopped:
                    return
                if not self._pending:
                    self._cv.wait()
                    continue
                now = time.monotonic()
                due = [(p, kind) for p, (at, kind) in self._pending.items() if at <= now]
                if not due:
                    nxt = min(at for at, _ in self._pending.values())
                    self._cv.wait(max(0.0, nxt - now))
                    continue
                for path, _ in due:
                    del self._pending[path]
            # Settle outside the lock: it does file hashing and sqlite work, and
            # a slow settle must not block the watchdog thread's debounce()
            # calls. Re-arming a path mid-settle just queues it again.
            for path, kind in due:
                try:
                    self._settle_fn(path, kind)
                except Exception:  # noqa: BLE001 - one bad file must not kill the watcher
                    logger.exception("settle failed for %s", path)


class FindandSeekHandler(FileSystemEventHandler):
    def __init__(self, conn: sqlite3.Connection, debouncer: Debouncer) -> None:
        self.conn = conn
        self.debouncer = debouncer

    def _handle(self, path: str, kind: str) -> None:
        p = Path(path)
        if p.name.startswith("."):
            return
        if is_excluded_dir(p.parts):
            return
        if any(str(p).startswith(x) for x in EXCLUDED_PREFIXES):
            return
        # User-excluded subtree (carved out of a granted root). Read fresh per
        # event so edits to exclusions.json take effect without a restart.
        if is_user_excluded(str(p)):
            return
        ext = p.suffix.lower()
        if ext and ext not in WATCHED_EXTENSIONS:
            return
        self.debouncer.debounce(str(p), kind)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle(event.src_path, "created")

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle(event.src_path, "modified")

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle(event.src_path, "deleted")

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._handle(event.src_path, "deleted")
        self._handle(event.dest_path, "created")


def settle(conn: sqlite3.Connection, path: str, kind: str) -> None:
    p = Path(path)
    if kind == "deleted" or not p.exists():
        enqueue(conn, path, "deleted")
        conn.commit()
        return

    try:
        h = sha256_file(p)
    except OSError:
        return

    stored = get_stored_hash(conn, path)
    if stored == h:
        return

    existing = file_with_hash(conn, h)
    if existing and existing["path"] != path and not Path(existing["path"]).exists():
        update_path(conn, int(existing["id"]), path)
        conn.commit()
        return

    event_type = "created" if stored is None else "modified"
    enqueue(conn, path, event_type)
    conn.commit()


def start_watcher(conn: sqlite3.Connection, roots: list[Path] | None = None) -> Observer:
    if roots is None:
        from find_and_seek.watch.roots import list_roots

        roots = [r for r in list_roots() if r.exists()]
    debouncer = Debouncer(lambda path, kind: settle(conn, path, kind))
    handler = FindandSeekHandler(conn, debouncer)
    observer = Observer()
    for root in roots:
        observer.schedule(handler, str(root), recursive=True)
        logger.info("Watching %s", root)
    observer.start()
    return observer


ROOTS_POLL_SECONDS = 5.0


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    from find_and_seek.watch.roots import list_roots
    from find_and_seek.watch.scan import scan_roots

    conn = init_db()
    roots = [r for r in list_roots() if r.exists()]

    # Backfill existing files once, then watch for changes.
    logger.info("Initial scan of %d root(s)…", len(roots))
    enqueued = scan_roots(conn, roots)
    logger.info("Enqueued %d file(s) from initial scan", enqueued)

    debouncer = Debouncer(lambda path, kind: settle(conn, path, kind))
    handler = FindandSeekHandler(conn, debouncer)
    observer = Observer()
    watched: dict[str, object] = {}
    for root in roots:
        watched[str(root)] = observer.schedule(handler, str(root), recursive=True)
        logger.info("Watching %s", root)
    observer.start()

    try:
        while True:
            time.sleep(ROOTS_POLL_SECONDS)
            current = {str(r) for r in list_roots() if r.exists()}
            for r in current - set(watched):  # newly added folder
                watched[r] = observer.schedule(handler, r, recursive=True)
                logger.info("Now watching %s", r)
                # New roots are backfilled by the API on add; this covers
                # folders added by editing roots.json directly.
                scan_roots(conn, [Path(r)])
            for r in set(watched) - current:  # removed folder
                observer.unschedule(watched.pop(r))  # type: ignore[arg-type]
                logger.info("Stopped watching %s", r)
    except KeyboardInterrupt:
        logger.info("Stopping…")
    finally:
        # Stop the observer first so no new events arrive, then drain-stop the
        # debouncer — the other order lets a late event re-arm a dead worker.
        observer.stop()
        debouncer.stop()
    observer.join()


if __name__ == "__main__":
    main()
