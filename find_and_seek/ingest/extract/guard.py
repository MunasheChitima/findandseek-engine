"""Hard wall-clock bound on extraction + chunking.

Some large/malformed PDFs hang PyMuPDF indefinitely inside a native C call —
no exception, no return. That's a native busy-loop, not an I/O wait, so there's
no confirmed GIL release during the hang: a thread-based timeout only bounds
the *wait*, and the caller can still stall forever trying to reacquire the GIL
from a thread stuck mid-C-call that will never yield it. Only killing a
separate OS process guarantees reclaiming control regardless of what the child
is doing internally, so this runs extraction in a forked subprocess and kills
it if it doesn't report back in time.

Two fork-specific hazards this deliberately works around:

1. ``fork()`` duplicates the *entire* parent process — every open file
   descriptor (including an open sqlite3.Connection's WAL/SHM files) and
   every lock's current state, but none of the other threads that might own
   those locks. If the parent holds any lock at the instant of fork (SQLite's
   own WAL-mode mutex, the logging module's lock, ...), the child inherits it
   already-held with no owning thread ever able to release it. Letting the
   child return normally and fall into Python's ordinary interpreter
   shutdown/GC/``atexit`` sequence can then deadlock forever trying to
   finalize an inherited-but-unused resource. The fix: the child calls
   ``os._exit()`` right after handing back its result, skipping all Python
   shutdown machinery entirely (the OS closes its file descriptors anyway).
2. ``multiprocessing.Queue.put()`` hands off to a background feeder thread
   that serializes onto the pipe *asynchronously* — racing an immediate
   ``os._exit()`` in the same process can kill that thread before it flushes,
   silently dropping the result. ``multiprocessing.Pipe().send()`` writes
   synchronously in the calling thread, so it's safe to exit right after.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from typing import Callable

from find_and_seek.ingest.chunk import Chunk
from find_and_seek.ingest.chunk import chunk_blocks as _default_chunk_blocks
from find_and_seek.ingest.extract.base import ExtractResult
from find_and_seek.ingest.extract.router import extract_file as _default_extract_file


class ExtractTimeoutError(Exception):
    pass


def _extract_and_chunk(
    path_str: str,
    path_hint: str,
    extract_fn: Callable[[Path], ExtractResult],
    chunk_fn: Callable[..., list[Chunk]],
    conn,  # multiprocessing.connection.Connection (the child's end of a Pipe)
) -> None:
    try:
        result = extract_fn(Path(path_str))
        chunks = chunk_fn(result.blocks, path_hint=path_hint)
        conn.send(("ok", (result, chunks)))
    except Exception as e:  # noqa: BLE001 — relayed to the parent, not swallowed
        # Send the actual exception object (send() pickles it) so the caller
        # sees the real type/attributes — e.g. OSError.errno, which callers
        # like _is_transient_oserror() key off of. Collapsing it to a string
        # here would silently break that classification.
        conn.send(("err", e))
    finally:
        conn.close()
        # Skip Python's normal shutdown (atexit/GC/__del__) — see module
        # docstring: an inherited-but-unused lock from the parent (e.g. its
        # open sqlite connection) can deadlock a normal exit forever.
        os._exit(0)


def extract_with_timeout(
    p: Path,
    timeout_s: float,
    extract_fn: Callable[[Path], ExtractResult] = _default_extract_file,
    chunk_fn: Callable[..., list[Chunk]] = _default_chunk_blocks,
) -> tuple[ExtractResult, list[Chunk]]:
    """Run extract_fn + chunk_fn in a child process, killing it on timeout.

    extract_fn/chunk_fn are injectable (rather than hardcoded imports) so a
    caller's own module-level references — which tests monkeypatch, e.g.
    ``worker.extract_file`` — are what actually runs. With the ``fork`` start
    method this needs no pickling: the child inherits the parent's memory
    (including any monkeypatched callables) directly.

    Raises ExtractTimeoutError if the child had to be killed, or exited
    without sending a result. Otherwise re-raises whatever exception the
    child itself raised, faithfully (the same type and attributes, e.g.
    OSError.errno) — sent as the real object across the Pipe, not collapsed
    to a string.
    """
    ctx = multiprocessing.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_extract_and_chunk,
        args=(str(p), p.name, extract_fn, chunk_fn, child_conn),
    )
    proc.start()
    child_conn.close()  # only the child writes; drop the parent's write end

    # Drain the pipe *before* joining, not after: a payload bigger than the OS
    # pipe buffer (large ExtractResult — embedded images, long text) leaves the
    # child blocked in write() until someone reads. join(timeout_s) first would
    # wait for the child to exit, which it can't do until we read — deadlock.
    # poll(timeout_s) both bounds the wait and starts us reading as soon as
    # anything is available, unblocking the writer.
    try:
        if not parent_conn.poll(timeout_s):
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            raise ExtractTimeoutError(f"{p} exceeded {timeout_s}s during extract/chunk")

        try:
            kind, payload = parent_conn.recv()
        except EOFError:
            # Child exited without sending anything (killed by the OS, e.g.
            # OOM) rather than hitting our own timeout.
            proc.join(5)
            raise ExtractTimeoutError(
                f"{p}: extraction subprocess exited (code {proc.exitcode}) with no result"
            )
    finally:
        parent_conn.close()

    proc.join(5)  # reap; the child exits via os._exit() right after sending
    if proc.is_alive():
        proc.kill()
        proc.join()

    if kind == "err":
        raise payload
    result, chunks = payload
    return result, chunks
