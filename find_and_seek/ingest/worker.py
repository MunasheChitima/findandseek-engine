"""Ingest worker loop."""

from __future__ import annotations

import errno
import gc
import logging
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import psutil

from find_and_seek.db.connection import init_db, transaction
from find_and_seek.db.store import (
    INDEX_VERSION,
    MAX_ATTEMPTS,
    claim_queue,
    clone_indexed_file,
    indexed_twin,
    mark_done,
    mark_failed,
    purge_file_by_path,
    replace_chunks,
    requeue_pending,
    requeue_stale_processing,
    sha256_file,
    upsert_file,
    write_entities,
    write_facts,
    write_summary,
)
from find_and_seek.config.profiles import get_profile
from find_and_seek.ingest.chunk import chunk_blocks
from find_and_seek.ingest.embed import embed_texts
from find_and_seek.ingest.facts import extract_facts
from find_and_seek.ingest.extract.guard import ExtractTimeoutError, extract_with_timeout
from find_and_seek.ingest.extract.router import extract_file
from find_and_seek.ingest.florence import process_image
from find_and_seek.ingest.model_manager import model_manager
from find_and_seek.ingest.ner import extract_entities, extract_kf_entities
from find_and_seek.ingest.summarise import fast_classify, summarise_file, summarise_files
from find_and_seek.safe_log import safe_error
from find_and_seek import diagnostics

logger = logging.getLogger(__name__)

from find_and_seek.config.settings import (
    LOW_RAM,
    MAX_FILE_MB,
    default_batch_size,
    default_min_free_gb,
    ocr_enabled_default,
)
from find_and_seek.config.power_settings import get_mode
from find_and_seek.ingest.power import low_power_mode, on_ac_power

# Batch + memory-pause defaults scale with installed RAM so the worker is safe on
# 8 GB Macs (smaller batches, a reachable free-RAM threshold). Env vars override.
BATCH_SIZE = int(os.environ.get("FINDANDSEEK_BATCH_SIZE") or default_batch_size())

# Pause ingest if free RAM falls below this many GB. Prevents the worker from
# pushing the machine into swap (the OOM that took down the 3-worker run). The
# default scales with RAM — a flat 5 GB would block ingest forever on 8 GB.
MIN_FREE_GB = float(os.environ.get("FINDANDSEEK_MIN_FREE_GB") or default_min_free_gb())

# OCR loads the ~2.9 GB vision model, so it's off by default on low-RAM machines
# (text is still indexed; images just aren't transcribed). Opt in/out explicitly.
_ocr_env = os.environ.get("FINDANDSEEK_ENABLE_OCR")
OCR_ENABLED = ocr_enabled_default() if _ocr_env is None else _ocr_env.lower() in ("1", "true", "yes")

# Transient OS-level errors seen on plain file syscalls (open/stat/read) during
# parse — NOT defects in the file. The dominant one in production is EDEADLK
# ("Resource deadlock avoided", errno 11): a file read loses a kernel
# lock-ordering race against concurrent Metal/MLX GPU work and the kernel's
# deadlock-avoidance fails the syscall. The file is perfectly indexable a moment
# later, so we retry in-loop and, if still contended, requeue for a calmer pass
# rather than permanently dead-lettering it (AAR-031 §6b.1 — this was 76% of all
# ingest failures, ~728 files silently dropped from the index). EAGAIN/EBUSY are
# included for the same "try again shortly" reason.
_TRANSIENT_ERRNOS = frozenset({errno.EDEADLK, errno.EAGAIN, errno.EBUSY})
# In-loop immediate retries per file (the deadlock usually clears in ms once the
# concurrent GPU op finishes); each retry first hands MLX buffers back to the OS.
_PARSE_RETRIES = int(os.environ.get("FINDANDSEEK_PARSE_RETRIES") or 3)
_PARSE_BACKOFF = float(os.environ.get("FINDANDSEEK_PARSE_BACKOFF") or 0.1)

# Some large/malformed PDFs hang extract_file() indefinitely inside a native
# PyMuPDF call — no exception, ever. That's unbounded today; this caps it.
# 120s is generous headroom over even a 500+ legitimate scanned pages
# (PyMuPDF text extraction is roughly 10-50ms/page) while bounding the
# worst-case wedge per file to MAX_ATTEMPTS x this, not "forever".
EXTRACT_TIMEOUT_S = float(os.environ.get("FINDANDSEEK_EXTRACT_TIMEOUT_S") or 120)


def _is_transient_oserror(e: BaseException) -> bool:
    return isinstance(e, OSError) and e.errno in _TRANSIENT_ERRNOS


def machine_is_busy(cpu_threshold: float = 0.6) -> bool:
    return psutil.cpu_percent(interval=0.5) / 100.0 > cpu_threshold


def _free_ram_gb() -> float:
    return psutil.virtual_memory().available / 1e9


def _release_caches() -> None:
    """Hand transient memory back to the OS — Python garbage + MLX buffer cache."""
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:  # noqa: BLE001 — MLX not loaded / not installed
        pass


def memory_guard() -> bool:
    """True if it's safe to start another batch. When memory is tight, release
    caches, wait, and report not-safe so the worker idles instead of OOMing."""
    free = _free_ram_gb()
    if free < MIN_FREE_GB:
        logger.warning("Low memory (%.1f GB free < %.1f) — releasing caches, pausing", free, MIN_FREE_GB)
        diagnostics.record("mem_pause", free_gb=round(free, 2), threshold=MIN_FREE_GB)
        _release_caches()
        time.sleep(5)
        return False
    return True


def index_file(conn: sqlite3.Connection, path: str) -> None:
    """Index a single file synchronously (for tests and direct ingest)."""
    p = Path(path)
    if not p.exists():
        purge_file_by_path(conn, str(p))
        return

    try:
        content_hash = sha256_file(p)
    except OSError as e:
        logger.warning("Cannot read %s: %s", p, e)
        return

    try:
        result, chunks = extract_with_timeout(p, EXTRACT_TIMEOUT_S, extract_file, chunk_blocks)
        images = result.images

        # Only load the heavy vision model when there are images AND OCR is on
        # (off by default on 8 GB — see OCR_ENABLED).
        if images and OCR_ENABLED:
            with model_manager.use("florence"):
                for img in images:
                    img_chunks = process_image(img.data, img.location_ref)
                    for ic in img_chunks:
                        ic.chunk_index = len(chunks)
                        chunks.append(ic)

        vectors = embed_texts([c.text for c in chunks]) if chunks else []
        # use("summary") unloads the vision model first, so the heavy summariser
        # and VLM are never resident together (§5.7 — matters on 8 GB machines).
        with model_manager.use("summary"):
            summary = summarise_file(chunks, p.name)
        summary_vec = embed_texts([summary.get("summary_text", "") or p.name])[0]

        with transaction(conn):
            file_id = upsert_file(
                conn,
                str(p.resolve()),
                content_hash,
                result.file_type,
                page_count=result.page_count,
            )
            chunk_ids = replace_chunks(conn, file_id, chunks, vectors)
            entities = extract_entities(chunks, chunk_ids)
            entities.extend(extract_kf_entities(summary))
            write_entities(conn, file_id, entities)
            write_summary(conn, file_id, summary, summary_vec)
            write_facts(conn, file_id, extract_facts(summary, entities))

        mark_done(conn, str(p))
        conn.commit()
    except Exception as e:
        logger.error("Failed indexing %s: %s", p, safe_error(e))
        with transaction(conn):
            upsert_file(conn, str(p.resolve()), content_hash, "document", status="failed")
        raise e


def _dead_letter(
    conn: sqlite3.Connection,
    parsed: list[tuple],
    failures: list[tuple[int, Exception]],
    stage: str,
) -> list[tuple]:
    """Dead-letter the files that failed a heavy batch stage as individuals.

    Mirrors the per-file guard already used by the parse (:178) and index (:274)
    stages: each failing file is recorded, ``mark_failed``'d, and dropped from
    ``parsed`` so it never poisons its batch-mates — the rest of the batch
    proceeds. Returns ``parsed`` with the failed indices removed.
    """
    if not failures:
        return parsed
    bad: set[int] = set()
    for idx, exc in failures:
        item, p = parsed[idx][0], parsed[idx][2]
        path = item["path"]
        logger.error("Failed %s for %s: %s", stage, path, safe_error(exc))
        diagnostics.record_error(stage, exc, ext=p.suffix)
        mark_failed(conn, path, safe_error(exc))
        bad.add(idx)
    conn.commit()
    return [t for i, t in enumerate(parsed) if i not in bad]


def _parse_item(conn: sqlite3.Connection, item) -> tuple | None:
    """Parse one (non-deleted) queue item.

    Returns the ``(item, h, p, result, chunks)`` tuple to append to the batch, or
    ``None`` if the item was fully resolved here (missing file, byte-identical to
    an already-current row, or a duplicate cloned from a twin). Raises on a real
    extraction failure (handled by the caller's retry/dead-letter logic).
    """
    path = item["path"]
    p = Path(path)
    if not p.exists():
        mark_done(conn, path)
        return None
    # Max-file-size guard: extraction loads the whole file into RAM (text._read_text
    # does path.read_bytes()) and memory_guard only checks BETWEEN batches — so one
    # multi-GB file (log/DB-dump/export in a watched folder) can OOM the worker
    # mid-parse. Skip it gracefully: mark_done so it leaves the queue (it is NOT
    # poison, so no dead-letter), and record the skip for the tray. sha256_file
    # streams in 64 KB blocks, so hashing a big file is safe — only extraction is
    # the hazard, and we never reach it for an over-cap file.
    try:
        size_mb = p.stat().st_size / (1024 ** 2)
    except OSError:
        size_mb = 0.0
    if size_mb > MAX_FILE_MB:
        logger.info("Skipping oversized file (%.0f MB > %d MB cap): %s", size_mb, MAX_FILE_MB, path)
        diagnostics.record("file_skipped", reason="too_large",
                           ext=p.suffix, size_mb=round(size_mb, 1))
        mark_done(conn, path)
        return None
    h = sha256_file(p)
    resolved = str(p.resolve())
    row = conn.execute("SELECT content_hash, index_version FROM files WHERE path=?", (path,)).fetchone()
    if row and row[0] == h and row[1] == INDEX_VERSION:
        mark_done(conn, path)
        return None
    # Byte-identical duplicate already indexed → clone instead of re-extracting,
    # re-embedding, and re-summarising.
    twin = indexed_twin(conn, h, resolved)
    if twin is not None:
        tw = conn.execute("SELECT file_type, page_count FROM files WHERE id=?", (twin,)).fetchone()
        with transaction(conn):
            clone_indexed_file(conn, twin, resolved, h, tw["file_type"], page_count=tw["page_count"])
        mark_done(conn, path)
        conn.commit()
        return None
    result, chunks = extract_with_timeout(p, EXTRACT_TIMEOUT_S, extract_file, chunk_blocks)
    return (item, h, p, result, chunks)


def _on_parse_failure(conn: sqlite3.Connection, item, e: Exception) -> None:
    """Record a parse failure. Transient OS-level contention (EDEADLK &c.) is
    requeued for a later pass instead of dead-lettered — bounded by attempts so a
    genuinely stuck row can't loop forever; everything else is dead-lettered as
    before."""
    path = item["path"]
    if _is_transient_oserror(e):
        diagnostics.record_error("parse_transient", e, ext=Path(path).suffix)
        if item["attempts"] >= MAX_ATTEMPTS:
            # Last allowed claim — dead-letter so the row can't get stuck pending
            # (claim_queue won't re-select a row at the attempts ceiling).
            logger.error("Parse still contended after %d attempts, dropping %s: %s",
                         item["attempts"], path, safe_error(e))
            mark_failed(conn, path, safe_error(e))
        else:
            logger.warning("Transient OS error parsing %s (errno %s), requeuing: %s",
                           path, e.errno, safe_error(e))
            requeue_pending(conn, path, note=safe_error(e))
        conn.commit()
        return
    if isinstance(e, ExtractTimeoutError):
        # A genuine hang (no exception, ever, inside PyMuPDF) rather than a
        # momentary lock race — unlikely to clear on its own, so bounded the
        # same way as transient contention but on its own branch (no .errno).
        diagnostics.record_error("parse_timeout", e, ext=Path(path).suffix)
        if item["attempts"] >= MAX_ATTEMPTS:
            logger.error("Parse still timing out after %d attempts, dropping %s: %s",
                         item["attempts"], path, safe_error(e))
            mark_failed(conn, path, safe_error(e))
        else:
            logger.warning("Parse timed out on %s, requeuing: %s", path, safe_error(e))
            requeue_pending(conn, path, note=safe_error(e))
        conn.commit()
        return
    logger.error("Failed parsing %s: %s", path, safe_error(e))
    diagnostics.record_error("parse", e, ext=Path(path).suffix)
    try:
        p = Path(path)
        if p.exists():
            upsert_file(conn, str(p.resolve()), sha256_file(p), "document", status="failed")
    except OSError:
        pass
    mark_failed(conn, path, safe_error(e))
    conn.commit()


# Pacing knobs for "smooth" mode. A quarter-size batch plus a cool-down between
# batches keeps sustained CPU/GPU draw — and the fans — down, and on a laptop
# holds average draw below what the charger supplies so the battery doesn't
# discharge while plugged in. "full" mode ignores all of this and blitzes.
_SMOOTH_BATCH_SIZE = max(1, BATCH_SIZE // 4)
_SMOOTH_COOLDOWN_S = float(os.environ.get("FINDANDSEEK_SMOOTH_COOLDOWN_S") or 8.0)


@dataclass
class _Pacing:
    pause: bool
    batch_size: int
    cooldown_s: float
    reason: str


def _pacing() -> _Pacing:
    """Resolve performance mode + power state into how to run this batch.

    full   → blitz: full batch, no cool-down, ignore power state. The user ticked
             "Full power" and explicitly accepted the heat / battery cost.
    smooth → (default) be a good citizen:
               • Low-Power Mode or on battery  → pause (protect the battery).
               • on AC but the machine is busy → pause this cycle (yield to the
                 active user — keeps the fans down while they work).
               • on AC and idle               → gentle: quarter-size batch with a
                 cool-down after it so average draw and fan noise stay low.
    """
    if get_mode() == "full":
        return _Pacing(pause=False, batch_size=BATCH_SIZE, cooldown_s=0.0, reason="full")
    # smooth (default)
    if low_power_mode():
        return _Pacing(True, 0, 0.0, "low_power_mode")
    if not on_ac_power():
        return _Pacing(True, 0, 0.0, "on_battery")
    if machine_is_busy():
        return _Pacing(True, 0, 0.0, "machine_busy")
    return _Pacing(False, _SMOOTH_BATCH_SIZE, _SMOOTH_COOLDOWN_S, "smooth_ac")


def process_batch(conn: sqlite3.Connection) -> int:
    profile = get_profile()
    if profile.get("ingest_mode") == "idle_only" and machine_is_busy():
        return 0

    # Performance gate (D45): "smooth" (default) protects the battery and keeps the
    # fans quiet — pause on battery/Low-Power, yield to an actively-used machine,
    # and on AC index gently (small batches + cool-down). "full" blitzes. The tray
    # reads these diagnostics events to show why ingest is paused/throttled.
    pacing = _pacing()
    if pacing.pause:
        diagnostics.record("power_pause", reason=pacing.reason, mode=get_mode(),
                           on_ac=on_ac_power(), low_power=low_power_mode())
        time.sleep(5)
        return 0

    if not memory_guard():
        return 0

    batch_size = pacing.batch_size
    if pacing.cooldown_s > 0:
        diagnostics.record("power_throttle", reason=pacing.reason, mode=get_mode(),
                           count=batch_size)

    batch = claim_queue(conn, batch_size)
    if not batch:
        return 0

    t0 = time.perf_counter()
    parsed: list[tuple] = []

    for item in batch:
        path = item["path"]
        if item["event_type"] == "deleted":
            purge_file_by_path(conn, path)
            mark_done(conn, path)
            continue
        # Parse with a bounded in-loop retry: a transient EDEADLK (file read vs.
        # concurrent Metal/MLX GPU work) clears in milliseconds, so retry a few
        # times before falling through to requeue/dead-letter. Permanent errors
        # take the first failure straight to _on_parse_failure.
        for attempt in range(_PARSE_RETRIES):
            try:
                entry = _parse_item(conn, item)
                if entry is not None:
                    parsed.append(entry)
                break
            except Exception as e:
                if _is_transient_oserror(e) and attempt < _PARSE_RETRIES - 1:
                    logger.debug("Transient parse error on %s (errno %s), retry %d/%d",
                                 path, e.errno, attempt + 1, _PARSE_RETRIES)
                    _release_caches()  # hand MLX buffers back — may relieve the contention
                    time.sleep(_PARSE_BACKOFF * (attempt + 1))
                    continue
                _on_parse_failure(conn, item, e)
                break

    # Vision batch — only load the heavy VLM if OCR is on and some file has images.
    # OCR is an enhancement (the file's text is already chunked), so a vision
    # failure must never take the batch — or the worker — down: a model-load
    # failure skips OCR for this batch, and a per-file failure skips just that
    # file's images. Both keep the file and its text in the index.
    if OCR_ENABLED and any(r.images for (_, _, _, r, _) in parsed):
        try:
            with model_manager.use("florence"):
                for i, (item, h, p, result, chunks) in enumerate(parsed):
                    try:
                        for img in result.images:
                            for ic in process_image(img.data, img.location_ref):
                                ic.chunk_index = len(chunks)
                                chunks.append(ic)
                        parsed[i] = (item, h, p, result, chunks)
                    except Exception as e:  # noqa: BLE001 — OCR is optional
                        logger.warning("OCR failed for %s, indexing without image text: %s", p, safe_error(e))
                        diagnostics.record_error("vision", e, ext=p.suffix)
        except Exception as e:  # noqa: BLE001 — vision model unavailable
            logger.warning("Vision model unavailable, skipping OCR this batch: %s", safe_error(e))
            diagnostics.record_error("vision_load", e)

    # Embed — guarded per file so one poison file (MLX OOM, bad chunk) is
    # dead-lettered alone instead of escaping process_batch and killing the
    # worker mid-batch (which would dead-letter every good file it shared with).
    # Ollama is I/O-bound (HTTP round trips to a server that can serve several
    # requests at once), so fan the per-file calls out across threads there —
    # mirrors the ThreadPoolExecutor convention summarise_ollama.py already uses.
    # MLX stays sequential: it holds a single GPU context (see the EDEADLK note
    # on parse retries above) and doesn't benefit from Python-thread fan-out.
    embed_failures: list[tuple[int, Exception]] = []
    if os.environ.get("FINDANDSEEK_BACKEND") == "ollama" and len(parsed) > 1:
        concurrency = int(os.environ.get("FINDANDSEEK_OLLAMA_EMBED_CONCURRENCY", "8"))
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(lambda chunks: embed_texts([c.text for c in chunks]) if chunks else [], chunks): i
                for i, (item, h, p, result, chunks) in enumerate(parsed)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    vectors = fut.result()
                    item, h, p, result, chunks = parsed[i]
                    parsed[i] = (item, h, p, result, chunks, vectors)
                except Exception as e:  # noqa: BLE001 — isolate the failing file
                    embed_failures.append((i, e))
    else:
        for i, (item, h, p, result, chunks) in enumerate(parsed):
            try:
                vectors = embed_texts([c.text for c in chunks]) if chunks else []
                parsed[i] = (item, h, p, result, chunks, vectors)
            except Exception as e:  # noqa: BLE001 — isolate the failing file
                embed_failures.append((i, e))
    parsed = _dead_letter(conn, parsed, embed_failures, "embed")

    # Cascade — confidently-typed files (by container or header signal) are
    # classified deterministically and skip the LLM entirely; only the genuinely
    # ambiguous remainder goes to the 3B. If a whole batch is easy, the summary
    # model never even loads. The LLM stage is ~88% of ingest, so this is a big
    # throughput win and *more* accurate on the easy cases.
    summaries: dict[str, dict] = {}
    llm_idx: list[int] = []
    for i, (item, h, p, result, chunks, vectors) in enumerate(parsed):
        fast = fast_classify(result.file_type, chunks, p.name)
        if fast is not None:
            summaries[str(p)] = fast
        else:
            llm_idx.append(i)

    if llm_idx:
        # Summarise batch — one MLX forward pass for the whole batch (≈9× the
        # per-file path; Foundation Models can't batch). Falls back to per-file
        # internally for non-MLX backends.
        summ_failures: list[tuple[int, Exception]] = []
        with model_manager.use("summary"):
            batch_items = [(parsed[i][4], parsed[i][2].name) for i in llm_idx]
            try:
                batch_summaries = summarise_files(batch_items)
                for i, summ in zip(llm_idx, batch_summaries):
                    summaries[str(parsed[i][2])] = summ
            except Exception as e:  # noqa: BLE001 — batch pass failed
                # One poison file (or MLX OOM) failed the whole-batch forward
                # pass. Retry per-file so the good files still summarise and only
                # the genuinely bad one is dead-lettered — the batch is the fast
                # path, per-item is the safety net.
                logger.warning("Summarise batch failed, falling back per-file: %s", safe_error(e))
                diagnostics.record_error("summarise_batch", e)
                for i in llm_idx:
                    try:
                        summaries[str(parsed[i][2])] = summarise_file(parsed[i][4], parsed[i][2].name)
                    except Exception as fe:  # noqa: BLE001 — isolate the failing file
                        summ_failures.append((i, fe))
        parsed = _dead_letter(conn, parsed, summ_failures, "summarise")

    # Document TYPE is a deliberate, evidence-based decision (organize/classify),
    # decoupled from the summariser: hard signals (format/metadata) resolve ~a
    # third of files with no model; the rest go to the dedicated classifier; and
    # anything uncertain becomes 'needs-review' rather than a confident guess. We
    # override the type the summariser emitted (which was an incidental field).
    from find_and_seek.organize.classify import classify_document
    for item, h, p, result, chunks, vectors in parsed:
        summ = summaries.get(str(p))
        if not summ:
            continue
        text = "\n".join(c.text for c in chunks[:6])
        res = classify_document(str(p), p.name, text)
        summ["document_type"] = res.slug
        # Persist the structured confidence as a first-class field: the UI hedges
        # the badge by it and MCP hands it to agents, so a guess never wears the
        # authority of a fact. needs-review carries its own low-confidence note.
        summ["classification_confidence"] = res.confidence
        summ["suggested_category"] = res.suggested
        if res.needs_review:
            summ["confidence_note"] = "needs-review"

    # Summary-text embedding is another per-file HTTP round trip on Ollama;
    # precompute it concurrently ahead of the write loop below, which must stay
    # sequential (single sqlite connection). No-op dict for other backends, so
    # the inline embed_texts() fallback keeps their behavior unchanged.
    summary_vecs: dict[str, object] = {}
    summary_vec_errors: dict[str, Exception] = {}
    if os.environ.get("FINDANDSEEK_BACKEND") == "ollama" and len(parsed) > 1:
        concurrency = int(os.environ.get("FINDANDSEEK_OLLAMA_EMBED_CONCURRENCY", "8"))
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {}
            for item, h, p, result, chunks, vectors in parsed:
                summary = summaries.get(str(p))
                if summary is None:
                    continue
                futures[pool.submit(
                    lambda summary, p=p: embed_texts([summary.get("summary_text", "") or p.name])[0],
                    summary,
                )] = str(p)
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    summary_vecs[key] = fut.result()
                except Exception as e:  # noqa: BLE001 — isolate the failing file
                    summary_vec_errors[key] = e

    for item, h, p, result, chunks, vectors in parsed:
        path = item["path"]
        try:
            summary = summaries[str(p)]
            if str(p) in summary_vec_errors:
                raise summary_vec_errors[str(p)]
            summary_vec = summary_vecs.get(str(p))
            if summary_vec is None:
                summary_vec = embed_texts([summary.get("summary_text", "") or p.name])[0]
            with transaction(conn):
                file_id = upsert_file(
                    conn,
                    str(p.resolve()),
                    h,
                    result.file_type,
                    page_count=result.page_count,
                )
                chunk_ids = replace_chunks(conn, file_id, chunks, vectors)
                entities = extract_entities(chunks, chunk_ids)
                entities.extend(extract_kf_entities(summary))
                write_entities(conn, file_id, entities)
                write_summary(conn, file_id, summary, summary_vec)
                write_facts(conn, file_id, extract_facts(summary, entities))
            mark_done(conn, path)
        except Exception as e:
            logger.error("Failed indexing %s: %s", path, safe_error(e))
            diagnostics.record_error("index", e, ext=p.suffix)
            mark_failed(conn, path, safe_error(e))
        conn.commit()

    # Keep RSS flat between batches: drop the batch's parsed payloads and return
    # MLX's transient buffers to the OS.
    n = len(parsed)
    if n:
        diagnostics.record("ingest_batch", files=n,
                           seconds=round(time.perf_counter() - t0, 2),
                           free_gb=round(_free_ram_gb(), 2))
    parsed.clear()
    _release_caches()
    # Smooth-mode cool-down: sleep after each batch so sustained CPU/GPU draw drops
    # between batches — fans spin down and, on a laptop, average draw stays under
    # the charger. Only when we actually did work, to keep an empty queue from
    # idling slower than the main loop's own sleep.
    if pacing.cooldown_s > 0 and n:
        time.sleep(pacing.cooldown_s)
    return n


def _precompute_organize(conn: sqlite3.Connection) -> None:
    """After indexing settles, build the Organize tags + plan in the background so
    the UI opens instantly on precomputed data (never an on-click wait)."""
    if os.environ.get("FINDANDSEEK_ORGANIZE_PRECOMPUTE", "1") != "1":
        return
    try:
        from find_and_seek.organize.precompute import refresh_artifacts

        refresh_artifacts(conn)
    except sqlite3.OperationalError as e:
        # Another worker is already refreshing — fine, it's idempotent.
        logger.debug("organize precompute skipped (busy): %s", e)
    except Exception as e:  # noqa: BLE001 — never let analytics crash ingest
        logger.warning("organize precompute failed: %s", safe_error(e))


def run_worker(db_path: str | None = None) -> None:
    conn = init_db(db_path)
    # Crash recovery: reclaim rows a previous worker left mid-flight. This runs
    # BEFORE the main loop's guards, so it gets its own lock-retry — under heavy
    # write contention (e.g. the watcher bulk-enqueuing a backlog) an unguarded
    # "database is locked" here would crash the worker at startup and launchd
    # would just respawn it into the same contention (a startup crash-loop).
    for _attempt in range(10):
        try:
            requeued, dead = requeue_stale_processing(conn)
            if requeued or dead:
                logger.info("Recovered queue: %d requeued, %d dead-lettered", requeued, dead)
            break
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                logger.debug("startup recovery: queue busy, retrying: %s", e)
                time.sleep(1)
                continue
            raise
    logger.info("Ingest worker started")
    diagnostics.record("worker_start", tier=("low" if LOW_RAM else "full"),
                       count=BATCH_SIZE)
    did_work = False
    while True:
        try:
            n = process_batch(conn)
        except sqlite3.OperationalError as e:
            # Transient lock contention (multiple workers + bulk enqueue). Back
            # off briefly and retry rather than crashing the worker.
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                logger.debug("queue busy, backing off: %s", e)
                time.sleep(0.5)
                continue
            raise
        except Exception as e:  # noqa: BLE001 — last line of defence
            # The per-stage guards in process_batch should already isolate any
            # single bad file, so an escape here is unexpected. Don't let it kill
            # the worker (launchd would just restart it onto the SAME batch and
            # crash-loop): log, account for the in-flight rows, back off, retry.
            # claim_queue already bumped attempts, so a truly poisonous batch is
            # still bounded by MAX_ATTEMPTS — requeue_stale_processing requeues
            # rows with attempts left and dead-letters those that are exhausted.
            logger.error("Ingest batch crashed, recovering: %s", safe_error(e))
            diagnostics.record_error("worker_batch", e)
            try:
                requeued, dead = requeue_stale_processing(conn)
                if requeued or dead:
                    logger.info("post-crash recovery: %d requeued, %d dead-lettered", requeued, dead)
            except Exception as re:  # noqa: BLE001 — recovery is best-effort
                logger.warning("queue recovery after crash failed: %s", safe_error(re))
            _release_caches()
            time.sleep(2)
            continue
        if n == 0:
            # Queue just drained after doing work → refresh Organize artifacts once.
            if did_work:
                _precompute_organize(conn)
                did_work = False
            time.sleep(2)
        else:
            did_work = True


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    from find_and_seek.config.logging_setup import configure as _log_configure
    _log_configure()
    run_worker()


if __name__ == "__main__":
    main()
