"""Hybrid search: KNN + BM25 + RRF + rerank."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import sqlite_vec

from find_and_seek.config.settings import (
    FILENAME_BOOST,
    KW_WEIGHT,
    MAX_CHUNKS_PER_FILE,
    RRF_K,
    SUMMARY_BOOST,
    VEC_WEIGHT,
)
from find_and_seek.db.store import estimated_tokens_for_file
from find_and_seek.ingest.embed import embed_query_vector
from find_and_seek.search.rerank import rerank


@dataclass
class SearchHit:
    file_id: int
    filename: str
    path: str
    document_type: str | None
    one_line_anchor: str | None
    summary: str | None
    relevance: float
    modified_at: str | None
    top_chunk_preview: str
    top_chunk_id: int
    top_chunk_location: str | None
    estimated_tokens_if_opened: int
    # The single most distinguishing typed fact (top money amount, else a content
    # date year) — the app composes the row title as "{filename identity} · {fact}"
    # so a row is a recognition hook, not a sentence (design panel). None when the
    # file has no salient fact (the title is then the filename alone).
    headline_fact: str | None = None
    # Classifier confidence in `document_type` ('high'|'medium'|'low'|'none'|None).
    # Carried through to the API/MCP so the type never reads as a hard fact when the
    # model wasn't sure — agents and the UI hedge on it. None = pre-confidence DB.
    classification_confidence: str | None = None
    # Human-readable confidence in the match itself, computed from the blended
    # relevance score. Parallels classification_confidence so agents can act on a
    # labeled band rather than reverse-engineering the fusion math.
    # "strong"  → CE clearly fired (≥0.55)
    # "moderate" → above the floor, likely CE-supported (≥RELEVANCE_FLOOR)
    # "weak"    → lexical rescue only (below floor but grounded by word overlap)
    match_confidence: str | None = None


# The most distinguishing typed fact for a row title, as a correlated subquery on
# `f.id`: the largest money amount, else the highest-confidence content date's year.
# Shared by search (hybrid) and browse (api/server) so titles compose identically.
HEADLINE_FACT_SQL = """COALESCE(
    (SELECT value_text FROM facts WHERE file_id = f.id AND fact_type = 'money'
     ORDER BY value_number DESC LIMIT 1),
    (SELECT substr(value_date, 1, 4) FROM facts WHERE file_id = f.id
     AND fact_type = 'date' AND value_date IS NOT NULL
     ORDER BY confidence DESC LIMIT 1)
)"""


def _fts_query(query: str) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression.

    Raw queries crash FTS5 on punctuation (``.``, ``:``, ``"``, ``*``, ``-``,
    ``(`` …). We extract word tokens, quote each as a phrase, and OR them so a
    query still matches on any term (recall) while the hybrid fusion + rerank
    handle precision.
    """
    terms = re.findall(r"\w+", query, flags=re.UNICODE)
    terms = [t for t in terms if len(t) > 1]
    if not terms:
        return ""
    return " OR ".join(f'"{t}"' for t in terms)


def _scope_clause(scope: str) -> tuple[str, list[Any]]:
    # Anchoring and LIKE-metacharacter escaping both live in db.scope so every
    # scoped query in the engine agrees on what a folder scope means.
    from find_and_seek.db.scope import scope_and_clause

    return scope_and_clause(scope)


# vec0 KNN can't JOIN-filter inside the MATCH, so we over-fetch candidates from
# the (fast, indexed) ANN operator and filter by status/scope afterwards. This is
# ~90x faster than scanning every vector with scalar vec_distance_cosine().
_OVERFETCH = 4
# Strongest keyword / vector hits force-fed into the rerank pool (see search()).
_GUARANTEED_KW = 10
_GUARANTEED_VEC = 10
# Cross-encoder rerank cost scales with candidates × text length, so bound both.
RERANK_CANDIDATES = int(os.environ.get("FINDANDSEEK_SEARCH_RERANK_CANDIDATES", "24"))
RERANK_TEXT_CHARS = 360

# ── Confidence gate ──────────────────────────────────────────────────
# The blended relevance score has a ~0.35 floor: the fused-norm term gives the
# top result that score even when the cross-encoder judges it irrelevant. So a
# query with NO real match (e.g. a multi-word agency name on an index
# that has no such document) would still surface vague vector neighbours at the
# floor. A hit is trusted only if EITHER its score clears the floor (the
# cross-encoder actually fired) OR the query's own words appear in it (lexical
# grounding). When nothing is confident we return zero results, so the UI / MCP
# can honestly say "couldn't find that" instead of padding with noise.
# Tuned on the live index; override via env if needed.
RELEVANCE_FLOOR = float(os.environ.get("FINDANDSEEK_SEARCH_RELEVANCE_FLOOR", "0.40"))
LEXICAL_FLOOR = float(os.environ.get("FINDANDSEEK_SEARCH_LEXICAL_FLOOR", "0.34"))
# Lexical grounding only rescues a hit if it ALSO clears this softer relevance
# floor. A genuine cross-encoder-floored match sits at ~0.35 (it's the top fused
# item); a coincidental whole-word collision on a short token (e.g. a
# three-letter headword in a bilingual dictionary) sits well below. This keeps the former and
# drops the latter.
LEXICAL_MIN_RELEVANCE = float(os.environ.get("FINDANDSEEK_SEARCH_LEXICAL_MIN_REL", "0.30"))

# ── Recall floor (agent / generation callers) ────────────────────────
# The confidence gate above is tuned for precision: on a vocabulary-gap query
# it will return [] or a single hit even when the best real document sits just
# below the floor (common once the vector arm does real work — the true match
# scores ~0.20, under the 0.40 floor). For an autonomous generation agent,
# one-or-zero results forces an extra search round-trip, which is the dominant
# cost. When FINDANDSEEK_SEARCH_MIN_RESULTS > 0, backfill the shortlist from the
# already-reranked candidates that clear a low ABSOLUTE floor — banded "weak"
# so callers can tell rescued hits from confident ones — while a query that
# genuinely matches nothing (no candidate clears the low floor) still returns
# empty. Default 0 preserves the precision-first behaviour for every other
# caller and the test suite.
SEARCH_MIN_RESULTS = int(os.environ.get("FINDANDSEEK_SEARCH_MIN_RESULTS", "0"))
SEARCH_BACKFILL_FLOOR = float(os.environ.get("FINDANDSEEK_SEARCH_BACKFILL_FLOOR", "0.12"))
_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "at", "is",
    "are", "was", "were", "be", "with", "from", "this", "that", "my", "your",
    "his", "her", "its", "it", "any", "all", "what", "which", "when", "where",
    "who", "how", "do", "does", "did", "about",
    # Generic verbs/quantifiers that carry no document vocabulary. Counting
    # them as content terms dilutes grounding: "how much I get paid each
    # fortnight" became 5 terms of which a payslip can only ever ground
    # "paid" (1/5 < floor), where the real signal is 1/2 of {paid, fortnight}.
    "much", "get", "got", "each", "every", "send", "sent", "out", "picked",
    # 2-char function words — 2-char tokens count as content terms (an
    # acronym like "CV" can be the only word that bridges query and document),
    # so the generic ones must be excluded explicitly.
    "up", "as", "by", "me", "we", "so", "no", "if", "go", "us", "ok", "he",
    "she", "i",
}


def _match_confidence(relevance: float) -> str:
    """Label the blended relevance score for agent consumption.

    The score is 0.65·sigmoid(CE) + 0.35·fused_norm, so the meaningful range
    starts at ~0.35 (fused-norm floor when CE contributes nothing).
      strong   ≥ 0.55  — cross-encoder clearly fired on real content overlap
      moderate ≥ floor — above the confidence gate, likely CE-supported
      weak     < floor — lexical rescue only (the gate let it through on word match)
    """
    if relevance >= 0.55:
        return "strong"
    if relevance >= RELEVANCE_FLOOR:
        return "moderate"
    return "weak"


def _content_terms(query: str) -> list[str]:
    """Meaningful query words for lexical grounding — drop stopwords and
    single chars. 2-char tokens stay: an acronym ("CV", "GP") can be the only
    word shared between the query and the matching document, and the generic
    2-char words are stopworded explicitly."""
    return [t for t in re.findall(r"\w+", query.lower())
            if len(t) > 1 and t not in _STOPWORDS]


def _match_window(text: str, terms: list[str], width: int) -> str:
    """Slice `width` chars of `text` centred on the first whole-word query-term
    hit, so the cross-encoder judges the passage that actually matched. Plain
    head-truncation decapitates keyword evidence: a BM25 hit deep in a page
    (e.g. a medicine name halfway down a dispense-record table) reached the
    reranker as 360 chars of page header, the CE scored the query ~0 against
    text that never contained it, and a true match fell out of the rerank cut
    — the "medicine name returns nothing" confident false negative."""
    if len(text) <= width:
        return text
    low = text.lower()
    first = -1
    for t in terms:
        m = re.search(rf"\b{re.escape(t)}\b", low)
        if m and (first == -1 or m.start() < first):
            first = m.start()
    if first == -1:
        return text[:width]
    start = max(0, min(first - width // 3, len(text) - width))
    return text[start:start + width]


_STEM_SUFFIXES = ("s", "es", "ed", "ly")


def _term_in(term: str, hay_words: set[str]) -> bool:
    """Whole-word match, tolerant of simple plural/adverb suffixes in either
    direction (fortnight↔fortnightly, employer↔employers). Short tokens stay
    exact-only — "tac" must not match "tacked"; and only listed suffixes count,
    so "contract" never matches "contractor"."""
    if term in hay_words:
        return True
    if len(term) >= 5:
        for suf in _STEM_SUFFIXES:
            if term + suf in hay_words:
                return True
            if term.endswith(suf) and len(term) - len(suf) >= 5 and term[: -len(suf)] in hay_words:
                return True
    return False


def _lexically_grounded(hit: SearchHit, terms: list[str]) -> bool:
    """Enough of `terms` appear in the hit (name / anchor / summary / matched
    passage) as WHOLE WORDS — not substrings, otherwise "TAC" spuriously
    matches inside "atTAChment" — and the hit clears the soft relevance floor.

    Used standalone for hits whose relevance score was computed against a
    DIFFERENT query (expansion variants): there the score can't be trusted as
    confidence in a match for the user's query, only the user's own words can."""
    uniq = set(terms)
    if not uniq:
        return False  # no lexical signal to fall back on
    hay = " ".join(filter(None, [
        hit.filename, hit.one_line_anchor, hit.summary, hit.top_chunk_preview,
    ])).lower()
    hay_words = set(re.findall(r"\w+", hay))
    overlap = sum(1 for t in uniq if _term_in(t, hay_words)) / len(uniq)
    return overlap >= LEXICAL_FLOOR and hit.relevance >= LEXICAL_MIN_RELEVANCE


def _is_confident(hit: SearchHit, terms: list[str]) -> bool:
    """True if this hit is trustworthy: a cross-encoder-backed score above the
    fusion floor, OR enough of the query's words actually appear in the file."""
    return hit.relevance >= RELEVANCE_FLOOR or _lexically_grounded(hit, terms)


def _vec_knn_chunks(
    conn: sqlite3.Connection,
    qvec: list[float],
    k: int,
    scope: str,
) -> list[tuple[int, float]]:
    blob = sqlite_vec.serialize_float32(np.array(qvec, dtype=np.float32))
    rows = conn.execute(
        "SELECT chunk_id, distance FROM chunk_vectors WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        [blob, k * _OVERFETCH],
    ).fetchall()
    if not rows:
        return []
    dist = {int(r[0]): float(r[1]) for r in rows}
    order = [int(r[0]) for r in rows]
    valid = _filter_chunks(conn, order, scope)
    return [(cid, dist[cid]) for cid in order if cid in valid][:k]


def _filter_chunks(conn: sqlite3.Connection, chunk_ids: list[int], scope: str) -> set[int]:
    if not chunk_ids:
        return set()
    scope_sql, params = _scope_clause(scope)
    ph = ",".join("?" for _ in chunk_ids)
    # `+f.status` tells SQLite NOT to treat status as an indexed lookup term.
    # Without it the planner drives the join from idx_files_status — scanning
    # *every* indexed file (~3k) and probing file_chunks per row (~58ms). The
    # unary plus forces the intended plan: PK lookup of the ~200 candidate
    # chunk_ids, then a PK join to files (~0.07ms, identical rows).
    rows = conn.execute(
        f"""
        SELECT fc.id FROM file_chunks fc JOIN files f ON f.id = fc.file_id
        WHERE fc.id IN ({ph}) AND +f.status = 'indexed'{scope_sql}
        """,
        [*chunk_ids, *params],
    ).fetchall()
    return {int(r[0]) for r in rows}


def _vec_knn_summaries(
    conn: sqlite3.Connection,
    qvec: list[float],
    k: int,
    scope: str,
) -> list[tuple[int, float]]:
    blob = sqlite_vec.serialize_float32(np.array(qvec, dtype=np.float32))
    rows = conn.execute(
        "SELECT file_id, distance FROM summary_vectors WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        [blob, k * _OVERFETCH],
    ).fetchall()
    if not rows:
        return []
    dist = {int(r[0]): float(r[1]) for r in rows}
    order = [int(r[0]) for r in rows]
    scope_sql, params = _scope_clause(scope)
    ph = ",".join("?" for _ in order)
    valid = {
        int(r[0])
        for r in conn.execute(
            # `+f.status`: force the PK-on-f.id plan instead of an idx_files_status scan.
            f"SELECT f.id FROM files f WHERE f.id IN ({ph}) AND +f.status='indexed'{scope_sql}",
            [*order, *params],
        ).fetchall()
    }
    return [(fid, dist[fid]) for fid in order if fid in valid][:k]


def _fts_bm25(
    conn: sqlite3.Connection,
    query: str,
    k: int,
    scope: str,
) -> list[tuple[int, float]]:
    fts_query = _fts_query(query)
    if not fts_query:
        return []
    scope_sql, params = _scope_clause(scope)
    # Rank + LIMIT inside the FTS5 subquery first (its optimized top-k path), THEN
    # join to filter by status/scope. Joining before the LIMIT forces bm25 over
    # every match (thousands of rows for common terms) — ~1000x slower.
    try:
        rows = conn.execute(
            f"""
            SELECT fc.id, sub.score FROM (
                SELECT rowid AS rid, bm25(chunk_fts) AS score
                FROM chunk_fts WHERE chunk_fts MATCH ?
                ORDER BY score LIMIT ?
            ) sub
            JOIN file_chunks fc ON fc.id = sub.rid
            JOIN files f ON f.id = fc.file_id
            WHERE f.status = 'indexed'{scope_sql}
            ORDER BY sub.score
            """,
            [fts_query, k * _OVERFETCH, *params],
        ).fetchall()
    except sqlite3.DatabaseError as e:
        # A damaged keyword index must never crash a query — degrade to
        # vector-only retrieval. init_db self-heals chunk_fts on startup, so
        # this is the last-resort guard for an index that broke mid-session.
        logging.getLogger(__name__).warning("FTS bm25 failed, degrading to vector-only: %s", e)
        return []
    return [(int(r[0]), float(-r[1])) for r in rows][:k]


def _section_anchor_for(section_anchors_json: str | None, location_ref: str | None) -> str | None:
    """For a composite file, the anchor of the page that matched (so a hit on
    page 2's invoice shows the invoice, not the file's page-1 anchor)."""
    if not section_anchors_json:
        return None
    from find_and_seek.ingest.sections import page_of

    page = page_of(location_ref)
    if page is None:
        return None
    try:
        anchors = json.loads(section_anchors_json)
    except (json.JSONDecodeError, TypeError):
        return None
    val = anchors.get(str(page)) if isinstance(anchors, dict) else None
    return val or None


def _boost_best_chunk_of_file(fused: dict[int, float], file_id: int, boost: float, conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT id FROM file_chunks WHERE file_id=? ORDER BY chunk_index LIMIT 1",
        (file_id,),
    ).fetchone()
    if row:
        fused[int(row[0])] = fused.get(int(row[0]), 0.0) + boost


def _cap_per_file(ranked: list[tuple[int, float]], max_per_file: int, conn: sqlite3.Connection) -> list[tuple[int, float]]:
    counts: dict[int, int] = {}
    out: list[tuple[int, float]] = []
    for cid, score in ranked:
        row = conn.execute("SELECT file_id FROM file_chunks WHERE id=?", (cid,)).fetchone()
        if not row:
            continue
        fid = int(row[0])
        if counts.get(fid, 0) >= max_per_file:
            continue
        counts[fid] = counts.get(fid, 0) + 1
        out.append((cid, score))
    return out


def _hits_from_scored(
    conn: sqlite3.Connection,
    scored: list[tuple[int, float]],
    type_filter: str | None,
    preview_by_cid: dict[int, str] | None = None,
) -> list[SearchHit]:
    """Materialise (chunk_id, relevance) pairs into SearchHits.

    `preview_by_cid` overrides the default head-of-chunk preview with the
    match-centred window (rescue path) — both so the agent/UI excerpt shows
    the passage that matched and so the gate's lexical grounding judges that
    passage rather than whatever happened to start the chunk."""
    candidates: list[SearchHit] = []
    for cid, score in scored:
        row = conn.execute(
            f"""
            SELECT fc.id, fc.text, fc.location_ref, fc.file_id,
                   f.filename, f.path, f.modified_at,
                   fs.document_type, fs.classification_confidence,
                   fs.one_line_anchor, fs.summary_text, fs.section_anchors,
                   {HEADLINE_FACT_SQL} AS headline_fact
            FROM file_chunks fc
            JOIN files f ON f.id = fc.file_id
            LEFT JOIN file_summaries fs ON fs.file_id = f.id
            WHERE fc.id = ?
            """,
            (cid,),
        ).fetchone()
        if not row:
            continue
        fid = int(row["file_id"])
        if type_filter and row["document_type"] and row["document_type"] != type_filter:
            continue
        preview = (preview_by_cid or {}).get(cid) or (row["text"] or "")[:300]
        # Composite (multi-document) file: show the anchor for the page that
        # actually matched, not the whole-file (page-1-only) anchor.
        anchor = _section_anchor_for(row["section_anchors"], row["location_ref"]) or row["one_line_anchor"]
        candidates.append(
            SearchHit(
                file_id=fid,
                filename=row["filename"],
                path=row["path"],
                document_type=row["document_type"],
                classification_confidence=row["classification_confidence"],
                one_line_anchor=anchor,
                summary=row["summary_text"],
                relevance=float(score),
                modified_at=(row["modified_at"] or "")[:10] if row["modified_at"] else None,
                top_chunk_preview=preview,
                top_chunk_id=int(row["id"]),
                top_chunk_location=row["location_ref"],
                estimated_tokens_if_opened=estimated_tokens_for_file(conn, fid),
                headline_fact=row["headline_fact"],
                match_confidence=_match_confidence(float(score)),
            )
        )
    return candidates


def _fused_retrieval(
    conn: sqlite3.Connection,
    query: str,
    scope: str,
) -> tuple[dict[int, float], list[tuple[int, float]], list[tuple[int, float]]]:
    """One retrieval pass: chunk-vector KNN + BM25 fused via RRF, plus the
    filename-token and summary-vector boosts. Returns (fused chunk scores, raw
    BM25 list, raw KNN list) — the caller guarantees the raw lists' top hits
    reach the reranker."""
    qvec = embed_query_vector(query)

    knn = _vec_knn_chunks(conn, qvec, k=50, scope=scope)
    fknn = _vec_knn_summaries(conn, qvec, k=20, scope=scope)
    kw = _fts_bm25(conn, query, k=50, scope=scope)

    fused: dict[int, float] = {}
    for rank, (cid, _) in enumerate(knn):
        fused[cid] = fused.get(cid, 0.0) + VEC_WEIGHT * (1.0 / (RRF_K + rank + 1))
    for rank, (cid, _) in enumerate(kw):
        fused[cid] = fused.get(cid, 0.0) + KW_WEIGHT * (1.0 / (RRF_K + rank + 1))

    # Filename token boost — only touch files whose name actually matches a
    # query term, instead of scanning every chunk in the corpus per query.
    # Dedupe and cap the term count: each term becomes an OR'd LIKE clause, and
    # SQLite caps expression-tree depth at 1000, so a pasted paragraph (1000+
    # tokens) would otherwise raise OperationalError. 32 distinct terms is far
    # more than any real query needs. 2-char tokens participate when they are
    # not stopwords — "CV" in a filename is a strong signal.
    q_terms = list(dict.fromkeys(
        t for t in query.lower().split()
        if len(t) > 2 or (len(t) == 2 and t not in _STOPWORDS)
    ))[:32]
    if q_terms:
        scope_sql, scope_params = _scope_clause(scope)
        like_sql = " OR ".join(["LOWER(f.filename) LIKE ?"] * len(q_terms))
        rows = conn.execute(
            f"""
            SELECT f.id, f.filename FROM files f
            WHERE f.status = 'indexed'{scope_sql} AND ({like_sql})
            """,
            [*scope_params, *[f"%{t}%" for t in q_terms]],
        ).fetchall()
        for row in rows:
            hits = sum(1 for t in q_terms if t in row["filename"].lower())
            if hits:
                _boost_best_chunk_of_file(fused, int(row["id"]), FILENAME_BOOST * hits, conn)

    for rank, (fid, _) in enumerate(fknn):
        _boost_best_chunk_of_file(fused, fid, SUMMARY_BOOST * (1.0 / (RRF_K + rank + 1)), conn)

    return fused, kw, knn


def _union_pool(
    conn: sqlite3.Connection,
    query: str,
    variants: list[str],
    scope: str,
) -> dict[int, float]:
    """Pooled retrieval prior over the original query plus every variant: the
    FULL fusion per phrasing (vec + bm25 + filename/summary boosts — a cut-down
    fusion was measured to bury the right files under generic vector
    neighbours), max-merged per chunk across phrasings. No consensus reward —
    relevance judging stays with the caller."""
    pool: dict[int, float] = {}
    for v in [query] + variants:
        try:
            fused, _, _ = _fused_retrieval(conn, v, scope)
        except Exception:  # noqa: BLE001 — rescue is best-effort
            continue
        for cid, s in fused.items():
            pool[cid] = max(pool.get(cid, 0.0), s)
    return pool


def _variant_union_rescue(
    conn: sqlite3.Connection,
    query: str,
    terms: list[str],
    variants: list[str],
    scope: str,
    limit: int,
    type_filter: str | None,
) -> list[SearchHit]:
    """Second-chance pass for the vocabulary gap: variants WIDEN the net, the
    user's ORIGINAL words judge relevance.

    Each per-variant search in the fallback gates its hits against the
    variant's own vocabulary — which usually doesn't overlap the document's
    either (that's the gap), so the right file surfaces at rank 1 of a variant
    pass and is then dropped ("Biweekly earnings rate" retrieves the payslip,
    but no payslip says "biweekly"). Here we pool what the original query and
    every variant retrieved and let the cross-encoder score that pool against
    the ORIGINAL query over match-centred windows — measured to fire on the
    original phrasing once it sees the right passage. The normal confidence
    gate still applies, so a query about nothing stays empty: a catering menu
    won't clear the floor for "recipe for chocolate brownies" no matter how
    many variants retrieve it."""
    pool = _union_pool(conn, query, variants, scope)
    if not pool:
        return []
    top_ids = [cid for cid, _ in sorted(pool.items(), key=lambda x: x[1], reverse=True)[:RERANK_CANDIDATES * 2]]
    # Window on the original terms first so the cross-encoder sees the user's
    # own vocabulary when the chunk has it; variant terms locate the passage
    # when it doesn't (the usual case in this path).
    window_terms = list(dict.fromkeys(terms + [t for v in variants for t in _content_terms(v)]))
    ph = ",".join("?" for _ in top_ids)
    texts = [
        (int(r[0]), _match_window(r[1] or "", window_terms, RERANK_TEXT_CHARS))
        for r in conn.execute(f"SELECT id, text FROM file_chunks WHERE id IN ({ph})", top_ids)
    ]
    scored = rerank(query, texts, fused_scores=pool, top_k=RERANK_CANDIDATES)
    diversified = _cap_per_file(scored, max_per_file=MAX_CHUNKS_PER_FILE, conn=conn)
    candidates = _hits_from_scored(conn, diversified, type_filter, preview_by_cid=dict(texts))
    return [h for h in candidates if _is_confident(h, terms)][:limit]


def _afm_judge_rescue(
    conn: sqlite3.Connection,
    query: str,
    terms: list[str],
    variants: list[str],
    scope: str,
    limit: int,
    type_filter: str | None,
    exclude_files: set[int],
) -> list[SearchHit]:
    """Final rescue: the on-device LLM judges the pooled candidates against
    the original query (see search/judge_afm.py for why and when).

    Both score-based arbiters are exhausted by the time this runs — the
    cross-encoder can't bridge pure synonymy ("resume" vs files that only say
    "CV") and lexical grounding has no shared word to hold. The pool is capped
    at one chunk per file and ~12 files so the judge reads a short, concrete
    list; it may select nothing, and a no-verdict (unavailable/timeout) adds
    nothing."""
    from find_and_seek.search import judge_afm

    if not judge_afm.available():
        return []
    pool = _union_pool(conn, query, variants, scope)
    if not pool:
        return []
    ranked = sorted(pool.items(), key=lambda x: x[1], reverse=True)
    diversified = _cap_per_file(ranked, max_per_file=1, conn=conn)[:24]
    if not diversified:
        return []
    window_terms = list(dict.fromkeys(terms + [t for v in variants for t in _content_terms(v)]))
    ph = ",".join("?" for _ in diversified)
    windows = {
        int(r[0]): _match_window(r[1] or "", window_terms, RERANK_TEXT_CHARS)
        for r in conn.execute(
            f"SELECT id, text FROM file_chunks WHERE id IN ({ph})", [c for c, _ in diversified]
        )
    }
    hits = [
        h for h in _hits_from_scored(conn, diversified, type_filter, preview_by_cid=windows)
        if h.file_id not in exclude_files
    ]
    # One judge slot per distinct DOCUMENT, not per copy: the pool routinely
    # holds the same file under several paths and as .docx + exported .pdf
    # (same name stem). Duplicates crowded the real target out of the list
    # and burned verification calls on the same content.
    seen_stems: set[str] = set()
    deduped: list[SearchHit] = []
    for h in hits:
        stem = h.filename.rsplit(".", 1)[0].lower()
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        deduped.append(h)
    hits = deduped[:12]
    if not hits:
        return []
    # Selection reads the summary, not the chunk window: scanned files often
    # window to OCR noise or an image caption ("baked_goods, bread" — which
    # the judge happily matched to a brownie recipe), while summaries are
    # dense, model-written prose.
    entries = [
        (h.file_id,
         f"{h.filename} [{h.document_type or 'file'}] — "
         f"{(h.one_line_anchor or '')[:120]} — {(h.summary or '')[:200]}")
        for h in hits
    ]
    picked = judge_afm.judge(query, entries)
    if not picked:  # None (no verdict) and [] (nothing matches) both add nothing
        return []
    selected = []
    for h in hits:
        if h.file_id not in set(picked) or len(selected) >= 3:
            continue
        # Stage-2: every pick is verified against its actual content — the
        # selection stage proposes, it does not decide. Unverified → dropped.
        passage = " ".join(filter(None, [h.one_line_anchor, h.summary, h.top_chunk_preview]))
        if judge_afm.verify(query, h.filename, h.document_type or "file", passage):
            selected.append(h)
    for h in selected:
        # The label reflects the basis of trust: an LLM that read the passage
        # vouched for the match. `relevance` stays the raw retrieval prior —
        # consumers are told to branch on the band, not the float.
        h.match_confidence = "moderate"
    return selected[:limit]


def search(
    conn: sqlite3.Connection,
    query: str,
    scope: str = "all",
    limit: int = 10,
    type_filter: str | None = None,
    _expanded: bool = False,
) -> tuple[list[SearchHit], float]:
    t0 = time.perf_counter()
    fused, kw, knn = _fused_retrieval(conn, query, scope)

    # Rerank is the expensive stage (cross-encoder pads every candidate to the
    # longest one), so only feed it the strongest handful and cap each text's
    # length — we surface ~10 results, 3 chunks/file max.
    pool = [cid for cid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:RERANK_CANDIDATES]]
    # Guarantee the strongest keyword and vector hits reach the reranker even if
    # RRF fusion (which rewards items hit by BOTH signals) would rank them out of
    # the pool. Otherwise an exact keyword match that isn't also a vector
    # neighbour — the common case for short, precise queries — gets buried under
    # vague vector hits and is never reranked. The cross-encoder, which reads the
    # actual text, is the right arbiter, so make sure it sees them.
    pool_set = set(pool)
    for cid, _ in kw[:_GUARANTEED_KW] + knn[:_GUARANTEED_VEC]:
        if cid not in pool_set:
            pool.append(cid)
            pool_set.add(cid)
    top = [(cid, fused.get(cid, 0.0)) for cid in pool]
    texts: list[tuple[int, str]] = []
    terms = _content_terms(query)
    if top:
        ph = ",".join("?" for _ in top)
        text_by_id = {
            int(r[0]): _match_window(r[1] or "", terms, RERANK_TEXT_CHARS)
            for r in conn.execute(
                f"SELECT id, text FROM file_chunks WHERE id IN ({ph})", [c for c, _ in top]
            ).fetchall()
        }
        texts = [(cid, text_by_id[cid]) for cid, _ in top if cid in text_by_id]

    reranked = rerank(query, texts, fused_scores=fused, top_k=RERANK_CANDIDATES)
    if not reranked:
        reranked = top

    diversified = _cap_per_file(reranked, max_per_file=MAX_CHUNKS_PER_FILE, conn=conn)
    candidates = _hits_from_scored(conn, diversified, type_filter)

    # Confidence gate: keep only results we can stand behind. If none qualify we
    # return [] so the caller can honestly report "no match" rather than noise.
    hits = [h for h in candidates if _is_confident(h, terms)][:limit]

    # Vocabulary-gap fallback (F-1.2): the fast path found nothing the
    # cross-encoder stands behind — likely the query's words just don't overlap
    # the document's. Generate a few SHORT rephrasings in different vocabulary
    # registers, run each, and fuse (best relevance per file). One register is
    # a lottery; three cover the space. Fires when the result is SPARSE (<3
    # hits) or all-weak, not only when empty: a stray hit must not silence the
    # fallback that could find the real match (measured twice — first a weak
    # grounded junk hit, then a single strong hit on a *guide about* failed
    # payments, each masking the user's actual document). Existing hits are
    # kept and merged, so a confident sparse answer keeps its rank and only
    # pays latency; every retry still passes the confidence gate — a query
    # about nothing stays empty.
    if not _expanded and (
        len(hits) < min(3, limit)
        or all(h.match_confidence == "weak" for h in hits)
    ):
        from find_and_seek.search.expand import expand_query, expand_variants

        variants = expand_variants(query)
        if not variants:
            one = expand_query(query)          # legacy single-shot, last resort
            variants = [one] if one != query else []
        best: dict[int, SearchHit] = {h.file_id: h for h in hits}
        borrowed: dict[int, SearchHit] = {}
        for v in variants:
            vhits, _ = search(conn, v, scope, limit, type_filter, _expanded=True)
            for h in vhits:
                if h.file_id in best:
                    continue  # the user's own query already vouched for it
                cur = borrowed.get(h.file_id)
                if cur is None or h.relevance > cur.relevance:
                    borrowed[h.file_id] = h

        # A borrowed hit passed a VARIANT's gate — its score and grounding
        # belong to the rephrasing, not to what the user asked ("Home Loan
        # Agreement" grounds a home-loan letter for a mortgage-contract query;
        # "Employment application" CE-fires on a separation certificate for a
        # resume query). Only the original query may pronounce it relevant:
        # keep it if it grounds in the user's own words, else ask the AFM
        # judge; without a verdict (or AFM at all) it is dropped — except on
        # machines without Apple Intelligence, where the judge can't exist and
        # we keep the pre-judge recall-over-precision behaviour.
        from find_and_seek.search import judge_afm

        judging = judge_afm.available()
        verify_budget = 4
        for h in sorted(borrowed.values(), key=lambda x: x.relevance, reverse=True):
            if _lexically_grounded(h, terms):
                # Its relevance was scored against the VARIANT, so any band it
                # carries is unearned here. What admitted it is word overlap —
                # that is the definition of the "weak" band. (An inflated band
                # also blocks the all-weak judge trigger below.)
                h.match_confidence = "weak"
                best[h.file_id] = h
            elif not judging:
                best[h.file_id] = h  # no judge on this machine — old behaviour
            elif verify_budget > 0:
                verify_budget -= 1
                passage = " ".join(filter(None, [h.one_line_anchor, h.summary, h.top_chunk_preview]))
                if judge_afm.verify(query, h.filename, h.document_type or "file", passage):
                    h.match_confidence = "moderate"
                    best[h.file_id] = h
        hits = sorted(best.values(), key=lambda h: h.relevance, reverse=True)[:limit]

        # Still nothing: the variants retrieved candidates but none survived
        # their own pass's gate (judged against the variant's vocabulary, not
        # the user's). Re-judge the pooled retrieval against the ORIGINAL query.
        if not hits and variants:
            hits = _variant_union_rescue(conn, query, terms, variants, scope, limit, type_filter)

        # Still sparse — or nothing better than weak lexical rescues: hand the
        # pooled candidates to the on-device LLM judge (AFM), the only arbiter
        # here that can bridge pure synonymy ("resume" vs files that only say
        # "CV"). Existing hits keep their rank; judged additions fill the tail.
        if len(hits) < min(3, limit) or all(h.match_confidence == "weak" for h in hits):
            judged = _afm_judge_rescue(
                conn, query, terms, variants, scope, limit, type_filter,
                exclude_files={h.file_id for h in hits},
            )
            if judged:
                hits = (hits + judged)[:limit]

    # Recall floor: fire ONLY when the precision gate returned nothing. A
    # zero-result search is what forces a generation agent to burn a round-trip
    # re-searching; that's the failure worth preventing. When the gate did
    # return a confident hit (the common case), we add nothing — padding every
    # search with extra weak cards just inflates the agent's input tokens for no
    # recall gain. So: only when empty, surface up to SEARCH_MIN_RESULTS of the
    # already-reranked candidates that clear the low absolute floor, banded
    # "weak". A query that genuinely matches nothing (no candidate above
    # SEARCH_BACKFILL_FLOOR) still returns [].
    if SEARCH_MIN_RESULTS and not hits:
        for h in sorted(candidates, key=lambda x: x.relevance, reverse=True):
            if len(hits) >= min(SEARCH_MIN_RESULTS, limit):
                break
            if h.relevance < SEARCH_BACKFILL_FLOOR:
                continue
            h.match_confidence = "weak"
            hits.append(h)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return hits, elapsed_ms


def hit_to_dict(hit: SearchHit) -> dict[str, Any]:
    return {
        "file_id": hit.file_id,
        "filename": hit.filename,
        "path": hit.path,
        "document_type": hit.document_type or "other",
        "classification_confidence": hit.classification_confidence,
        "one_line_anchor": hit.one_line_anchor or "",
        "summary": hit.summary or "",
        "relevance": round(hit.relevance, 4),
        # "strong" (≥0.55, CE fired) | "moderate" (above gate) | "weak" (lexical rescue).
        # The raw `relevance` score is 0.65·sigmoid(CE)+0.35·fused_norm; meaningful
        # range ~0.35–0.90. Use this band rather than the raw float for branching logic.
        "match_confidence": hit.match_confidence,
        "modified_at": hit.modified_at,
        "top_chunk_preview": hit.top_chunk_preview,
        "top_chunk_id": hit.top_chunk_id,
        "top_chunk_location": hit.top_chunk_location,
        "estimated_tokens_if_opened": hit.estimated_tokens_if_opened,
        "headline_fact": hit.headline_fact,
    }
