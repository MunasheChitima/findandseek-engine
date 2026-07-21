"""Query expansion — a fallback for the semantic vocabulary gap (F-1.2).

When a search returns nothing, it's often not that the document is missing but
that the query's words don't overlap the document's at all ("how much I get paid"
vs a payslip that says "Annual Salary / Hourly Rate / Pay Date"). Neither the
embedding nor the lexical arm bridges that, and the confidence gate then honestly
returns []. Rather than lower the gate (measured counter-productive — it surfaces
wrong docs), we EXPAND the query once with the terms a matching document would
actually contain, and search again.

This is a fallback, not a default: it only runs when the fast path found nothing,
so the common case keeps its ~15ms latency and only the queries that currently
fail pay the model cost. Reuses the resident classifier model. Disable with
FINDANDSEEK_SEARCH_EXPAND=0.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)


def _generate(messages: list[dict[str, str]], max_tokens: int) -> str:
    """LLM completion for expansion. Apple Foundation Models first — on-device,
    zero model load, greedy (deterministic variants make a query's results
    reproducible). Falls back to the resident MLX classifier on machines
    without Apple Intelligence; raises only if both are unavailable (callers
    fail open)."""
    if os.environ.get("FINDANDSEEK_TEST") != "1":
        from find_and_seek.config.settings import _afm_available

        if _afm_available():
            from find_and_seek.search.judge_afm import complete

            out = complete("\n\n".join(m["content"] for m in messages), max_tokens=max_tokens)
            if out:
                return out
            logger.debug("AFM expansion gave no output — falling back to classifier")
    from find_and_seek.organize.classify import _classify_generate

    return _classify_generate(messages, max_tokens=max_tokens)


_SYSTEM = (
    "You expand a search query into the words a matching document would actually "
    "contain. Given the user's phrasing, output 6-12 alternative keywords and short "
    "phrases: synonyms, formal/technical terms, and closely related concepts a "
    "relevant document would use. Output ONLY a space-separated list of terms — no "
    "numbering, no explanation, no sentences. If the query is already specific, "
    "still give formal synonyms."
)

# One register is a lottery: "payment that bounced" can expand to NSF/chargeback
# bank-jargon while the actual letter says "direct debit repayment hasn't gone
# through". Three SHORT rephrasings in different registers cover the space; each
# stays short so it can't dilute the embedding, and reads as natural language so
# the cross-encoder scores it sanely.
# The registers are SPELLED OUT one per line: under greedy decoding (AFM) the
# earlier "vary the register (e.g. ...)" wording collapsed into three
# word-order shuffles of the same phrase, which lost the recall the variants
# exist to provide.
_VARIANTS_SYSTEM = (
    "Rewrite the search query three times, one rewrite per line:\n"
    "Line 1 — the formal/institutional vocabulary an official document (bank, "
    "government, employer) would use for it.\n"
    "Line 2 — the plain everyday words a person would label their own file with.\n"
    "Line 3 — the domain-specific term of art or document name for it.\n"
    "Each line: 3-6 words, concrete document vocabulary, no filler like "
    "'document about'. The three lines must use DIFFERENT key words from each "
    "other. Output exactly 3 lines — no numbering, no explanation."
)


def expand_query(query: str) -> str:
    """Return a document-vocabulary REPLACEMENT for `query` (the terms a matching
    document would contain), or the original query unchanged if expansion is off or
    the model is unavailable (fail open).

    Deliberately returns the terms ALONE, not `query + terms`: measured on the live
    index, prepending the oblique phrasing dilutes the embedding and skews the
    cross-encoder so the right docs fall back below the gate, while the terms alone
    retrieve them cleanly ("how much I get paid" → [] but "salary wages payslip
    gross net pay date" → the payslips). The terms are model-derived FROM the query,
    so intent is preserved; this only runs as a fallback after the original found
    nothing, so a literal query that already worked never reaches here."""
    if os.environ.get("FINDANDSEEK_SEARCH_EXPAND", "1") == "0":
        return query
    from find_and_seek.config.agent_mode import agent_search_enabled

    if agent_search_enabled():
        return query
    try:
        raw = _generate(
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": f"Query: {query}"}],
            max_tokens=60,
        )
        # Keep word-ish tokens only; drop any stray punctuation/numbering the model
        # adds. Cap so one query can't balloon the search.
        terms = re.findall(r"[A-Za-z][A-Za-z'/-]+", raw)
        # Keep it TIGHT: a long term-list dilutes the embedding and skews the
        # cross-encoder back below the gate (measured — a 12-term expansion misses
        # docs a 3-term one finds). First ~8 distinct concept words.
        uniq = list(dict.fromkeys(t for t in terms if len(t) > 2))[:8]
        extra = " ".join(uniq)
        if len(uniq) < 3:                 # too thin to trust — keep the original
            return query
        logger.debug("expanded %r -> %r", query, extra)
        return extra
    except Exception as e:  # noqa: BLE001 — expansion is best-effort, never break search
        logger.debug("query expansion unavailable: %s", e)
        return query


def expand_variants(query: str, n: int = 3) -> list[str]:
    """Up to `n` short, register-varied rephrasings of `query` in document
    vocabulary. [] when expansion is off/unavailable (fail open). The caller runs
    each variant and fuses results — multiple registers beat one term-soup shot."""
    if os.environ.get("FINDANDSEEK_SEARCH_EXPAND", "1") == "0":
        return []
    from find_and_seek.config.agent_mode import agent_search_enabled

    if agent_search_enabled():
        return []
    try:
        raw = _generate(
            [{"role": "system", "content": _VARIANTS_SYSTEM},
             {"role": "user", "content": f"Query: {query}"}],
            max_tokens=80,
        )
        out: list[str] = []
        q_words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'/-]+", query)}
        for line in raw.splitlines():
            # Strip any numbering/bullets the model adds despite instructions.
            words = re.findall(r"[A-Za-z][A-Za-z'/-]+", line)
            if not 2 <= len(words) <= 7:
                continue
            # A variant whose words are all already in the query brings no new
            # vocabulary — it can't bridge the gap, it can only DROP one of the
            # query's constraints ("tax return from 2019" → "tax return"
            # grounded junk that the full query correctly rejected).
            if all(w.lower() in q_words for w in words):
                continue
            cand = " ".join(words)
            if cand.lower() != query.lower() and cand not in out:
                out.append(cand)
            if len(out) >= n:
                break
        return out
    except Exception as e:  # noqa: BLE001 — expansion is best-effort, never break search
        logger.debug("variant expansion unavailable: %s", e)
        return []
