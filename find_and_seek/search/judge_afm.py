"""Last-resort LLM relevance judge — Apple Foundation Models (macOS 26+).

The cross-encoder is the relevance arbiter for the fast path, but it is a
shallow model: on vocabulary-gap pairs ("the resume I send out to employers"
vs documents that only ever say "CV") its score is indistinguishable from
noise, and lexical grounding has no shared word to hold on to. Retrieval still
gets the right document into the candidate pool — judging is what's missing.

AFM is the on-device system LLM (zero extra RAM, ~3s/call): strong enough to
bridge register and synonymy, private by construction. It runs ONLY on the
rescue path — queries that would otherwise return sparse-or-nothing — so the
common case never pays for it. Its verdict obeys the same honesty rule as the
confidence gate: it can only pick from what retrieval surfaced, and "NONE of
these" is an expected, common answer — a query about nothing stays empty.

Disable with FINDANDSEEK_SEARCH_AFM_JUDGE=0. Never active under FINDANDSEEK_TEST.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import re

logger = logging.getLogger(__name__)

_TIMEOUT_S = 15.0

# Stage 1 — selection. Recall-oriented: a small model free-associates under
# pressure to answer, so its picks are PROPOSALS only; stage 2 verifies each.
_PROMPT = (
    "You are judging search results for a personal file index.\n"
    'Query: "{query}"\n'
    "\n"
    "Candidates:\n"
    "{numbered}\n"
    "\n"
    "Which candidates are files the user is asking for? Judge by MEANING, not "
    "shared words — a CV matches a query about a resume; a dishonoured direct "
    "debit matches a bounced payment. Be strict: a candidate only matches if "
    "it actually contains what the query asks for. Exclude guides, articles "
    "and templates that merely discuss the topic, unless the query asks for "
    "one. It is common that NONE match.\n"
    "Reply with ONLY the matching numbers separated by commas, or NONE."
)

# Stage 2 — verification of a single pick against the actual passage. A
# binary entailment question over concrete text is the task a small on-device
# model is actually reliable at; the open-ended selection above is not.
_VERIFY_PROMPT = (
    'Query: "{query}"\n'
    "\n"
    'Candidate file "{filename}" (document type: {kind}):\n'
    "{text}\n"
    "\n"
    "Would this file satisfy the user's query?\n"
    "- Match on meaning, not wording: a CV satisfies a resume query; a "
    "dishonoured direct debit satisfies a bounced-payment query.\n"
    "- A query about an EVENT (a missed delivery, a failed payment) is "
    "satisfied by any document recording that event.\n"
    "- A query naming a FORM of document (a conversation, a receipt, a photo) "
    "is satisfied only by that form — a tender, report or letter about the "
    "same topic does NOT count.\n"
    "- A guide, article or template about the general topic never satisfies "
    "the query.\n"
    "Reply with ONLY one word: YES or NO."
    # Phrasing A/B-tested greedy against 8 known true/false pairs (7/8; the
    # miss is a tolerable false positive). The traps each rule closes: "is
    # this what the query asks for" → hyper-literal NO on a letter recording
    # a failed payment; no form rule → tender docs satisfy a query about
    # conversations; form rule without the event rule → "a payment" read as
    # a form and the failed-payment letter rejected again.
)


def available() -> bool:
    """Judge is usable: AFM present, not disabled, not in test mode."""
    if os.environ.get("FINDANDSEEK_TEST") == "1":
        return False
    if os.environ.get("FINDANDSEEK_SEARCH_AFM_JUDGE", "1") == "0":
        return False
    from find_and_seek.config.settings import _afm_available

    return _afm_available()


def complete(prompt: str, max_tokens: int = 64, timeout_s: float = _TIMEOUT_S) -> str | None:
    """One AFM completion in a dedicated thread with its own event loop, so the
    call works identically from sync callers and from inside a running loop
    (worker, API, MCP). On timeout we abandon the thread rather than block.

    GREEDY sampling: search-side AFM calls must be deterministic — with
    default sampling the same verification question flipped YES/NO across
    runs. (Also reused by query expansion; deterministic variants make
    search results reproducible for a given query and index.)"""

    def _run() -> str:
        import apple_fm_sdk as fm

        session = fm.LanguageModelSession(model=fm.SystemLanguageModel())
        opts = fm.GenerationOptions(
            sampling=fm.SamplingMode(fm.SamplingModeType.GREEDY),
            maximum_response_tokens=max_tokens,
        )
        return asyncio.run(session.respond(prompt, options=opts))

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(_run).result(timeout=timeout_s)
    except Exception as e:  # noqa: BLE001 — best-effort, never break search
        logger.debug("AFM completion unavailable/timed out: %s", e)
        return None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _respond(prompt: str, timeout_s: float) -> str | None:
    return complete(prompt, max_tokens=64, timeout_s=timeout_s)


def judge(query: str, candidates: list[tuple[int, str]]) -> list[int] | None:
    """Ask AFM which candidates genuinely answer `query`.

    `candidates` is (key, display_text) — keys are returned for the selected
    entries. Returns [] when AFM says NONE, or None when there is NO VERDICT
    (unavailable, timeout, unparseable) — the caller must treat None as
    "judge didn't run", never as "nothing matches"."""
    if not candidates:
        return []
    numbered = "\n".join(f"{i + 1}. {text}" for i, (_, text) in enumerate(candidates))
    raw = _respond(_PROMPT.format(query=query, numbered=numbered), _TIMEOUT_S)
    if raw is None:
        return None
    reply = raw.strip()
    if re.search(r"\bnone\b", reply, re.IGNORECASE):
        return []
    picks = {int(n) for n in re.findall(r"\d+", reply) if 1 <= int(n) <= len(candidates)}
    if not picks:
        return None  # garbage reply — no verdict
    return [key for i, (key, _) in enumerate(candidates) if i + 1 in picks]


def verify(query: str, filename: str, kind: str, text: str) -> bool:
    """Stage-2 check of one selected candidate against its actual passage.
    Conservative by construction: no verdict (unavailable, timeout, garbage)
    counts as NO — an unverified pick must never reach the user."""
    raw = _respond(
        _VERIFY_PROMPT.format(query=query, filename=filename, kind=kind, text=text[:800]),
        _TIMEOUT_S,
    )
    if raw is None:
        return False
    reply = raw.strip().upper()
    return bool(re.match(r"\W*YES\b", reply))
