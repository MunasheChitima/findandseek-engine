"""MLX summariser backend — in-process LLM via mlx-lm (the production summariser).

Implements ``summarise.summarise_file`` and reuses its shared prompt/parse
helpers.

Unlike the embedding model (small, always resident), the summariser is a *heavy*
model, so managing its residency is on us: ``load()`` / ``unload()`` here
implement the §5.7 "never two heavy models resident" discipline, and
``active_memory_gb()`` lets us verify the unload actually frees memory.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from find_and_seek.ingest.chunk import Chunk
from find_and_seek.ingest.summarise import _heuristic_summary, build_messages, parse_summary

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_model_id: str | None = None
_load_seconds: float = 0.0


def active_memory_gb() -> float:
    """MLX active (allocated) memory in GB — for verifying load/unload frees RAM."""
    import mlx.core as mx

    getter = getattr(mx, "get_active_memory", None) or getattr(getattr(mx, "metal", None), "get_active_memory", None)
    return (getter() / (1024**3)) if getter else 0.0


def load() -> tuple:
    """Load the summariser into memory. Idempotent; cached process-wide."""
    global _model, _tokenizer, _model_id, _load_seconds
    if _model is not None:
        return _model, _tokenizer

    from mlx_lm import load as mlx_load

    from find_and_seek.config.models import mlx_summary_candidates

    explicit = os.environ.get("FINDANDSEEK_MLX_SUMMARY_MODEL")
    candidates = [explicit] if explicit else mlx_summary_candidates()

    last_err: Exception | None = None
    for repo in candidates:
        try:
            t0 = time.perf_counter()
            model, tokenizer = mlx_load(repo)
            _load_seconds = time.perf_counter() - t0
            _model, _tokenizer, _model_id = model, tokenizer, repo
            logger.info("MLX summary model loaded: %s (%.1fs)", repo, _load_seconds)
            return _model, _tokenizer
        except Exception as e:  # noqa: BLE001 — probe each candidate
            last_err = e
            logger.debug("MLX summary model %s failed: %s", repo, e)
    raise RuntimeError(f"No MLX summary model could be loaded; last error: {last_err}")


def unload() -> None:
    """Drop references and free MLX buffers — the heavy-model unload (§5.7)."""
    global _model, _tokenizer
    _model = None
    _tokenizer = None
    try:
        import gc

        import mlx.core as mx

        gc.collect()
        clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
        if clear:
            clear()
    except Exception:  # noqa: BLE001
        pass


def model_info() -> dict:
    return {"model_id": _model_id, "load_seconds": round(_load_seconds, 2)}


def _generate(messages: list[dict[str, str]], max_tokens: int = 1024) -> str:
    from mlx_lm import generate as mlx_generate

    model, tokenizer = load()
    # enable_thinking=False is REQUIRED for hybrid-thinking Qwen3 checkpoints
    # (the 1.7B low-RAM tier): their default <think> preamble breaks
    # parse_summary and silently degraded 8/9 docs to heuristic cards in the
    # 2026-07-17 bench. Non-thinking templates ignore the extra variable.
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )

    kwargs: dict[str, Any] = {"max_tokens": max_tokens}
    try:
        from mlx_lm.sample_utils import make_sampler

        kwargs["sampler"] = make_sampler(temp=0.2)
    except Exception:  # noqa: BLE001 — older mlx-lm without sampler API
        pass

    return mlx_generate(model, tokenizer, prompt=prompt, verbose=False, **kwargs)


def summarise_file(chunks: list[Chunk], filename: str) -> dict[str, Any]:
    if not chunks:
        return _heuristic_summary(chunks, filename)
    # Retry ONCE before degrading: the 2026-07-17 bench showed parse failures
    # are frequently transient (the same doc parsed cleanly on rerun). The
    # heuristic card is the floor, and it self-labels as degraded — but a
    # single retry recovers most of them for one generation's cost.
    for _attempt in range(2):
        raw = _generate(build_messages(chunks, filename))
        parsed = parse_summary(raw, filename)
        if parsed is not None:
            return parsed
    from find_and_seek import diagnostics

    diagnostics.record("summary_fallback", file=filename, backend="mlx")
    return _heuristic_summary(chunks, filename)


# ── Batched catalog pass ─────────────────────────────────────────────
# The throughput unlock: mlx_lm.batch_generate runs many documents through one
# forward pass (Apple Foundation Models can't — it serialises). A *lean* schema
# (no multi-sentence prose) keeps the output short, which is what dominates
# generation time. The full prose summary is produced lazily on demand instead.
LEAN_SYSTEM = (
    "You summarise documents for a local search index. Return ONLY compact JSON with keys: "
    "document_type (exactly one of: invoice, contract, report, letter, receipt, cv, "
    "spreadsheet, slide deck, note, email, other), one_line_anchor (<=20 words), "
    "key_facts (object, up to 5 entries). "
    "one_line_anchor = the ONE line telling the user what this is and why it matters; LEAD with "
    "the decisive fact. When the document's PURPOSE is a payment (invoice, receipt, bill, account "
    "statement) the decisive fact is the MONEY: the total or amount due/paid with the party/"
    "purpose — e.g. 'MRI claim — $562.73 benefit paid'. Otherwise lead with what the document "
    "actually is; NEVER put '$0', 'no amount', or a price in the anchor of a non-payment document. "
    "For data/spreadsheets, say in plain words WHAT the data is about (its real-world subject, "
    "inferred from the values and filename) — e.g. 'Apparel product catalogue with prices and "
    "sizes'. Do NOT list or name the columns, and do NOT state a total row count or 'N rows' "
    "(the sample is partial — you cannot know the total). Never a header, letterhead, or "
    "boilerplate. "
    "key_facts: ONLY facts actually stated in the document — never invent, infer, or assume the "
    "reader's identity. Omit anything not present (an empty object is fine); NEVER write "
    "placeholder values like 'not specified', 'N/A', 'TBD', 'xxxxx' or redactions — drop the key. "
    "Use natural keys "
    "(amount, total, due, date, party, id, ref) — never prefix keys with 'total_'. Include "
    "'amount'/'total' ONLY when the document genuinely involves a payment; do NOT add amount:0 or "
    "'N/A' to non-financial documents. "
    "No summary_text, no prose, no markdown, no thinking."
)


def _lean_messages(chunks: list[Chunk], filename: str) -> list[dict[str, str]]:
    from find_and_seek.config.settings import SUMMARY_MAX_CHARS

    text = "\n\n".join(c.text for c in chunks[:8])[:SUMMARY_MAX_CHARS]
    return [
        {"role": "system", "content": LEAN_SYSTEM},
        {"role": "user", "content": f"Filename: {filename}\n\nContent:\n{text}"},
    ]


def summarise_batch(
    items: list[tuple[list[Chunk], str]], max_tokens: int = 220
) -> list[dict[str, Any]]:
    # 96 was too tight: rich key_facts (amount/total/party/date) + a money-leading
    # anchor overflow it, truncating the JSON → a crude header-dump fallback (the
    # MRI invoice lost its $562.73). 220 lets the salient facts complete.
    """Summarise a whole batch in one MLX forward pass. ``items`` is a list of
    (chunks, filename). Returns one parsed summary dict per item, in order."""
    if not items:
        return []
    from mlx_lm import batch_generate

    model, tokenizer = load()
    prompts: list[list[int]] = []
    for chunks, filename in items:
        prompts.append(
            tokenizer.apply_chat_template(
                _lean_messages(chunks, filename),
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    resp = batch_generate(model, tokenizer, prompts=prompts, max_tokens=max_tokens, verbose=False)

    out: list[dict[str, Any]] = []
    for (chunks, filename), raw in zip(items, resp.texts):
        parsed = parse_summary(raw, filename)
        if parsed is None:
            # One per-file retry through the full (non-lean) path before the
            # degraded floor — parse failures are frequently transient.
            try:
                parsed = summarise_file(chunks, filename)
            except Exception:  # noqa: BLE001 — the floor must hold
                from find_and_seek import diagnostics

                diagnostics.record("summary_fallback", file=filename, backend="mlx-batch")
                parsed = _heuristic_summary(chunks, filename)
        # Lean schema omits summary_text; back-fill from the anchor so the summary
        # vector and preview still have content (full prose is generated on open).
        if not parsed.get("summary_text"):
            parsed["summary_text"] = parsed.get("one_line_anchor") or filename
        out.append(parsed)
    return out
