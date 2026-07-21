"""Document summarisation via in-process MLX (Qwen3), with a heuristic fallback."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from find_and_seek.ingest import card_guard
from find_and_seek.ingest.chunk import Chunk
from find_and_seek.organize.taxonomy import canonicalize_type

logger = logging.getLogger(__name__)


# Lines that carry no decision-useful signal — letterhead, page furniture, and
# boilerplate. When the heuristic must pick an anchor (LLM unavailable/failed) we
# skip these so the anchor isn't a phone number or a confidentiality notice.
_BOILERPLATE_RE = re.compile(
    r"(confidential|all rights reserved|copyright|©|page \d+|www\.|https?://|"
    r"@[\w.-]+\.\w+|\+?\d[\d\s().-]{7,}\d|acknowledg\w* (the|country|traditional)|"
    r"^\s*(to|from|date|subject|cc|bcc|re|attn)\s*:)",
    re.I,
)


def _pick_anchor_line(chunks: list[Chunk]) -> str:
    """First line with real content — skips letterhead/contact/boilerplate so the
    fallback anchor reads as the document's subject, not its footer."""
    for c in chunks[:3]:
        for line in c.text.splitlines():
            line = line.strip()
            # Need a few words of actual prose; skip furniture and bare numbers.
            if len(re.findall(r"[A-Za-z]{2,}", line)) < 3:
                continue
            if _BOILERPLATE_RE.search(line):
                continue
            return line[:200]
    # Nothing clean found — fall back to the very first non-empty line.
    for c in chunks[:3]:
        for line in c.text.splitlines():
            if line.strip():
                return line.strip()[:200]
    return ""


def _heuristic_summary(chunks: list[Chunk], filename: str) -> dict[str, Any]:
    text = " ".join(c.text for c in chunks[:5])[:3000]
    first_line = _pick_anchor_line(chunks) if chunks else text[:200]
    doc_type = "other"
    # Type detection from CONTENT ONLY — filename deliberately excluded here.
    # The filename is often a template name, slug, or working title that does not
    # reflect the document's actual kind (e.g. "cover-letter-template.docx" that
    # contains a contract body). The heuristic fallback is a last resort; it must
    # not magnify filename bias when the LLM is unavailable.
    content_lower = text.lower()
    if "invoice" in content_lower:
        doc_type = "invoice"
    elif "contract" in content_lower:
        doc_type = "contract"
    elif "letter" in content_lower:
        doc_type = "letter"
    elif "report" in content_lower:
        doc_type = "report"

    key_facts: dict[str, Any] = {}
    # Decimals of any length + magnitude suffixes, so "$1.2 billion" is captured
    # whole rather than as the useless "$1" (2026-07-17 bench artifact).
    money = re.search(
        r"[\$£€]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|trillion|bn|mn|[kmb]))?\b",
        text,
        re.I,
    )
    if money:
        key_facts["amount"] = money.group()
    date = re.search(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        text,
        re.I,
    )
    if date:
        key_facts["date"] = date.group()

    words = re.findall(r"\w+", first_line)
    anchor = " ".join(words[:25])

    return {
        "summary_text": first_line[:500],
        "one_line_anchor": anchor,
        "document_type": canonicalize_type(doc_type),
        "key_facts": key_facts,
        "suggested_filename": filename,
        # Degraded output must SAY it is degraded (2026-07-17 bench: three
        # separate silent-fallback incidents in one day). fast_classify
        # overwrites this with "fast-path" for its confident deterministic cards.
        "confidence_note": "degraded: heuristic fallback",
    }


# ── Cascade fast-path (skip the LLM for confidently-typed files) ─────────
# The summariser (LLM) is ~88% of ingest and prefill-bound. A large fraction of
# files have an unambiguous type from their container or a header keyword — those
# need no 3B at all, and a deterministic label is *more* reliable than an LLM
# guess. fast_classify handles those; it returns None for genuinely ambiguous
# files, which still go to the LLM. Anchors/key_facts reuse the heuristic path.

# Container format → document_type, when the format alone fixes the type.
# NOTE: spreadsheets are deliberately NOT here. The deterministic heuristic anchor
# for tabular data is a column-name dump ("CSV with columns name product_id …"),
# which is exactly the useless anchor users complained about. Routing them to the
# LLM yields a readable subject line ("Apparel product catalogue with prices");
# the *type* is still set deterministically downstream by classify_document, so we
# lose nothing by deferring the summary. This matches the re-summarise path
# (refresh.resummarize), which never fast-classifies.
_FORMAT_DOCTYPE = {
    "email": "email",
    "presentation": "slide deck",
}

# (keyword, document_type) — first hit wins. Matched against the content title
# line. These are only fired when the document's OWN first line contains the
# keyword, so they reflect what the document says about itself, not what a
# filing clerk named the file.
_HEADER_SIGNALS: tuple[tuple[str, str], ...] = (
    ("invoice", "invoice"),
    ("purchase order", "invoice"),
    ("receipt", "receipt"),
    ("curriculum vitae", "cv"),
    ("non-disclosure", "contract"),
    ("terms and conditions", "contract"),
    ("agreement", "contract"),
    ("contract", "contract"),
)

# A tighter subset of keywords reliable enough to use AS A TIEBREAKER on the
# filename ALONE — only when the title line produces no hit. These must be
# precise enough that seeing the word in a filename is nearly conclusive
# evidence regardless of content (e.g. "invoice_2024.pdf" is almost always an
# invoice). Deliberately excludes generic words like "letter", "report",
# "agreement", "contract" — those appear in template/working filenames far too
# often to be trustworthy without content confirmation.
_FNAME_ONLY_SIGNALS: frozenset[str] = frozenset({"invoice", "receipt", "curriculum vitae"})


def fast_classify(
    file_type: str, chunks: list[Chunk], filename: str
) -> dict[str, Any] | None:
    """Confidently summarise an easy-to-type file WITHOUT the LLM, or return None
    to defer to it. Conservative by design — precision over coverage.

    This is the ingest fast-path: the LLM summary stage is ~88% of ingest, and a
    large fraction of files are self-describing from their container (`.eml`,
    `.pptx`) or first-line header ("INVOICE", "CONTRACT OF EMPLOYMENT"). For
    those a deterministic label is both faster (no 3B forward pass) and *more*
    reliable than the LLM. Everything ambiguous returns None and defers to the
    LLM. Filename bias is deliberately resisted (see `_FNAME_ONLY_SIGNALS`);
    the precision guarantees are pinned by `eval/classify_eval.py --test-bias`
    (cases P-12a–e) and `tests/test_improvements.py`."""
    doc_type = _FORMAT_DOCTYPE.get(file_type)
    fname = filename.lower()
    # .csv/.tsv are tabular too — defer to the LLM for the same reason (see
    # _FORMAT_DOCTYPE), rather than emitting a column-name dump.
    if doc_type is None and fname.endswith((".csv", ".tsv")):
        return None
    if doc_type is None:
        # Content title is the primary signal: a doc whose first line IS "RECEIPT"
        # or "INVOICE" is self-describing and unambiguous.
        # Filename is a secondary tiebreaker, but only for high-precision keywords
        # in _FNAME_ONLY_SIGNALS — words like "agreement", "contract", "letter"
        # are excluded because they appear constantly in template/working filenames
        # (e.g. "cover-letter-template.docx", "partnership-agreement-draft.pdf")
        # and would misclassify documents whose content says otherwise.
        lines = chunks[0].text.strip().splitlines() if chunks else []
        title = (lines[0] if lines else "")[:120].lower()
        title_hit: str | None = None
        for kw, dt in _HEADER_SIGNALS:
            if kw in title:
                title_hit = dt
                break
        if title_hit is not None:
            doc_type = title_hit
        else:
            # No title hit — try filename, but only the high-precision subset.
            for kw, dt in _HEADER_SIGNALS:
                if kw in _FNAME_ONLY_SIGNALS and kw in fname:
                    doc_type = dt
                    break
    if doc_type is None:
        return None
    summ = _heuristic_summary(chunks, filename)
    summ["document_type"] = canonicalize_type(doc_type)
    summ["confidence_note"] = "fast-path"
    return summ


# document_type is constrained to the spec §4.3 enum so the label is consistent
# across models/backends (otherwise models free-form "excel sheet" vs "spreadsheet",
# "legal contract" vs "contract" — which breaks type_filter matching downstream).
DOCUMENT_TYPES = (
    "invoice, contract, report, letter, receipt, cv, spreadsheet, "
    "slide deck, note, product, email, other"
)
SYSTEM_PROMPT = (
    "You summarise documents for a local search index. "
    "Return ONLY valid JSON with keys: summary_text, one_line_anchor (<=25 words), "
    "document_type, key_facts (object), suggested_filename, confidence_note (or null). "
    "summary_text length must scale to the SOURCE, not to a fixed target: a short "
    "document (roughly under 300 characters — a single sentence, a one-line policy, "
    "a 3-row CSV) gets a summary_text that is no longer than the source itself, "
    "reusing its own wording rather than restating it in generic framing language "
    "('This document outlines...', 'This spreadsheet details...', 'This document "
    "appears to be...'). Only longer source documents (multi-paragraph, multi-page) "
    "earn a multi-sentence summary_text (2-3 sentences, never more — output length "
    "is the main speed cost) — the length must be earned "
    "by actual content, never padded to hit a sentence count. "
    f"document_type MUST be exactly one of: {DOCUMENT_TYPES}. "
    "Use 'other' when the document does not clearly fit a type — do NOT force a "
    "fit. An 'invoice' is a bill that requests or confirms a specific payment "
    "(it has an amount due, an invoice/account number, or a 'bill to'); a product "
    "list, price catalogue, sales/inventory spreadsheet, listing, or general "
    "payment record is NOT an invoice. Tabular data is 'spreadsheet'. "
    "one_line_anchor must be the single most decision-useful line of THIS document — "
    "never a letterhead, confidentiality notice, page header, land acknowledgement, or "
    "boilerplate. LEAD with the decisive fact: when the document's PURPOSE is a payment "
    "(invoice, receipt, bill, account statement) that is the MONEY — the total or amount due/"
    "paid with the party/purpose (e.g. 'MRI claim — $562.73 benefit paid'); otherwise lead with "
    "what the document actually is and NEVER put '$0', 'no amount', or a price in the anchor of a "
    "non-payment document; for data/spreadsheets, say in plain words WHAT the data is about (its "
    "real-world subject, inferred from the values and filename) — e.g. 'Apparel product catalogue "
    "with prices and sizes'; do NOT list the column names, and do NOT state a total row count (the "
    "sample is partial). "
    "key_facts: ONLY facts actually stated in the document — never invent, infer, or assume the "
    "reader's identity; omit anything not present (an empty object is fine); NEVER write "
    "placeholder values ('not specified', 'N/A', 'TBD', 'xxxxx', redactions) — drop the key. "
    "Be EXHAUSTIVE: extract every distinct fact stated (every name, date, amount, id, party, "
    "location) as its own key — do not stop after the first one you notice. "
    "Use natural keys "
    "(amount, total, due, date, party, id, ref) — never prefix with 'total_'. Include 'amount'/"
    "'total' ONLY when the document genuinely involves a payment; do NOT add amount:0 or 'N/A' "
    "to non-financial documents. "
    "No markdown, no thinking. "
    # Typed facets ride as an addendum so the core prompt stays stable; the
    # model files each distinct structure separately, which is what makes the
    # observed conflation class (escrow welded onto earn-out) structurally hard.
    + card_guard.FACET_PROMPT
)


def select_summary_text(chunks: list[Chunk], budget: int) -> str:
    """Stratified selection of model-visible text within ``budget`` chars.

    Head-only feeding capped long documents at their first pages — the
    2026-07-17 bench produced a 149-page Hansard card describing only the
    member roll, because that is literally all any model ever saw. Short
    documents keep the old behaviour (whole head, best quality). Long ones
    split the budget: half to the head (type/anchor/key facts live there —
    see SUMMARY_MAX_CHARS rationale in settings), half spread over evenly
    spaced middle samples plus the tail. Non-adjacent samples are joined with
    an explicit gap marker so the model cannot read false continuity across
    them."""
    if not chunks:
        return ""
    head = "\n\n".join(c.text for c in chunks[:12])
    if len(head) <= budget or len(chunks) <= 2:
        return head[:budget]

    head_part = head[: budget // 2]
    rest = budget - len(head_part)
    # Sample up to 4 later chunks: evenly spaced through the middle + the tail.
    n_samples = min(4, len(chunks) - 1)
    span = len(chunks) - 1
    idxs = sorted({1 + round(i * (span - 1) / max(1, n_samples - 1)) for i in range(n_samples)})
    idxs = [i for i in idxs if i < len(chunks)] or [len(chunks) - 1]
    per = max(200, rest // len(idxs))
    parts = [head_part] + [chunks[i].text[:per] for i in idxs]
    return "\n\n[…]\n\n".join(p for p in parts if p.strip())[:budget + 64]


def build_messages(chunks: list[Chunk], filename: str) -> list[dict[str, str]]:
    """Prompt shared by the MLX single and batched summary paths."""
    from find_and_seek.config.settings import SUMMARY_MAX_CHARS

    text = select_summary_text(chunks, SUMMARY_MAX_CHARS)
    user = f"Filename: {filename}\n\nContent:\n{text}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# Placeholder values that carry no information — often template fields the model
# faithfully extracts ("registration_no": "xxxxx"). Dropped deterministically.
_PLACEHOLDERS = {
    "", "n/a", "na", "n.a.", "not specified", "not stated", "not provided", "tbd",
    "unknown", "none", "null", "nil", "-", "—", "redacted", "[redacted]", "xxx",
    "xxxx", "xxxxx", "xxxxxx", "...", "todo", "tba",
}

# Tabular-dump artifacts the model emits for spreadsheets: a literal column list,
# or per-sample-row keys ("Sample Row 2", "row 4"). These echo the table's
# structure, not a decision-useful fact — drop them so key_facts stays meaningful.
_TABLE_DUMP_KEYS = {"columns", "column_names", "column names", "headers", "fields"}
_ROW_KEY_RE = re.compile(r"^(sample\s+)?row\s*\d+$", re.I)


def _clean_facts(kf: Any) -> dict[str, Any]:
    """Drop fact entries whose value is an empty/placeholder/redaction token, the
    spreadsheet column/row-dump artifacts, and leading 'total_' key prefixes the
    model sometimes adds."""
    if not isinstance(kf, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in kf.items():
        if isinstance(v, str):
            s = v.strip().lower().rstrip(".")
            if s in _PLACEHOLDERS or set(s) <= {"x"} and len(s) >= 3:
                continue
        key = str(k)
        klow = key.strip().lower()
        if klow in _TABLE_DUMP_KEYS or _ROW_KEY_RE.match(klow):
            continue
        if key.startswith("total_") and key not in ("total_rows", "total"):
            key = key[len("total_"):]
        out[key] = v
    return out


def _repair_truncated_json(s: str) -> dict[str, Any] | None:
    """Best-effort recovery of JSON truncated by the output-token cap. The lean
    batch path caps tokens, which can sever the string mid-value before the
    closing braces, losing an otherwise-good anchor/facts. Drop any dangling
    partial pair, close open quotes/brackets, and retry. Returns None if still
    unparseable — the caller then falls back to the heuristic."""
    def _balance(t: str) -> str:
        """Close any quote/array/object the string left open."""
        t = t.rstrip().rstrip(",")
        if t.count('"') % 2:
            t += '"'
        t += "]" * (t.count("[") - t.count("]"))
        t += "}" * (t.count("{") - t.count("}"))
        return t

    s = s.strip()
    # First try closing as-is: if the cut landed right after a complete pair, this
    # preserves it (e.g. a finished anchor with no trailing comma).
    try:
        data = json.loads(_balance(s))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    # Otherwise the tail is a partial key/value — trim back to the last value
    # boundary (a closed }/] or the comma after a completed pair) and retry.
    cut = max(s.rfind("}"), s.rfind("]"), s.rfind(","))
    if cut <= 0:
        return None
    try:
        data = json.loads(_balance(s[: cut + 1]))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def parse_summary(raw: str, filename: str) -> dict[str, Any] | None:
    """Shared JSON extraction/normalisation. Returns None if no JSON found."""
    raw = re.sub(r"<\|channel\|>thought[\s\S]*?<\|channel\|>final", "", raw, flags=re.I)
    raw = re.sub(r"<\|[^|]+\|>", "", raw)
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        # No closing brace at all — the cap severed it. Recover from the opening
        # brace onward rather than discarding the whole (possibly good) output.
        open_brace = raw.find("{")
        data = _repair_truncated_json(raw[open_brace:]) if open_brace >= 0 else None
        if data is None:
            return None
    else:
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            # Malformed or truncated JSON — try to repair before giving up so a
            # severed closing brace doesn't cost the whole summary.
            data = _repair_truncated_json(match.group())
            if data is None:
                return None
    if not isinstance(data, dict):
        return None
    # Typed facets (if the model emitted any) flatten into the flat key_facts
    # contract — the stored card shape is unchanged, downstream consumers see
    # ordinary key/value facts.
    facts = _clean_facts(data.get("key_facts"))
    for k, v in card_guard.flatten_facets(data).items():
        facts.setdefault(k, v)
    return {
        "summary_text": data.get("summary_text", ""),
        "one_line_anchor": data.get("one_line_anchor", "")[:200],
        # Canonicalise to the fixed taxonomy so type_filter matching is reliable:
        # models drift ("Invoice"/"invoice", "slide deck"/"slide_deck") and
        # free-form past the enum ("decision", "job_posting", "product_description")
        # — all of which would otherwise become their own un-filterable pseudo-types.
        "document_type": canonicalize_type(data.get("document_type")),
        "key_facts": _clean_facts(facts),
        "suggested_filename": _keep_extension(data.get("suggested_filename"), filename),
        # The model's self-reported confidence is discarded by policy — it is
        # miscalibrated by construction. finalize_card rebuilds the note from
        # checks code can prove (containment, coverage, fallback markers).
        "confidence_note": None,
    }


def _keep_extension(suggested: Any, filename: str) -> str:
    """A rename suggestion must keep the file's real extension — the bench
    caught models proposing '<doc>.json' for a PDF three separate times."""
    if not suggested or not isinstance(suggested, str):
        return filename
    ext = os.path.splitext(filename)[1]
    root, sext = os.path.splitext(suggested.strip())
    if not ext:
        return suggested.strip()
    return (root if sext else suggested.strip()) + ext


def _composite_overlay(chunks: list[Chunk]) -> dict[str, Any] | None:
    """If the chunks are a scanned bundle of unrelated documents, return an
    honest anchor/summary + per-page section_anchors to merge over the base
    summary. Never raises.

    Off by default: heuristic composite detection still mislabels some coherent
    documents (dictionaries, web listings, multi-page certificates) as bundles,
    which would *corrupt* their anchors. Opt in with FINDANDSEEK_COMPOSITE_ANCHORS=1
    only once detection is trustworthy enough for the target corpus."""
    if os.environ.get("FINDANDSEEK_COMPOSITE_ANCHORS", "0") != "1":
        return None
    try:
        from find_and_seek.ingest.sections import composite_summary

        return composite_summary(chunks)
    except Exception:  # noqa: BLE001 — composite handling must never break ingest
        return None


def summarise_files(items: list[tuple[list[Chunk], str]]) -> list[dict[str, Any]]:
    """Summarise a batch of (chunks, filename) in one shot. The MLX backend runs
    them through a single batched forward pass (big throughput win); other
    backends fall back to summarising one file at a time."""
    if not items:
        return []
    from find_and_seek.config.settings import role_backend

    results: list[dict[str, Any]] | None = None
    backend = role_backend("summary")
    if backend == "ollama":
        from find_and_seek.ingest import summarise_ollama

        try:
            results = summarise_ollama.summarise_batch(items)
        except Exception as e:  # noqa: BLE001
            logger.debug("Ollama batch summariser fallback: %s", e)
    elif backend == "vllm":
        from find_and_seek.ingest import summarise_vllm

        try:
            results = summarise_vllm.summarise_batch(items)
        except Exception as e:  # noqa: BLE001
            logger.debug("vLLM batch summariser fallback: %s", e)
    elif backend == "afm":
        from find_and_seek.ingest import summarise_afm

        try:
            results = summarise_afm.summarise_batch(items)
        except Exception as e:  # noqa: BLE001
            logger.debug("AFM batch summariser fallback: %s", e)
    elif backend == "mlx":
        from find_and_seek.ingest import summarise_mlx

        try:
            results = summarise_mlx.summarise_batch(items)
        except Exception as e:  # noqa: BLE001
            logger.debug("MLX batch summariser fallback: %s", e)
    if results is None:
        results = [summarise_file(chunks, filename) for chunks, filename in items]

    # Composite (multi-document) files: replace the page-1-only anchor with an
    # honest bundle summary + per-page section anchors (deterministic, no model).
    for i, (chunks, _fn) in enumerate(items):
        overlay = _composite_overlay(chunks)
        if overlay:
            results[i] = {**results[i], **overlay}

    # Mechanical trust pass — every card leaves through this choke point
    # regardless of backend: containment-verify the facts, note window
    # coverage, keep only code-set degradation markers.
    from find_and_seek.config.settings import SUMMARY_MAX_CHARS

    for i, (chunks, _fn) in enumerate(items):
        source = " ".join(c.text for c in chunks)
        results[i] = card_guard.finalize_card(results[i], source, SUMMARY_MAX_CHARS)
    return results


def summarise_file(chunks: list[Chunk], filename: str, model: str | None = None) -> dict[str, Any]:
    if not chunks:
        return _heuristic_summary(chunks, filename)

    from find_and_seek.config.settings import role_backend

    _backend = role_backend("summary")
    if _backend == "test":
        base = _heuristic_summary(chunks, filename)
    elif _backend == "afm":
        from find_and_seek.ingest import summarise_afm

        try:
            base = summarise_afm.summarise_file(chunks, filename)
        except Exception as e:  # noqa: BLE001
            logger.debug("AFM summariser fallback: %s", e)
            base = _heuristic_summary(chunks, filename)
    elif _backend == "ollama":
        from find_and_seek.ingest import summarise_ollama

        try:
            base = summarise_ollama.summarise_file(chunks, filename)
        except Exception as e:  # noqa: BLE001
            logger.debug("Ollama summariser fallback: %s", e)
            base = _heuristic_summary(chunks, filename)
    elif _backend == "vllm":
        from find_and_seek.ingest import summarise_vllm

        try:
            base = summarise_vllm.summarise_file(chunks, filename)
        except Exception as e:  # noqa: BLE001
            logger.debug("vLLM summariser fallback: %s", e)
            base = _heuristic_summary(chunks, filename)
    else:
        # MLX in-process (the production backend, and the default for any non-test
        # value); heuristic fallback if the model can't load or emits no JSON.
        from find_and_seek.ingest import summarise_mlx

        try:
            base = summarise_mlx.summarise_file(chunks, filename)
        except Exception as e:  # noqa: BLE001
            logger.debug("MLX summariser fallback: %s", e)
            base = _heuristic_summary(chunks, filename)

    overlay = _composite_overlay(chunks)
    if overlay:
        base = {**base, **overlay}
    from find_and_seek.config.settings import SUMMARY_MAX_CHARS

    return card_guard.finalize_card(
        base, " ".join(c.text for c in chunks), SUMMARY_MAX_CHARS
    )
