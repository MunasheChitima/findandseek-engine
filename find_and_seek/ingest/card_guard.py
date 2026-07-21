"""Typed facets + mechanical trust layer for summariser cards.

Two jobs, both born from the 2026-07-17 adversarial bench (the "seabird" memo,
now in the golden set):

1. **Facets** — documents aren't types, they're combinations of a small set of
   facets (money terms, people, dates, conditions, references, metrics,
   contacts). The prompt asks the model to file each structure as its OWN entry
   in a typed list; ``flatten_facets`` then folds them into the flat
   ``key_facts`` contract. This makes the observed conflation class (escrow
   welded onto the earn-out) structurally hard: two structures cannot share one
   list entry.

2. **Mechanical confidence** — a model's self-reported confidence is
   miscalibrated by construction (it would have been confident about the
   conflation too), so the model's own ``confidence_note`` is DISCARDED and the
   note is rebuilt here from checks code can actually prove: which fact values
   appear verbatim in the source (containment), how much of the document the
   summary window covered, and whether the card is a degraded fallback. A note
   that can't bluff is the trust story.
"""

from __future__ import annotations

import re
from typing import Any

# ── Facet specs ──────────────────────────────────────────────────────────
# One spec per facet: (json key, entry fields, prompt clause). The prompt text
# is assembled from this table so adding a facet is one row, not a rewrite.
# Fields are all optional strings — validation tolerates partial entries and
# discards non-dict garbage rather than failing the whole card.
FACETS: dict[str, tuple[tuple[str, ...], str]] = {
    "money_terms": (
        ("label", "amount", "percentage", "duration", "party"),
        "one entry per DISTINCT financial structure (price, escrow, earn-out, "
        "deposit, fee, penalty) — NEVER merge two structures' numbers into one "
        "entry; if a figure was revised, the operative value goes in 'amount' "
        "and the superseded one in a separate entry labelled as superseded",
    ),
    "people": (
        ("name", "role", "org", "stance"),
        "one entry per person; keep similarly-named people as separate entries "
        "with their own role/org/stance — never blend two people",
    ),
    "dates": (
        ("label", "date", "revised_from"),
        "one entry per deadline/date; when a date was moved, put the current "
        "date in 'date' and the old one in 'revised_from'",
    ),
    "conditions": (
        ("label", "requirement", "negated"),
        "one entry per condition/obligation; set negated='true' when the "
        "document says something is NOT required or does NOT apply",
    ),
    "references": (
        ("label", "value"),
        "identifiers only: reference numbers, case/invoice/registration ids",
    ),
    "metrics": (
        ("label", "value", "unit"),
        "measured quantities that are not money: percentages, counts, scores",
    ),
    "contacts": (
        ("name", "channel", "value"),
        "explicit contact details only (email, phone, address)",
    ),
}

FACET_PROMPT = (
    "Optionally include a top-level key 'facets' (object). Use it ONLY for facet "
    "kinds the document actually contains — omit it entirely for simple documents. "
    + " ".join(
        f"facets.{name}: list of objects with keys {', '.join(fields)} — {clause}."
        for name, (fields, clause) in FACETS.items()
    )
    + " Every facet field value must be text stated in the document; omit fields "
    "you cannot fill. Facts you place in facets do NOT need repeating in key_facts."
)


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return s[:40] or "item"


def flatten_facets(data: dict[str, Any]) -> dict[str, Any]:
    """Fold a raw ``facets`` object into flat key_facts entries.

    Tolerant by design: unknown facet names, non-list values, and non-dict
    entries are skipped — a malformed facet must never cost the card.
    """
    facets = data.get("facets")
    if not isinstance(facets, dict):
        return {}
    flat: dict[str, Any] = {}

    def put(key: str, value: str) -> None:
        base, n = key, 2
        while key in flat:                      # collision-safe: label reuse
            key = f"{base}_{n}"
            n += 1
        flat[key] = value

    for name, entries in facets.items():
        if name not in FACETS or not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            e = {k: str(v).strip() for k, v in entry.items() if v not in (None, "")}
            if name == "money_terms" and e.get("label"):
                for field in ("amount", "percentage", "duration"):
                    if e.get(field):
                        put(f"{_slug(e['label'])}_{field}", e[field])
                if e.get("party"):
                    put(f"{_slug(e['label'])}_party", e["party"])
            elif name == "people" and e.get("name"):
                bits = [e.get("role"), e.get("org")]
                desc = ", ".join(b for b in bits if b)
                if e.get("stance"):
                    desc = f"{desc} — {e['stance']}" if desc else e["stance"]
                put(_slug(e["name"]), desc or e["name"])
            elif name == "dates" and e.get("label") and e.get("date"):
                val = e["date"]
                if e.get("revised_from"):
                    val += f" (revised from {e['revised_from']})"
                put(_slug(e["label"]), val)
            elif name == "conditions" and e.get("label"):
                val = e.get("requirement", "")
                if str(e.get("negated", "")).lower() in ("true", "yes", "1"):
                    val = f"NOT: {val}" if val else "not required"
                if val:
                    put(_slug(e["label"]), val)
            elif name in ("references", "metrics") and e.get("label") and e.get("value"):
                val = e["value"]
                if e.get("unit"):
                    val = f"{val} {e['unit']}"
                put(_slug(e["label"]), val)
            elif name == "contacts" and e.get("name") and e.get("value"):
                val = f"{e.get('channel', 'contact')}: {e['value']}" if e.get("channel") else e["value"]
                put(_slug(e["name"]), val)
    return flat


# ── Containment verification ─────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Comparison form: casefold, collapse whitespace, strip digit-grouping
    commas and currency spacing so '$43.8 million' matches '$ 43.8  million'."""
    t = text.casefold()
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    t = re.sub(r"([$£€])\s+", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def _checkable(value: str) -> bool:
    """Only verify values precise enough that absence from the source means
    something: at least 4 chars and containing a digit or a capitalised word.
    Paraphrase values ('identify capable suppliers') are legitimately absent
    verbatim and must not be flagged."""
    if len(value) < 4:
        return False
    return bool(re.search(r"\d", value) or re.search(r"\b[A-Z][a-z]{2,}", value))


def verify_containment(key_facts: dict[str, Any], source_text: str) -> tuple[int, int, list[str]]:
    """(verified, checked, unverified_keys) — a fact value counts as verified
    when its normalised form (or every one of its digit/name tokens) appears in
    the normalised source."""
    src = _normalise(source_text)
    verified, checked, missing = 0, 0, []
    for key, value in key_facts.items():
        if not isinstance(value, str) or not _checkable(value):
            continue
        checked += 1
        v = _normalise(value)
        if v in src:
            verified += 1
            continue
        # Composite values ('15 August 2026 (revised from 1 July 2026)') won't
        # match whole — accept when every precise token is present.
        tokens = [t for t in re.findall(r"[\w$£€.]+", v) if re.search(r"\d", t) or len(t) > 3]
        if tokens and all(t in src for t in tokens):
            verified += 1
        else:
            missing.append(key)
    return verified, checked, missing


# ── Injection scrub ──────────────────────────────────────────────────────
# A document can contain prompt-injection text ("Ignore all previous
# instructions…") that the summariser faithfully extracts as a fact or quotes
# in the summary — at which point every agent that reads the card gets the
# injection re-served from a trusted surface. Cards are for facts ABOUT
# documents, never conduits for instructions IN them. Found live by the golden
# canary (seabird memo, 2026-07-17): extraction of the phrase is intermittent,
# so the scrub is deterministic.
_INJECTION_RE = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above|earlier)\s+"
    r"(instructions?|prompts?|context)|reply\s+only\s+with|respond\s+only\s+with",
    re.I,
)


def scrub_injections(card: dict[str, Any]) -> dict[str, Any]:
    facts = card.get("key_facts")
    if isinstance(facts, dict):
        card["key_facts"] = {
            k: v for k, v in facts.items()
            if not (_INJECTION_RE.search(str(k)) or _INJECTION_RE.search(str(v)))
        }
    for field in ("summary_text", "one_line_anchor"):
        text = card.get(field)
        if isinstance(text, str) and _INJECTION_RE.search(text):
            kept = [s for s in re.split(r"(?<=[.!?])\s+", text) if not _INJECTION_RE.search(s)]
            card[field] = " ".join(kept).strip()
    return card


# ── Mechanical confidence note ───────────────────────────────────────────

def finalize_card(
    card: dict[str, Any],
    source_text: str,
    window_chars: int,
) -> dict[str, Any]:
    """Rebuild ``confidence_note`` from provable checks. Existing MECHANICAL
    notes ('degraded: heuristic fallback', 'fast-path', 'needs-review') are
    kept as the leading clause — they were set by code, not by the model."""
    card = scrub_injections(card)
    notes: list[str] = []
    prior = card.get("confidence_note")
    if prior in ("degraded: heuristic fallback", "fast-path", "needs-review"):
        notes.append(prior)

    facts = card.get("key_facts") or {}
    if facts and source_text:
        verified, checked, missing = verify_containment(facts, source_text)
        if checked:
            notes.append(f"verified {verified}/{checked} facts in source")
            if missing:
                notes.append("unverified: " + ", ".join(sorted(missing)[:5]))

    total = len(source_text)
    if total > window_chars * 2:
        notes.append(f"summary window covered ~{max(1, round(100 * window_chars / total))}% of document")

    card["confidence_note"] = "; ".join(notes) if notes else None
    return card
