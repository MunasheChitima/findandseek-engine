"""Entity extraction.

Two-tier approach:
  1. spaCy (en_core_web_sm) — pattern-reliable types only: DATE, MONEY.
     PERSON/ORG/LOC are excluded because en_core_web_sm is semantically
     unreliable on degraded PDF text (food items labeled as people, etc.).
  2. key_facts extraction (extract_kf_entities) — Qwen3-4B already reads every
     document and extracts parties/respondents/clients reliably into key_facts.
     This is the SOLE authoritative source for person/org entities.
     Called separately by the worker after summary is computed.
     These carry source='key_facts'.
"""

from __future__ import annotations

import re
from typing import Any

from find_and_seek.ingest.chunk import Chunk

_nlp = None

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
REF_RE = re.compile(r"\b(?:INV|REF|PO|#)[-#]?\d{3,}\b", re.I)
# Full decimal precision — preserve $33.8143, not just $33.81.
MONEY_RE = re.compile(r"[\$£€]\s?\d[\d,]*(?:\.\d+)?")

# Only NER head needed; disabling the tagger/parser/lemmatizer roughly halves
# spaCy time with no effect on the mechanical entities we keep.
_NER_ONLY_DISABLE = ["tagger", "parser", "lemmatizer", "attribute_ruler"]

# spaCy label → our entity_type. PERSON/ORG/GPE/LOC intentionally absent —
# en_core_web_sm is not reliable enough for names on degraded PDF text.
# Those come exclusively from extract_kf_entities (key_facts from Qwen3-4B).
_LABEL_MAP = {
    "DATE": "date",
    "MONEY": "money",
}

_MAX_NER_CHARS = 100_000

# ── Quality gates ─────────────────────────────────────────────────────
_NAME_WORD = re.compile(r"[A-Za-z]{2,}")


def _looks_like_name(val: str) -> bool:
    """A person/org/location must contain a real multi-letter word and not be
    dominated by digits — filters NER noise (phone fragments, IDs, stray glyphs)."""
    if len(val) < 2 or not _NAME_WORD.search(val):
        return False
    digits = sum(c.isdigit() for c in val)
    return digits <= len(val) / 2


# ── key_facts entity patterns ─────────────────────────────────────────
# Keys that strongly suggest the value is a named org or person.
_ORG_KEYS = re.compile(
    r"\b(company|employer|vendor|supplier|contractor|respondent|client|firm|entity|"
    r"organisation|organization|provider|insurer|tenant|lessor|lessee|counterparty|"
    r"landlord|funder|licen[cs]ee?|franchisor|franchisee)\b",
    re.I,
)
_PERSON_KEYS = re.compile(
    r"\b(employee|person|payee|recipient|author|signatory|witness|applicant|claimant|"
    r"defendant|plaintiff|director|officer|trustee|guardian|nominee)\b",
    re.I,
)
# Surface forms that mark an org (appear in the VALUE, not the key).
_ORG_SUFFIX = re.compile(
    r"\b(pty\s+ltd|pty\.?\s*limited|ltd\.?|limited|inc\.?|corp\.?|llc|plc|"
    r"partners?|associates?|holdings?|group|foundation|trust|co\.)\b",
    re.I,
)


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        try:
            _nlp = spacy.load("en_core_web_sm", disable=_NER_ONLY_DISABLE)
        except OSError:
            from spacy.cli import download

            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm", disable=_NER_ONLY_DISABLE)
    return _nlp


def _add_entity(
    entities: list[dict[str, Any]],
    entity_type: str,
    value: str,
    raw: str,
    chunk_id: int | None = None,
    source: str = "ner",
) -> None:
    norm = value.strip().lower() if entity_type != "money" else value.strip()
    entities.append(
        {
            "entity_type": entity_type,
            "entity_value": norm,
            "entity_raw": raw,
            "chunk_id": chunk_id,
            "source": source,
        }
    )


def extract_entities(chunks: list[Chunk], chunk_ids: list[int] | None = None) -> list[dict[str, Any]]:
    """Extract mechanical entities (email/phone/money/ref + spaCy date/money)
    from chunks. person/org/location come from extract_kf_entities instead."""
    import os
    is_test = os.environ.get("FINDANDSEEK_TEST") == "1"
    entities: list[dict[str, Any]] = []

    if not is_test:
        nlp = _get_nlp()
        docs = list(nlp.pipe([c.text[:_MAX_NER_CHARS] for c in chunks], batch_size=32))
    else:
        docs = [None] * len(chunks)

    for i, chunk in enumerate(chunks):
        cid = chunk_ids[i] if chunk_ids and i < len(chunk_ids) else None
        text = chunk.text

        for m in EMAIL_RE.finditer(text):
            _add_entity(entities, "email", m.group(), m.group(), cid)
        for m in PHONE_RE.finditer(text):
            _add_entity(entities, "phone", m.group(), m.group(), cid)
        for m in REF_RE.finditer(text):
            _add_entity(entities, "ref_number", m.group(), m.group(), cid)
        for m in MONEY_RE.finditer(text):
            _add_entity(entities, "money", m.group(), m.group(), cid)

        if not is_test:
            doc = docs[i]
            if doc is not None:
                for ent in doc.ents:
                    et = _LABEL_MAP.get(ent.label_)
                    if not et:
                        continue
                    if et == "date":
                        from find_and_seek.ingest.facts import parse_date
                        if not parse_date(ent.text):
                            continue
                    _add_entity(entities, et, ent.text, ent.text, cid)

    return entities


# ── LLM-backed person/org extraction ─────────────────────────────────

def extract_kf_entities(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Mine person/org entities from the summariser's key_facts and anchor.

    Qwen3-4B already reads every document and correctly extracts parties into
    key_facts (e.g. "Respondent": "BIBIS WORLD PTY LTD"). This is more reliable
    than spaCy en_core_web_sm on degraded PDF text. Called by the worker after
    summary is computed so person/org entities come from the stronger model.
    """
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(et: str, val: str) -> None:
        norm = val.strip().lower()
        if norm in seen or not _looks_like_name(val):
            return
        seen.add(norm)
        entities.append({
            "entity_type": et,
            "entity_value": norm,
            "entity_raw": val.strip(),
            "chunk_id": None,
            "source": "key_facts",
        })

    key_facts = summary.get("key_facts") or {}
    for k, v in key_facts.items():
        if not isinstance(v, str) or not v.strip():
            continue
        val = v.strip()
        # Determine entity type: key pattern first, then value surface form.
        if _ORG_KEYS.search(str(k)) or _ORG_SUFFIX.search(val):
            _add("org", val)
        elif _PERSON_KEYS.search(str(k)):
            _add("person", val)
        # Generic "party" key — type from value surface form.
        elif re.search(r"\bparty\b", str(k), re.I):
            et = "org" if _ORG_SUFFIX.search(val) else "person"
            _add(et, val)

    # Also scan the one_line_anchor for party names the model surfaced there.
    anchor = summary.get("one_line_anchor") or ""
    # Pattern: "Pay Slip for Firstname Lastname" or "... for COMPANY NAME"
    m = re.search(r"\bfor\s+([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)+)\b", anchor)
    if m:
        val = m.group(1)
        et = "org" if _ORG_SUFFIX.search(val) else "person"
        _add(et, val)

    return entities
