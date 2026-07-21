"""Typed-fact extraction (HANDOVER.md #1).

Turns the opaque ``key_facts`` JSON the summariser produces, plus the NER
entities, into normalized, citable rows for the ``facts`` table: money becomes a
number + currency, dates become ISO 8601, quantities become numbers — each with
``chunk_id`` provenance where known. Pure functions; the DB write lives in
``db.store.write_facts``.
"""

from __future__ import annotations

import os
import re
from datetime import date
from typing import Any

# ── money ────────────────────────────────────────────────────────────
_CURRENCY_CODE = {"USD", "GBP", "EUR", "AUD", "NZD", "CAD", "JPY"}

# The bare ``$`` symbol is ambiguous (USD/AUD/NZD/CAD/...). This deployment
# is Australian, so a bare ``$`` defaults to AUD. Override per-deployment with
# ``FINDANDSEEK_DEFAULT_CURRENCY`` (validated against ``_CURRENCY_CODE``); any
# unknown value falls back to the AUD default. Read once at import so
# ``parse_money`` stays a pure function. Unambiguous symbols (£ € ¥) and any
# explicit 3-letter code in the text remain authoritative.
_DEFAULT_AMBIGUOUS_CURRENCY = "AUD"


def _resolve_default_currency() -> str:
    val = (os.environ.get("FINDANDSEEK_DEFAULT_CURRENCY") or "").strip().upper()
    return val if val in _CURRENCY_CODE else _DEFAULT_AMBIGUOUS_CURRENCY


_DEFAULT_CURRENCY = _resolve_default_currency()
# Unambiguous symbols keep their fixed currency; the ambiguous ``$`` uses the
# configured default.
_CURRENCY_SYMBOL = {"$": _DEFAULT_CURRENCY, "£": "GBP", "€": "EUR", "¥": "JPY"}
_MONEY_RE = re.compile(
    r"(?P<sym>[\$£€¥])?\s?(?P<num>\d[\d,]*(?:\.\d+)?)\s?(?P<code>[A-Z]{3})?"
)


def parse_money(s: str) -> tuple[float, str | None] | None:
    """``"$1,234.50"`` → ``(1234.5, "AUD")`` (bare ``$`` uses the configured
    default, see ``_DEFAULT_CURRENCY``); ``"500 AUD"`` → ``(500.0, "AUD")``;
    ``"£5"`` → ``(5.0, "GBP")``. An explicit 3-letter code always wins."""
    if not s:
        return None
    m = _MONEY_RE.search(s)
    if not m:
        return None
    sym, num, code = m.group("sym"), m.group("num"), m.group("code")
    # Require some currency signal so we don't treat every bare number as money.
    if not sym and not (code and code in _CURRENCY_CODE):
        return None
    try:
        value = float(num.replace(",", ""))
    except ValueError:
        return None
    currency = (code if code in _CURRENCY_CODE else None) or _CURRENCY_SYMBOL.get(sym or "")
    return value, currency


# ── numbers ──────────────────────────────────────────────────────────
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = _NUM_RE.fullmatch(value.strip())
        if m:
            try:
                return float(m.group().replace(",", ""))
            except ValueError:
                return None
    return None


# ── dates ────────────────────────────────────────────────────────────
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_ISO_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
_DMY_RE = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b")
_MONTH_DAY_YEAR_RE = re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b")
_DAY_MONTH_YEAR_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b")


def _safe_date(y: int, m: int, d: int) -> str | None:
    if y < 100:
        y += 2000
    try:
        return date(y, m, d).isoformat()
    except ValueError:
        return None


def parse_date(s: str) -> str | None:
    """Best-effort → ISO ``YYYY-MM-DD``. Day/month ambiguity resolves to
    **day-first** (AU/UK convention) unless the first field is > 12.
    Returns ``None`` for anything it can't pin to a real calendar date."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()

    m = _ISO_RE.search(s)
    if m:
        return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = _DMY_RE.search(s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a > 12 and b <= 12:        # unambiguous day-first
            return _safe_date(y, b, a)
        if b > 12 and a <= 12:        # unambiguous month-first
            return _safe_date(y, a, b)
        return _safe_date(y, b, a)    # ambiguous → day-first

    m = _MONTH_DAY_YEAR_RE.search(s)  # "August 31, 2022"
    if m and m.group(1).lower() in _MONTHS:
        return _safe_date(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))

    m = _DAY_MONTH_YEAR_RE.search(s)  # "31 August 2022"
    if m and m.group(2).lower() in _MONTHS:
        return _safe_date(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))

    return None


# ── fact assembly ────────────────────────────────────────────────────
_NAME_WORD = re.compile(r"[A-Za-z]{2,}")


def _looks_like_name(val: str) -> bool:
    """A person/org/location should contain a real multi-letter word and not be
    dominated by digits — filters NER noise (phone fragments, IDs, stray glyphs)."""
    if len(val) < 2 or not _NAME_WORD.search(val):
        return False
    digits = sum(c.isdigit() for c in val)
    return digits <= len(val) / 2


_DATEISH_KEY = re.compile(r"date|hearing|due|issued|expir|deadline|dob|born", re.I)
_MONEYISH_KEY = re.compile(r"amount|total|price|cost|fee|balance|salary|value|paid", re.I)
_MAX_TEXT = 500
_MAX_ENTITY_FACTS_PER_TYPE = 500


def _fact(
    fact_type: str,
    *,
    key: str | None = None,
    value_text: str | None = None,
    value_number: float | None = None,
    value_date: str | None = None,
    unit: str | None = None,
    confidence: float | None = None,
    source: str = "key_facts",
    chunk_id: int | None = None,
) -> dict[str, Any]:
    return {
        "fact_type": fact_type,
        "key": key,
        "value_text": (value_text[:_MAX_TEXT] if value_text else value_text),
        "value_number": value_number,
        "value_date": value_date,
        "unit": unit,
        "confidence": confidence,
        "source": source,
        "chunk_id": chunk_id,
    }


def _facts_from_value(key: str, value: Any, out: list[dict[str, Any]], depth: int = 0) -> None:
    if value is None or value == "":
        return
    # Scalars
    if isinstance(value, str):
        money = parse_money(value)
        if money is not None or _MONEYISH_KEY.search(key):
            if money is not None:
                out.append(_fact("money", key=key, value_text=value.strip(),
                                 value_number=money[0], unit=money[1], confidence=0.8))
                return
        iso = parse_date(value)
        if iso and (_DATEISH_KEY.search(key) or re.search(r"\d", value)):
            out.append(_fact("date", key=key, value_text=value.strip(),
                             value_date=iso, confidence=0.8))
            return
        num = parse_number(value)
        if num is not None:
            out.append(_fact("quantity", key=key, value_text=value.strip(),
                             value_number=num, confidence=0.8))
            return
        out.append(_fact("attribute", key=key, value_text=value.strip(), confidence=0.7))
        return
    if isinstance(value, bool):
        out.append(_fact("attribute", key=key, value_text=str(value), confidence=0.7))
        return
    if isinstance(value, (int, float)):
        out.append(_fact("quantity", key=key, value_number=float(value),
                         value_text=str(value), confidence=0.8))
        return
    # Containers
    if isinstance(value, list):
        scalars = [v for v in value if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
        if scalars:
            joined = "; ".join(str(v).strip() for v in scalars)
            out.append(_fact("attribute", key=key, value_text=joined, confidence=0.6))
        if depth == 0:
            for v in value:
                if isinstance(v, dict):
                    _facts_from_dict(v, out, depth + 1, key_prefix=key)
        return
    if isinstance(value, dict) and depth == 0:
        _facts_from_dict(value, out, depth + 1, key_prefix=key)


def _facts_from_dict(d: dict[str, Any], out: list[dict[str, Any]], depth: int = 0,
                     key_prefix: str = "") -> None:
    for k, v in d.items():
        key = f"{key_prefix}.{k}" if key_prefix else str(k)
        _facts_from_value(key, v, out, depth)


def _facts_from_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}
    for ent in entities:
        et = ent.get("entity_type")
        raw = (ent.get("entity_raw") or ent.get("entity_value") or "").strip()
        val = (ent.get("entity_value") or "").strip()
        cid = ent.get("chunk_id")
        if not et or not val:
            continue
        dedup_key = (et, val)
        if dedup_key in seen:
            continue
        if counts.get(et, 0) >= _MAX_ENTITY_FACTS_PER_TYPE:
            continue

        if et == "money":
            money = parse_money(raw)
            if money is None:
                continue
            f = _fact("money", key="money", value_text=raw, value_number=money[0],
                      unit=money[1], confidence=0.6, source="ner", chunk_id=cid)
        elif et == "date":
            iso = parse_date(raw)
            if not iso:
                continue
            f = _fact("date", key="date", value_text=raw, value_date=iso,
                      confidence=0.6, source="ner", chunk_id=cid)
        elif et in ("person", "org", "location"):
            # Quality gate: NER mislabels stray tokens (digits, single letters,
            # punctuation) as names/places. Require a real word so junk like a
            # phone fragment or "#" doesn't become a person/location fact.
            if not _looks_like_name(val):
                continue
            f = _fact(et, key=et, value_text=val, confidence=0.6, source="ner", chunk_id=cid)
        elif et in ("email", "phone", "ref_number"):
            f = _fact(et, key=et, value_text=val, confidence=0.6, source="ner", chunk_id=cid)
        else:
            continue

        seen.add(dedup_key)
        counts[et] = counts.get(et, 0) + 1
        out.append(f)
    return out


def extract_facts(
    summary: dict[str, Any] | None,
    entities: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Normalize a file's ``key_facts`` + NER entities into typed fact rows."""
    out: list[dict[str, Any]] = []
    if summary:
        key_facts = summary.get("key_facts")
        if isinstance(key_facts, dict):
            _facts_from_dict(key_facts, out)
    if entities:
        out.extend(_facts_from_entities(entities))
    return out
