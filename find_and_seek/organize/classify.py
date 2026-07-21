"""Deliberate document classification against the predefined taxonomy.

This is the "where does this belong?" decision, done on purpose:
  * The model is shown every category WITH its definition and exclusions, and
    asked to place the document — not to free-form a label.
  * The answer is VALIDATED against the active taxonomy. Anything outside the set
    is impossible to store: an out-of-set or unparseable answer becomes
    `needs-review`, never a junk type.
  * Low model confidence, or text too thin to judge, also routes to
    `needs-review` — honest uncertainty over a confident mistake.

Used by ingest (initial typing) and by reclassify (when the taxonomy changes), so
both paths make the same considered decision against the same categories.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

from find_and_seek.organize.categories import NEEDS_REVIEW, Category, load_taxonomy

logger = logging.getLogger(__name__)

# Below this many characters of real content, we can't responsibly judge — the
# extraction is too thin (image-only PDFs, blank scans). Route to review.
_MIN_CHARS = 40

# Confidence is DERIVED, not just self-reported. Small models are badly
# calibrated (the live index came back 84/89 "high", including documents whose
# body never extracted), so the model's claim is capped by a ceiling computed
# from how much of the document we could actually read. Hard signals are exempt:
# a .csv IS a spreadsheet regardless of text volume.
_CONF_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


def signal_ceiling(body_chars: int) -> str:
    """The highest confidence the extraction volume can honestly support."""
    if body_chars >= 500:
        return "high"
    if body_chars >= 120:
        return "medium"
    return "low"


def cap_confidence(model_conf: str, body_chars: int) -> str:
    ceiling = signal_ceiling(body_chars)
    if _CONF_ORDER.get(model_conf, 0) > _CONF_ORDER[ceiling]:
        return ceiling
    return model_conf

# Dedicated classifier model — decoupled from the summariser so summaries stay
# fast (the summary model stays the small/quick one). Default measured 11/12 on
# the ground-truth sample vs 7/12 for the 3B summariser. Override with
# FINDANDSEEK_MLX_CLASSIFY_MODEL. NB: in ingest this is a second heavy model; the
# hard-signal layer keeps it off ~a third of files, and low-RAM residency tuning
# (sharing one model) is a follow-up.
_CLASSIFY_DEFAULT = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
_cmodel = None
_ctok = None


def _classify_generate(messages: list[dict[str, str]], max_tokens: int = 120) -> str:
    from find_and_seek.config.settings import role_backend

    if role_backend("summary") == "ollama":
        from find_and_seek.ingest.summarise_ollama import classify_generate
        return classify_generate(messages, max_tokens=max_tokens)

    if role_backend("summary") == "vllm":
        from find_and_seek.ingest.summarise_vllm import classify_generate
        return classify_generate(messages, max_tokens=max_tokens)

    global _cmodel, _ctok
    from mlx_lm import generate as mlx_generate

    if _cmodel is None:
        from mlx_lm import load as mlx_load
        repo = os.environ.get("FINDANDSEEK_MLX_CLASSIFY_MODEL", _CLASSIFY_DEFAULT)
        _cmodel, _ctok = mlx_load(repo)
        logger.info("classifier model loaded: %s", repo)
    prompt = _ctok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    kwargs: dict = {"max_tokens": max_tokens}
    try:
        from mlx_lm.sample_utils import make_sampler
        kwargs["sampler"] = make_sampler(temp=0.1)
    except Exception:  # noqa: BLE001
        pass
    return mlx_generate(_cmodel, _ctok, prompt=prompt, verbose=False, **kwargs)


def unload_classifier() -> None:
    """Free the classifier model (residency discipline)."""
    global _cmodel, _ctok
    _cmodel = _ctok = None
    try:
        import gc
        import mlx.core as mx
        gc.collect()
        (getattr(mx, "clear_cache", None) or (lambda: None))()
    except Exception:  # noqa: BLE001
        pass


@dataclass(frozen=True)
class ClassifyResult:
    slug: str                 # a taxonomy slug, or NEEDS_REVIEW
    confidence: str           # "high" | "medium" | "low" | "none"
    reason: str
    raw: str = ""             # what the model actually said (for auditing)
    # A short noun-phrase the model would name this kind of document, in ITS OWN
    # words — e.g. "tender response", "meeting agenda". This is the seed of an
    # EMERGENT category: when many uncertain docs share a suggestion, the app can
    # offer "add this category?". Side-channel only — never a stored type.
    suggested: str = ""

    @property
    def needs_review(self) -> bool:
        return self.slug == NEEDS_REVIEW


def build_messages(text: str, filename: str, taxonomy: list[Category], evidence: str = "") -> list[dict[str, str]]:
    lines = []
    for c in taxonomy:
        excl = f" (NOT: {c.not_})" if c.not_ else ""
        lines.append(f"- {c.slug}: {c.definition}{excl}")
    catalogue = "\n".join(lines)
    system = (
        "You are a precise document classifier. Decide which ONE category a document "
        "belongs to, judging STRICTLY by what the CONTENT shows against the definitions "
        "below — not by a brand name, not because prices or money appear, and NOT by the "
        "filename. The filename is only a weak hint and is OFTEN MISLEADING: a file named "
        "'..._Letter_...' may be a policy or code of conduct, not a letter; a 'chat' or "
        "'message' export is not a letter. When the filename and the content disagree, "
        "trust the content.\n\n"
        f"Categories:\n{catalogue}\n\n"
        "Rules:\n"
        "- Pick the single best-fitting category by its exact slug, based on what the "
        "document IS, not what it is named.\n"
        "- It is BETTER to leave a document unsorted than to place it in the wrong "
        "category. If it does not CLEARLY fit one specific category by its definition, "
        "set confidence to 'low' — do not force a fit.\n"
        "- A category names a KIND of document, never a project or topic. A report, "
        "spreadsheet, letter, or specification that merely BELONGS to some project "
        "(a tender, a case, a job) keeps its structural kind — it does not move into "
        "the project's category just because the topic matches.\n"
        "- Financial documents: proof that payment was COMPLETED (an amount paid, a "
        "till/tax-invoice docket) is a receipt; a request for payment still owing is an "
        "invoice; a blank or fillable official document is a form. What the document DOES "
        "decides — not whether it has boxes, codes, or the words 'tax invoice'.\n"
        "- Use 'other' only for a real document that genuinely fits no category at all.\n"
        '- Set confidence "high" only when the content plainly matches the definition.\n'
        '- Always add "suggested": a 1-3 word, lowercase noun phrase naming what this '
        'document actually IS in your own words (e.g. "tender response", "meeting agenda", '
        '"pay slip"). This is most useful when nothing fits well — it may differ from the '
        "category you picked.\n"
        'Return ONLY JSON: {"category":"<slug>","confidence":"high|medium|low",'
        '"reason":"<=15 words","suggested":"<1-3 words>"}. No markdown, no thinking.'
    )
    body = text.strip()[:4000]
    ev = f"What we already know about this file:\n{evidence}\n\n" if evidence.strip() else ""
    # Content first; filename last and framed as a weak, possibly-misleading hint — so a
    # category-like word in the name can't drag the verdict (real-data bug:
    # 'Supplier_Code_of_Conduct_Letter' was classed 'letter' off the filename alone).
    user = f"{ev}Content:\n{body}\n\nFilename (weak hint only, may be misleading): {filename}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _verify_placement(slug: str, cat: Category, text: str) -> bool | None:
    """Second-opinion check: shown ONLY the chosen category's definition, does the
    document actually match it? A generator that picks the best of 17 options can
    be confidently wrong; a verifier judging one definition against the text is a
    much easier task for the same small model, so disagreement is a strong signal.
    Returns False on mismatch, True on match, None when unavailable (fail open —
    verification must never break ingest)."""
    body = text.strip()[:2500]
    excl = f" NOT: {cat.not_}" if cat.not_ else ""
    msgs = [
        {"role": "system", "content":
            "You are a strict verifier. Judge ONLY whether the document matches the "
            'definition. Return ONLY JSON: {"match": true|false, "reason": "<=12 words"}.'},
        {"role": "user", "content":
            f"Definition of '{slug}': {cat.definition}{excl}\n\nDocument:\n{body}\n\n"
            f"Does this document match the definition of '{slug}'?"},
    ]
    try:
        raw = _classify_generate(msgs, max_tokens=80)
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        return bool(json.loads(m.group()).get("match"))
    except Exception:  # noqa: BLE001
        return None


def _parse(raw: str, valid: set[str]) -> ClassifyResult:
    """Validate the model's answer against the taxonomy. Out-of-set / unparseable /
    low-confidence → needs-review."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return ClassifyResult(NEEDS_REVIEW, "none", "no JSON from classifier", raw[:200])
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return ClassifyResult(NEEDS_REVIEW, "none", "malformed classifier JSON", raw[:200])
    slug = str(data.get("category", "")).strip().lower()
    conf = str(data.get("confidence", "")).strip().lower()
    reason = str(data.get("reason", ""))[:120]
    suggested = _clean_suggestion(data.get("suggested", ""))
    if conf not in {"high", "medium", "low"}:
        conf = "low"
    if slug not in valid:
        # The model invented a label outside the taxonomy — exactly what we refuse
        # to store. Route to review, but keep the off-taxonomy label as the emergent
        # suggestion (it's the model naming a kind we don't have a category for yet).
        return ClassifyResult(NEEDS_REVIEW, conf, f"off-taxonomy: {slug!r}", raw[:200],
                              suggested=suggested or _clean_suggestion(slug))
    if conf == "low":
        return ClassifyResult(NEEDS_REVIEW, conf, reason or "low confidence", raw[:200],
                              suggested=suggested)
    return ClassifyResult(slug, conf, reason, raw[:200], suggested=suggested)


# A short, storable suggestion label: lowercase, a few words, no junk/slugs.
def _clean_suggestion(raw) -> str:
    s = re.sub(r"[^a-z0-9 /&-]", " ", str(raw).strip().lower())
    s = re.sub(r"\s+", " ", s).strip()
    words = s.split()
    return " ".join(words[:3]) if 0 < len(words) <= 4 else ""


def classify_text(text: str, filename: str, conn=None, taxonomy: list[Category] | None = None,
                  evidence: str = "") -> ClassifyResult:
    """Classify one document's text against the active taxonomy, optionally given
    soft `evidence` (origin/path hints) to reason with.

    Test mode and a missing/failed model both degrade to needs-review (honest) —
    never to a fabricated type. Thin extraction with no evidence also abstains.
    """
    from find_and_seek.config.settings import role_backend

    tax = taxonomy if taxonomy is not None else load_taxonomy(conn)
    valid = {c.slug for c in tax}

    if len((text or "").strip()) < _MIN_CHARS and not evidence.strip():
        return ClassifyResult(NEEDS_REVIEW, "none", "too little text to classify")

    if role_backend("summary") == "test":
        return ClassifyResult(NEEDS_REVIEW, "none", "test mode")

    try:
        messages = build_messages(text, filename, tax, evidence)
        raw = _classify_generate(messages, max_tokens=120)
        res = _parse(raw, valid)
        # Cap the self-reported confidence by extraction volume. The placement is
        # kept (the slug may still be right) — but a verdict reached on a thin body
        # must not WEAR more certainty than the evidence supports; UI/agents hedge.
        capped = cap_confidence(res.confidence, len((text or "").strip()))
        if capped != res.confidence:
            res = ClassifyResult(res.slug, capped, res.reason, res.raw, suggested=res.suggested)

        # Verify-then-retry: a placement that fails verification against its own
        # definition gets ONE re-ask excluding that category. If the retry finds a
        # DIFFERENT category that verifies, switch to it. Otherwise KEEP the
        # original placement demoted to 'medium' — verification failure is an
        # uncertainty signal, not a veto (a veto measured 35% Unsorted on the gold
        # set, rejecting even literal cover letters; demotion keeps coverage while
        # the hedge tells users/agents this one deserves a glance).
        # FINDANDSEEK_CLASSIFY_VERIFY=0 disables (e.g. bulk runs where speed wins).
        if (res.slug != NEEDS_REVIEW
                and os.environ.get("FINDANDSEEK_CLASSIFY_VERIFY", "1") != "0"):
            cat = next((c for c in tax if c.slug == res.slug), None)
            if cat is not None and _verify_placement(res.slug, cat, text) is False:
                retry = list(messages)
                retry[-1] = {"role": "user", "content": messages[-1]["content"] +
                             f"\n\nNote: verification determined this document is NOT "
                             f"'{res.slug}'. Choose the best OTHER category, or use low "
                             f"confidence if nothing else fits."}
                res2 = _parse(_classify_generate(retry, max_tokens=120), valid)
                cat2 = next((c for c in tax if c.slug == res2.slug), None)
                if (res2.slug not in (NEEDS_REVIEW, res.slug) and cat2 is not None
                        and _verify_placement(res2.slug, cat2, text) is not False):
                    capped2 = cap_confidence(res2.confidence, len((text or "").strip()))
                    return ClassifyResult(res2.slug, capped2, res2.reason, res2.raw,
                                          suggested=res2.suggested)
                demoted = "medium" if _CONF_ORDER.get(res.confidence, 0) > _CONF_ORDER["medium"] else res.confidence
                return ClassifyResult(res.slug, demoted,
                                      (res.reason + " (unverified)").strip(),
                                      res.raw, suggested=res.suggested)
        return res
    except Exception as e:  # noqa: BLE001 — model unavailable/failed: don't guess
        logger.debug("classify_text fell back to needs-review: %s", e)
        return ClassifyResult(NEEDS_REVIEW, "none", "classifier unavailable")


def classify_document(path: str, filename: str, text: str, conn=None,
                      taxonomy: list[Category] | None = None) -> ClassifyResult:
    """Full evidence-based classification — the entry point ingest should use.

    1. A hard signal (format, or precise metadata like an XFA form) sets the type
       directly: a .csv IS a spreadsheet; we don't ask the model.
    2. Otherwise the model decides from the content, handed the soft evidence
       (web origin, folder prior) so it judges with context instead of guessing.
    3. Low confidence / off-taxonomy / thin → needs-review.
    """
    from find_and_seek.organize.signals import gather_evidence, hard_signal

    tax = taxonomy if taxonomy is not None else load_taxonomy(conn)
    valid = {c.slug for c in tax}

    hs = hard_signal(path, filename)
    if hs is not None and hs.slug in valid:
        return ClassifyResult(hs.slug, "high", hs.reason, raw="hard-signal")

    evidence = gather_evidence(path, filename).as_prompt()
    return classify_text(text, filename, conn=conn, taxonomy=tax, evidence=evidence)
