"""Result reranking.

Two modes, picked automatically at runtime:

1. **Cross-encoder** — if a bundled ONNX model *and* its tokenizer are present
   (``models/cross-encoder.onnx`` + ``models/tokenizer.json``), each
   (query, chunk) pair is scored by real cross-encoder inference. Nothing is
   downloaded at runtime, preserving the zero-egress guarantee.
2. **Lexical+semantic blend** — otherwise, candidates are reranked by blending
   their fused retrieval score (the semantic/keyword prior) with a lexical
   overlap signal. This is an honest fallback, not a no-op: it still reorders
   results using both signals.
"""

from __future__ import annotations

import logging
import os
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Bundled .app overrides this (FINDANDSEEK_RERANK_DIR → Resources/models/reranker);
# in dev it falls back to the repo's ./models dir.
MODELS_DIR = Path(os.environ.get("FINDANDSEEK_RERANK_DIR")
                  or Path(__file__).resolve().parents[2] / "models")
MODEL_PATH = MODELS_DIR / "cross-encoder.onnx"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.json"

# Blend weights for the fallback path.
_FUSED_WEIGHT = 0.7
_LEXICAL_WEIGHT = 0.3

# How much the cross-encoder overrides the fused retrieval prior (0..1).
# 1.0 = pure cross-encoder (length-biased, regresses); 0 = ignore CE.
# 0.65 was the best/robust point on a 48-doc eval (recall@1 90%→94%, MRR 0.923→0.955),
# flat through 0.8. Tuned on synthetic data — treat as a sensible default, not gospel.
CE_WEIGHT = float(os.environ.get("FINDANDSEEK_RERANK_CE_WEIGHT", "0.65"))

_session = None
_tokenizer = None
_warned = False


def _get_session():
    global _session, _warned
    if _session is not None:
        return _session
    if not MODEL_PATH.exists():
        if not _warned:
            logger.info("Cross-encoder ONNX not found at %s — using lexical+semantic rerank", MODEL_PATH)
            _warned = True
        return None
    try:
        import onnxruntime as ort

        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    except Exception as e:  # pragma: no cover - depends on optional artifact
        logger.warning("Failed to load cross-encoder ONNX (%s) — falling back", e)
        _session = None
    return _session


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    if not TOKENIZER_PATH.exists():
        return None
    try:
        from tokenizers import Tokenizer

        _tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    except Exception as e:  # pragma: no cover - depends on optional artifact
        logger.warning("Cross-encoder tokenizer unavailable (%s) — falling back", e)
        _tokenizer = None
    return _tokenizer


def _cross_encoder_scores(
    session, tokenizer, query: str, candidates: list[tuple[int, str]],
    fused_scores: dict[int, float] | None = None,
) -> list[tuple[int, float]]:
    """Cross-encoder inference over (query, chunk) pairs, *blended* with the
    fused retrieval prior. Pure cross-encoder scoring is length-biased — long
    documents get many chunks and thus many chances at a spuriously high score —
    so we combine sigmoid(CE) with the normalised fused score rather than letting
    the cross-encoder override retrieval entirely."""
    encodings = [tokenizer.encode(query, text) for _, text in candidates]
    max_len = min(512, max((len(e.ids) for e in encodings), default=0))
    if max_len == 0:
        return []

    input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
    attention = np.zeros((len(encodings), max_len), dtype=np.int64)
    token_types = np.zeros((len(encodings), max_len), dtype=np.int64)
    for i, enc in enumerate(encodings):
        ids = enc.ids[:max_len]
        input_ids[i, : len(ids)] = ids
        attention[i, : len(ids)] = enc.attention_mask[:max_len]
        if enc.type_ids:
            token_types[i, : len(ids)] = enc.type_ids[:max_len]

    feed = {"input_ids": input_ids, "attention_mask": attention}
    expected = {inp.name for inp in session.get_inputs()}
    if "token_type_ids" in expected:
        feed["token_type_ids"] = token_types
    feed = {k: v for k, v in feed.items() if k in expected}

    logits = session.run(None, feed)[0]
    logits = np.asarray(logits, dtype=np.float32)
    relevance = logits[:, -1] if logits.ndim == 2 and logits.shape[1] > 1 else logits.reshape(-1)
    ce_norm = 1.0 / (1.0 + np.exp(-relevance))  # sigmoid -> [0,1]

    fused = fused_scores or {}
    max_fused = max(fused.values()) if fused else 0.0
    scored: list[tuple[int, float]] = []
    for (cid, _), ce in zip(candidates, ce_norm):
        fused_norm = (fused.get(cid, 0.0) / max_fused) if max_fused else 0.0
        scored.append((cid, CE_WEIGHT * float(ce) + (1.0 - CE_WEIGHT) * fused_norm))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _blended_scores(
    query: str,
    candidates: list[tuple[int, str]],
    fused_scores: dict[int, float] | None,
) -> list[tuple[int, float]]:
    """Honest fallback: combine the fused retrieval prior with lexical overlap."""
    q_tokens = {t for t in query.lower().split() if len(t) > 2}

    fused = fused_scores or {}
    max_fused = max(fused.values()) if fused else 0.0

    scored: list[tuple[int, float]] = []
    for cid, text in candidates:
        fused_norm = (fused.get(cid, 0.0) / max_fused) if max_fused else 0.0
        if q_tokens:
            t_tokens = set(text.lower().split())
            lexical = len(q_tokens & t_tokens) / len(q_tokens)
        else:
            lexical = 0.0
        score = _FUSED_WEIGHT * fused_norm + _LEXICAL_WEIGHT * lexical
        scored.append((cid, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def rerank(
    query: str,
    candidates: list[tuple[int, str]],
    fused_scores: dict[int, float] | None = None,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Rerank (chunk_id, text) pairs. Uses the cross-encoder when available,
    otherwise an honest fused+lexical blend."""
    if not candidates:
        return []

    session = _get_session()
    tokenizer = _get_tokenizer() if session is not None else None
    if session is not None and tokenizer is not None:
        try:
            return _cross_encoder_scores(session, tokenizer, query, candidates, fused_scores)[:top_k]
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Cross-encoder inference failed (%s) — falling back", e)

    return _blended_scores(query, candidates, fused_scores)[:top_k]
