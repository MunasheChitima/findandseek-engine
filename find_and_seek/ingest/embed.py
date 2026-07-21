"""Embeddings via in-process MLX, with a deterministic fallback."""

from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from typing import Sequence

import numpy as np

from find_and_seek.config.profiles import EMBED_DIM

logger = logging.getLogger(__name__)

from find_and_seek.config.agent_mode import query_embed_pseudo_only  # noqa: E402

# ── Query-embedding cache ─────────────────────────────────────────────
# The vector arm of hybrid search embeds the query on every search and, per
# file, on every evidence expansion. Repeated / near-identical queries are the
# norm inside one generation task, so memoise the query→vector map (keyed by
# backend so a config change can't serve stale vectors). Bounded LRU.
_QUERY_VEC_CACHE: "OrderedDict[tuple[str, str], list[float]]" = OrderedDict()
_QUERY_VEC_CACHE_MAX = int(os.environ.get("FINDANDSEEK_QUERY_EMBED_CACHE", "512"))


def clear_query_vec_cache() -> None:
    """Drop the query-embedding cache (tests / after a backend switch)."""
    _QUERY_VEC_CACHE.clear()


def _pseudo_embed(text: str) -> list[float]:
    # Seed a RNG from the content hash: deterministic, but (unlike reinterpreting
    # raw hash bytes as float32) never produces NaN/inf that would poison the norm.
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(EMBED_DIM).astype(np.float32)
    arr /= np.linalg.norm(arr) + 1e-9
    return arr.tolist()


def embed_texts(texts: Sequence[str], model: str | None = None) -> list[list[float]]:
    if not texts:
        return []
    from find_and_seek.config.settings import role_backend

    if role_backend("embed") == "test":
        return [_pseudo_embed(t) for t in texts]

    if role_backend("embed") == "ollama":
        try:
            from find_and_seek.ingest import embed_ollama

            return embed_ollama.embed_texts(texts)
        except Exception as e:  # noqa: BLE001
            logger.warning("Ollama embed failed (%s) — using pseudo-embeddings", e)
            return [_pseudo_embed(t) for t in texts]

    # MLX in-process (the production backend, and the default for any non-test
    # value). Fall back to deterministic pseudo-embeddings if the model can't load
    # (no cache / no network) so CI and degraded environments still work.
    try:
        from find_and_seek.ingest import embed_mlx

        return embed_mlx.embed_texts(texts)
    except Exception as e:  # noqa: BLE001
        logger.warning("MLX embed failed (%s) — using pseudo-embeddings", e)
        return [_pseudo_embed(t) for t in texts]


def embed_one(text: str, model: str | None = None) -> list[float]:
    return embed_texts([text], model=model)[0]


def embed_query_vector(text: str) -> list[float]:
    """Query-side vector for the hybrid-search vector arm.

    The corpus is embedded with a real model at ingest, so the query must be
    embedded with that *same* backend for the vector arm to match stored chunk
    vectors. We do that here — once per query, then cached. Previously the
    agent/MCP path forced a deterministic hash ("pseudo") vector to avoid a
    query-time model load; that made the vector arm noise and left ranking to
    keyword-only, which is the biggest driver of extra agent round-trips on
    multi-document generation. Set ``FINDANDSEEK_QUERY_EMBED=pseudo`` to restore
    the legacy zero-model-load behaviour for fully offline runs.

    On any embed failure (no backend / offline CI) we degrade to the pseudo
    vector rather than raise, so search never hard-fails on a missing model.
    """
    if query_embed_pseudo_only():
        return _pseudo_embed(text)

    from find_and_seek.config.settings import role_backend

    key = (role_backend("embed"), text)
    cached = _QUERY_VEC_CACHE.get(key)
    if cached is not None:
        _QUERY_VEC_CACHE.move_to_end(key)
        return cached

    try:
        vec = embed_one(text)
    except Exception as e:  # noqa: BLE001 — never hard-fail search on a missing model
        logger.warning("query embed failed (%s) — using pseudo query vector", e)
        vec = _pseudo_embed(text)

    _QUERY_VEC_CACHE[key] = vec
    _QUERY_VEC_CACHE.move_to_end(key)
    while len(_QUERY_VEC_CACHE) > _QUERY_VEC_CACHE_MAX:
        _QUERY_VEC_CACHE.popitem(last=False)
    return vec
