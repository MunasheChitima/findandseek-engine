"""Agent-time engine search — read pre-built indexes only.

When enabled, hybrid search must not load MLX embedders or classifier models
at query time. Vectors already live in the index from ingest; the query uses a
lightweight deterministic vector (same path as FINDANDSEEK_TEST pseudo-embed).
"""

from __future__ import annotations

import os


def agent_search_enabled() -> bool:
    return os.environ.get("FINDANDSEEK_AGENT_SEARCH", "0") == "1"


def query_embed_pseudo_only() -> bool:
    """Whether the *query* vector should use the legacy deterministic hash
    embedding (no model load) instead of a real model embedding.

    Default is False: the query is embedded with the same backend the corpus
    was embedded with at ingest, so the vector arm of hybrid search actually
    matches stored chunk vectors. Historically, agent-search mode forced the
    pseudo vector here, which silently neutralised the vector arm and left
    ranking to keyword-only — the corpus is real-embedded, so a pseudo query
    matches nothing semantically.

    Opt back into the zero-model-load hash vector (fully offline / hermetic
    runs, or environments with no embed backend) via
    ``FINDANDSEEK_QUERY_EMBED`` in {pseudo, 0, off, false}.
    """
    return os.environ.get("FINDANDSEEK_QUERY_EMBED", "").strip().lower() in {
        "pseudo",
        "0",
        "off",
        "false",
    }
