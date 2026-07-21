"""Embeddings via Ollama (OpenAI-compatible API).

Works against either a local Ollama daemon (the default when no OLLAMA_API_KEY
is set — the production embed path off-Mac) or Ollama Cloud.

Env vars:
  OLLAMA_API_KEY        — cloud only; unset ⇒ local daemon
  OLLAMA_BASE_URL       — default https://api.ollama.com with a key,
                          http://127.0.0.1:11434 without one
  FINDANDSEEK_OLLAMA_EMBED_MODEL — default nomic-embed-text (cloud) /
                          embeddinggemma (local — the same 768d model the MLX
                          backend embeds with on Macs, so indexes stay portable)
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import httpx

logger = logging.getLogger(__name__)

_LOCAL_URL = "http://127.0.0.1:11434"
# 60s suits large ingest batches, but at query time a cold/contended model load
# blocking a search for a full minute is worse than degrading to the pseudo
# vector. Deployments that embed the query online should lower this (e.g. 8s) so
# a slow embed fails fast and search falls back rather than stalling.
_TIMEOUT = float(os.environ.get("FINDANDSEEK_OLLAMA_TIMEOUT", "60"))


def _base_url() -> str:
    """Cloud when a key is configured, the local daemon otherwise."""
    env = os.environ.get("OLLAMA_BASE_URL", "").strip()
    if env:
        return env
    return "https://api.ollama.com" if os.environ.get("OLLAMA_API_KEY") else _LOCAL_URL


def _default_model() -> str:
    return "nomic-embed-text" if _base_url() != _LOCAL_URL else "embeddinggemma"


def _client() -> httpx.Client:
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return httpx.Client(base_url=_base_url(), headers=headers, timeout=_TIMEOUT)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []
    model = os.environ.get("FINDANDSEEK_OLLAMA_EMBED_MODEL") or _default_model()
    with _client() as client:
        resp = client.post(
            "/v1/embeddings",
            json={"model": model, "input": list(texts)},
        )
        resp.raise_for_status()
    data = resp.json()
    items = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in items]
