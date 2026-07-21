"""MLX embedding backend — in-process Apple-Silicon inference (the production
embedder, and the default).

Implements the public surface of ``ingest.embed`` (``embed_texts`` /
``embed_one``) so it sits behind the same call sites unchanged.

Model is resolved in this order:
  1. ``FINDANDSEEK_MLX_EMBED_MODEL`` (explicit HF / mlx-community repo id)
  2. first entry of ``_MODEL_CANDIDATES`` that loads

768-dimensional models are preferred so vectors drop straight into the existing
``FLOAT[768]`` sqlite-vec schema with no coercion.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Sequence

import numpy as np

from find_and_seek.config.profiles import EMBED_DIM

logger = logging.getLogger(__name__)

# embeddinggemma-300m is 768d-native, so its vectors drop straight into the
# FLOAT[768] sqlite-vec schema with no coercion. bge-m3 (1024d, coerced) is a
# fallback.
_MODEL_CANDIDATES = [
    "mlx-community/embeddinggemma-300m-bf16",
    "mlx-community/embeddinggemma-300m-8bit",
    "mlx-community/bge-m3-mlx-fp16",
]

_model = None
_tokenizer = None
_model_id: str | None = None
_load_seconds: float = 0.0
# Serialises the ~multi-second cold load. Without it the server warmup thread and
# a concurrent first `/search` can both pass the `_model is not None` check and
# redundantly load the model — doubling the exact cost warmup exists to hide.
_load_lock = threading.Lock()


def _load():
    """Lazily load an MLX embedding model + tokenizer. Cached process-wide."""
    global _model, _tokenizer, _model_id, _load_seconds
    if _model is not None:  # fast path once warm — no lock contention per embed call
        return _model, _tokenizer

    with _load_lock:
        # Double-check: another thread may have finished the load while we waited.
        if _model is not None:
            return _model, _tokenizer

        from mlx_embeddings.utils import load  # imported lazily so test mode needs no MLX

        explicit = os.environ.get("FINDANDSEEK_MLX_EMBED_MODEL")
        candidates = [explicit] if explicit else _MODEL_CANDIDATES

        last_err: Exception | None = None
        for repo in candidates:
            try:
                t0 = time.perf_counter()
                model, tokenizer = load(repo)
                _load_seconds = time.perf_counter() - t0
                _model, _tokenizer, _model_id = model, tokenizer, repo
                # Bound MLX's buffer cache. Without this MLX keeps every freed buffer
                # for reuse, so a worker's RSS climbs without limit across thousands of
                # embed calls — the root cause of the multi-worker OOM. 512 MB is ample
                # for batch-to-batch reuse; the rest is returned to the OS.
                try:
                    import mlx.core as mx

                    cap = int(os.environ.get("FINDANDSEEK_MLX_CACHE_MB", "512")) * 1024 * 1024
                    mx.set_cache_limit(cap)
                except Exception as e:  # noqa: BLE001
                    logger.debug("MLX set_cache_limit failed: %s", e)
                logger.info("MLX embed model loaded: %s (%.1fs)", repo, _load_seconds)
                return _model, _tokenizer
            except Exception as e:  # noqa: BLE001 — probe each candidate
                last_err = e
                logger.debug("MLX embed model %s failed to load: %s", repo, e)
                continue
        raise RuntimeError(f"No MLX embedding model could be loaded; last error: {last_err}")


def model_info() -> dict:
    return {"model_id": _model_id, "load_seconds": round(_load_seconds, 2)}


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-9)


def _fit_dim(vec: np.ndarray) -> list[float]:
    if vec.shape[0] == EMBED_DIM:
        return vec.astype(np.float32).tolist()
    out = np.zeros(EMBED_DIM, dtype=np.float32)
    n = min(EMBED_DIM, vec.shape[0])
    out[:n] = vec[:n]
    return out.tolist()


def embed_texts(texts: Sequence[str], batch_size: int = 16) -> list[list[float]]:
    """Embed a batch in-process via MLX. Returns L2-normalised 768d vectors."""
    if not texts:
        return []
    import mlx.core as mx

    model, tokenizer = _load()
    out: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        inputs = tokenizer.batch_encode_plus(
            batch,
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=512,
        )
        outputs = model(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        # mlx-embeddings exposes a pooled, normalised sentence embedding as
        # ``text_embeds``; fall back to mean-pooling the last hidden state.
        emb = getattr(outputs, "text_embeds", None)
        if emb is None:
            hidden = outputs.last_hidden_state  # (B, T, H)
            mask = inputs.get("attention_mask")
            if mask is not None:
                m = mask[..., None]
                emb = (hidden * m).sum(axis=1) / mx.maximum(m.sum(axis=1), 1)
            else:
                emb = hidden.mean(axis=1)
        mx.eval(emb)
        arr = _normalize(np.array(emb, dtype=np.float32))
        out.extend(_fit_dim(row) for row in arr)

    # Release transient buffers back to the OS so RSS stays flat between files.
    mx.clear_cache()
    return out


def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]
