"""The MLX embed cold-load is serialised: concurrent callers load the model once.

Regression for the warmup-vs-first-search race — the server fires a background
warmup thread precisely to hide the ~multi-second cold load, but without a lock
a concurrent `/search` could pass the `_model is not None` check before warmup
finished and load the model a second time, doubling the cost warmup exists to
avoid. This drives the real `_load()` with a fake, slow `mlx_embeddings.utils.load`
so no model download / MLX runtime is needed.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import find_and_seek.ingest.embed_mlx as embed_mlx


def test_concurrent_load_triggers_exactly_one_underlying_load(monkeypatch):
    calls = {"n": 0}
    start = threading.Barrier(4)

    def fake_load(repo):
        calls["n"] += 1
        time.sleep(0.2)  # widen the race window a real cold load would open
        return object(), object()  # (model, tokenizer)

    fake_utils = types.ModuleType("mlx_embeddings.utils")
    fake_utils.load = fake_load
    fake_pkg = types.ModuleType("mlx_embeddings")
    monkeypatch.setitem(sys.modules, "mlx_embeddings", fake_pkg)
    monkeypatch.setitem(sys.modules, "mlx_embeddings.utils", fake_utils)
    monkeypatch.setenv("FINDANDSEEK_MLX_EMBED_MODEL", "fake/model")

    # Reset the process-wide cache so this test loads from cold.
    monkeypatch.setattr(embed_mlx, "_model", None)
    monkeypatch.setattr(embed_mlx, "_tokenizer", None)

    results = []

    def worker():
        start.wait()
        results.append(embed_mlx._load())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls["n"] == 1, f"cold load ran {calls['n']}x under contention; lock failed"
    # Every caller gets the same cached (model, tokenizer) tuple.
    assert all(r == results[0] for r in results)
