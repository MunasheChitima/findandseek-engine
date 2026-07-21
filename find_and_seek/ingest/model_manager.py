"""Sequential heavy-model loading discipline (§5.7).

Only one heavy model resident at a time. With the MLX backend this is enforced
in our own code (load → use → unload + free buffers). Any non-MLX backend (test,
or the lightweight macOS Vision OCR path) has no in-process heavy model to manage,
so it is a no-op here.

Roles: ``summary`` (LLM) and ``florence`` (vision VLM — Qwen2.5-VL via MLX). The
embedder is small and stays resident throughout, so it is not managed here.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator

from find_and_seek.config.profiles import get_profile
from find_and_seek.config.settings import role_backend

logger = logging.getLogger(__name__)

# role name in this manager → (backend role key, MLX backend module path)
_MLX_MODULES = {
    "summary": ("summary", "find_and_seek.ingest.summarise_mlx"),
    "florence": ("vision", "find_and_seek.ingest.vision_mlx"),
}

# load_log is a debugging aid on a process that runs for weeks and swaps models
# per batch, so it is capped rather than left to grow for the worker's lifetime.
_LOAD_LOG_MAX = 200


class ModelManager:
    def __init__(self) -> None:
        self.resident: str | None = None
        self.profile = get_profile()
        self.load_log: list[tuple[str, str]] = []

    def _backend_for(self, name: str) -> str:
        role_key = _MLX_MODULES.get(name, ("summary", ""))[0]
        return role_backend(role_key)

    def _mlx_module(self, name: str):
        import importlib

        return importlib.import_module(_MLX_MODULES[name][1])

    def _log(self, op: str, name: str) -> None:
        # Bounded: this lives on a process that runs for weeks.
        self.load_log.append((op, name))
        if len(self.load_log) > _LOAD_LOG_MAX:
            del self.load_log[: len(self.load_log) - _LOAD_LOG_MAX]

    def _load(self, name: str) -> bool:
        """Load `name`. Returns whether the model is actually resident after.

        The return value matters: a load can fail transiently (OOM while another
        process holds memory), and the caller must not then record the model as
        resident. Doing so makes every later use() short-circuit the load and
        never retry, so one unlucky moment silently disables summaries or
        embeddings for the life of the worker.
        """
        logger.info("model_manager: load %s", name)
        self._log("load", name)
        if self._backend_for(name) != "mlx":
            return True  # test / macOS-OCR: no in-process heavy model to manage
        try:
            self._mlx_module(name).load()
            return True
        except Exception as e:  # noqa: BLE001 — degrade gracefully, but report
            logger.warning("MLX load(%s) failed, will retry on next use: %s", name, e)
            return False

    def _unload(self, name: str) -> None:
        logger.info("model_manager: unload %s", name)
        self._log("unload", name)
        if self._backend_for(name) != "mlx":
            return
        try:
            self._mlx_module(name).unload()
        except Exception as e:  # noqa: BLE001
            logger.debug("MLX unload(%s) failed: %s", name, e)

    @contextmanager
    def use(self, name: str) -> Generator[None, None, None]:
        if self.resident and self.resident != name:
            self._unload(self.resident)
            self.resident = None
        if self.resident != name:
            # Only claim residency if the load actually succeeded — otherwise the
            # next use() would short-circuit and never retry.
            self.resident = name if self._load(name) else None
        try:
            yield
        finally:
            # On memory-constrained machines, free the heavy model after each
            # batch. With MLX this actually releases the buffers (verified: the
            # summariser cycles 1.6→0 GB, the VLM 2.9→0 GB).
            if self.profile.get("ingest_mode") == "idle_only":
                self._unload(name)
                self.resident = None


model_manager = ModelManager()
