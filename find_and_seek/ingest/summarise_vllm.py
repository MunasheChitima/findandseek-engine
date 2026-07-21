"""Document summarisation through a vLLM OpenAI-compatible server.

vLLM gets its throughput from continuous batching: callers submit independent
requests concurrently and the server packs their prefill/decode work together.

Environment:
  FINDANDSEEK_VLLM_BASE_URL      default http://127.0.0.1:8000
  FINDANDSEEK_VLLM_CHAT_MODEL    served model name
  FINDANDSEEK_VLLM_CONCURRENCY   in-flight requests per ingest batch (default 32)
  FINDANDSEEK_VLLM_TIMEOUT       request timeout seconds (default 180)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from find_and_seek.ingest.chunk import Chunk
from find_and_seek.ingest.summarise import _heuristic_summary, build_messages, parse_summary

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "google/gemma-4-E4B-it-qat-w4a16-ct"


def _base_url() -> str:
    return os.environ.get("FINDANDSEEK_VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _model() -> str:
    return os.environ.get("FINDANDSEEK_VLLM_CHAT_MODEL", _DEFAULT_MODEL)


async def _chat(
    client: httpx.AsyncClient,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 512,
) -> str:
    response = await client.post(
        f"{_base_url()}/v1/chat/completions",
        json={
            "model": _model(),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"].get("content")
    return content or ""


async def _summarise_one(
    client: httpx.AsyncClient,
    chunks: list[Chunk],
    filename: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    if not chunks:
        return _heuristic_summary(chunks, filename)
    async with semaphore:
        try:
            raw = await _chat(client, build_messages(chunks, filename))
            parsed = parse_summary(raw, filename)
            if parsed is not None:
                return parsed
            logger.warning("vLLM returned invalid summary JSON for %s", filename)
        except Exception as exc:  # noqa: BLE001 — one bad request must not take the batch
            logger.warning("vLLM summarise failed (%s): %s", filename, exc)
    return _heuristic_summary(chunks, filename)


async def _summarise_batch_async(
    items: list[tuple[list[Chunk], str]],
) -> list[dict[str, Any]]:
    concurrency = max(1, int(os.environ.get("FINDANDSEEK_VLLM_CONCURRENCY", "32")))
    timeout = float(os.environ.get("FINDANDSEEK_VLLM_TIMEOUT", "180"))
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        return await asyncio.gather(
            *(
                _summarise_one(client, chunks, filename, semaphore)
                for chunks, filename in items
            )
        )


def summarise_batch(items: list[tuple[list[Chunk], str]]) -> list[dict[str, Any]]:
    if not items:
        return []
    return asyncio.run(_summarise_batch_async(items))


def summarise_file(chunks: list[Chunk], filename: str) -> dict[str, Any]:
    return summarise_batch([(chunks, filename)])[0]


def classify_generate(messages: list[dict[str, str]], max_tokens: int = 120) -> str:
    async def run() -> str:
        timeout = float(os.environ.get("FINDANDSEEK_VLLM_TIMEOUT", "180"))
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await _chat(client, messages, max_tokens=max_tokens)

    return asyncio.run(run())
