"""Document summarisation via Apple Foundation Models (on-device, macOS 26+).

Requires macOS 26+, Apple Intelligence enabled, and `apple-fm-sdk` installed.
Uses the system model already resident on-device — zero additional RAM or download.
Each document gets a fresh LanguageModelSession to prevent cross-document
context bleed. The batch path runs sequentially (AFM is single-threaded inference).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from find_and_seek.ingest.chunk import Chunk

logger = logging.getLogger(__name__)


async def _call(chunks: list[Chunk], filename: str) -> str:
    import apple_fm_sdk as fm
    from find_and_seek.ingest.summarise import SYSTEM_PROMPT, build_messages

    msgs = build_messages(chunks, filename)
    # AFM LanguageModelSession.respond() is single-turn; combine system + user.
    prompt = f"{msgs[0]['content']}\n\n{msgs[1]['content']}"
    session = fm.LanguageModelSession(model=fm.SystemLanguageModel())
    return await session.respond(prompt)


def summarise_file(chunks: list[Chunk], filename: str) -> dict[str, Any]:
    from find_and_seek.ingest.summarise import _heuristic_summary, parse_summary

    try:
        raw = asyncio.run(_call(chunks, filename))
        result = parse_summary(raw, filename)
        if result:
            return result
        logger.debug("AFM parse_summary returned None for %s, falling back", filename)
    except Exception as e:  # noqa: BLE001
        logger.warning("AFM summariser error for %s: %s", filename, e)
    return _heuristic_summary(chunks, filename)


def summarise_batch(items: list[tuple[list[Chunk], str]]) -> list[dict[str, Any]]:
    """Summarise a batch sequentially — AFM inference is single-threaded on-device."""
    return [summarise_file(chunks, filename) for chunks, filename in items]
