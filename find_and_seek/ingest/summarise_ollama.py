"""Summarisation + classification via Ollama (native /api/chat endpoint).

Works against either a local Ollama daemon (the default when no cloud API key
is configured — this is the production summary path off-Mac, where MLX doesn't
exist) or Ollama Cloud. Uses the same native Ollama API format as
tools/agent-panel/panel.py.

Env vars:
  FINDANDSEEK_OLLAMA_CLOUD_KEY  — preferred (matches panel convention)
  OLLAMA_API_KEY              — fallback
  OLLAMA_BASE_URL             — default https://ollama.com with a key,
                                http://127.0.0.1:11434 without one
  FINDANDSEEK_OLLAMA_CHAT_MODEL — default deepseek-v3.2 (cloud) / qwen3:4b (local,
                                the same model family MLX serves on Macs)
  FINDANDSEEK_OLLAMA_CONCURRENCY — parallel HTTP calls per batch (default 8)
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from find_and_seek.ingest.chunk import Chunk
from find_and_seek.ingest.summarise import _heuristic_summary, build_messages, parse_summary

logger = logging.getLogger(__name__)

_LOCAL_URL = "http://127.0.0.1:11434"
_TIMEOUT = 120


def _api_key() -> str:
    for name in ("FINDANDSEEK_OLLAMA_CLOUD_KEY", "OLLAMA_API_KEY"):
        val = os.environ.get(name, "").strip()
        if val:
            return val
    # Optional key file, for people who'd rather not put a secret in the
    # environment. Default location follows the XDG convention; override with
    # FINDANDSEEK_OLLAMA_CLOUD_KEY_FILE.
    key_file = os.environ.get("FINDANDSEEK_OLLAMA_CLOUD_KEY_FILE", "").strip()
    if not key_file:
        config_home = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
        key_file = os.path.join(config_home, "findandseek", "ollama-api-key")
    try:
        with open(key_file) as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _base_url() -> str:
    """Cloud when a key is configured, the local daemon otherwise."""
    env = os.environ.get("OLLAMA_BASE_URL", "").strip()
    if env:
        return env
    return "https://ollama.com" if _api_key() else _LOCAL_URL


def _default_model() -> str:
    return "deepseek-v3.2" if _base_url() != _LOCAL_URL else "qwen3:4b"


def _http_json(path: str, payload: dict, retries: int = 3) -> dict:
    url = _base_url().rstrip("/") + path
    key = _api_key()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = json.dumps(payload).encode()
    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            last_err = RuntimeError(f"HTTP {e.code}: {body[:300]}")
            if e.code < 500 and e.code != 429:
                raise last_err from e
        except Exception as e:  # noqa: BLE001
            last_err = RuntimeError(str(e))
        if attempt < retries - 1:
            time.sleep(1.5 * (2 ** attempt))
    raise last_err or RuntimeError(f"Request failed: {path}")


def _chat(messages: list[dict[str, str]], max_tokens: int = 512) -> str:
    model = os.environ.get("FINDANDSEEK_OLLAMA_CHAT_MODEL") or _default_model()
    resp = _http_json("/api/chat", {
        "model": model,
        "stream": False,
        "think": False,
        "options": {"num_predict": max_tokens, "temperature": 0.1},
        "messages": messages,
    })
    return resp.get("message", {}).get("content", "")


def summarise_file(chunks: list[Chunk], filename: str) -> dict[str, Any]:
    if not chunks:
        return _heuristic_summary(chunks, filename)
    messages = build_messages(chunks, filename)
    try:
        raw = _chat(messages)
        result = parse_summary(raw, filename)
        if result:
            return result
    except Exception as e:  # noqa: BLE001
        logger.warning("Ollama summarise failed (%s): %s", filename, e)
    return _heuristic_summary(chunks, filename)


def summarise_batch(items: list[tuple[list[Chunk], str]]) -> list[dict[str, Any]]:
    if not items:
        return []
    concurrency = int(os.environ.get("FINDANDSEEK_OLLAMA_CONCURRENCY", "8"))
    results: list[dict[str, Any]] = [{}] * len(items)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(summarise_file, chunks, filename): i
            for i, (chunks, filename) in enumerate(items)
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return results


def classify_generate(messages: list[dict[str, str]], max_tokens: int = 120) -> str:
    return _chat(messages, max_tokens=max_tokens)
