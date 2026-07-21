"""Fetch the model weights Find and Seek needs at first run.

On macOS this fetches MLX weights from Hugging Face:

  • the **embedding** model (search vectors)
  • the **MLX summary** model — Qwen3, ALWAYS required (owner decision
    2026-07-17: MLX is the production summariser on every Mac; AFM is out of
    the summary path). RAM tiering in config.models picks 4B vs 1.7B.
  • never the vision VLM (OCR is macOS Vision)

Weights land wherever ``HF_HOME`` points (the app uses Application Support).
After fetch, runtime sets ``HF_HUB_OFFLINE=1``.

Off-Mac the production backend is a local Ollama daemon, so this pulls the
equivalent Ollama models instead (qwen3:4b + embeddinggemma; OCR is tesseract,
no weights to fetch). Requires ``ollama`` on PATH with the daemon running.

Emits machine-readable progress lines the desktop app parses:
    MODEL <i>/<n> <repo>     # starting a model
    OK <repo>                # finished
    FAIL <repo>              # gave up after retries
    DONE ok|incomplete       # final line

Usage:
    findandseek-setup            # fetch required MLX models
    findandseek-setup --check    # report which are present (exit 1 if any missing)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from find_and_seek.config.models import MLX_MODELS, mlx_summary_candidates
from find_and_seek.config.settings import IS_MAC

# Ollama models the off-Mac backend defaults to (see summarise_ollama /
# embed_ollama — same Qwen3-4B + embeddinggemma family as the MLX stack).
OLLAMA_MODELS = ["qwen3:4b", "embeddinggemma"]


def models_to_fetch() -> list[str]:
    """MLX repos required for this machine.

    Always: embedder (semantic search) + the RAM-appropriate summary LLM.
    Never by default: vision VLM — set FINDANDSEEK_FETCH_MLX_VISION=1 for bench.
    """
    models = [MLX_MODELS["embed_model"], mlx_summary_candidates()[0]]
    if os.environ.get("FINDANDSEEK_FETCH_MLX_VISION", "").lower() in ("1", "true", "yes"):
        models.append(MLX_MODELS["vision_model"])
    return models


def is_present(repo: str) -> bool:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo, local_files_only=True)
        return True
    except Exception:
        return False


def fetch(repo: str, attempts: int = 6) -> bool:
    from huggingface_hub import snapshot_download

    for i in range(attempts):
        try:
            snapshot_download(repo, max_workers=4)
            print(f"OK {repo}", flush=True)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  … retry {i + 1}/{attempts} for {repo}: {type(e).__name__}", flush=True)
            time.sleep(5)
    print(f"FAIL {repo}", flush=True)
    return False


def ollama_present(model: str) -> bool:
    try:
        return subprocess.run(
            ["ollama", "show", model], capture_output=True, timeout=30
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def ollama_fetch(model: str) -> bool:
    try:
        ok = subprocess.run(["ollama", "pull", model], timeout=3600).returncode == 0
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"  … ollama pull {model} failed: {type(e).__name__}", flush=True)
        ok = False
    print(f"{'OK' if ok else 'FAIL'} {model}", flush=True)
    return ok


def _main_ollama(check_only: bool) -> int:
    n = len(OLLAMA_MODELS)
    all_ok = True
    for i, model in enumerate(OLLAMA_MODELS, 1):
        if check_only:
            present = ollama_present(model)
            print(f"{'present' if present else 'missing'} {model}", flush=True)
            all_ok &= present
        else:
            print(f"MODEL {i}/{n} {model}", flush=True)
            all_ok &= ollama_present(model) or ollama_fetch(model)
    if not check_only:
        print(f"DONE {'ok' if all_ok else 'incomplete'}", flush=True)
    return 0 if all_ok else 1


def main() -> int:
    check_only = "--check" in sys.argv
    if not IS_MAC:
        return _main_ollama(check_only)
    models = models_to_fetch()
    if not models:
        print("DONE ok", flush=True)
        return 0
    n = len(models)
    all_ok = True
    for i, repo in enumerate(models, 1):
        if check_only:
            present = is_present(repo)
            print(f"{'present' if present else 'missing'} {repo}", flush=True)
            all_ok &= present
        else:
            print(f"MODEL {i}/{n} {repo}", flush=True)
            all_ok &= fetch(repo)
    if not check_only:
        print(f"DONE {'ok' if all_ok else 'incomplete'}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
