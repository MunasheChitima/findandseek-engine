"""MLX model identifiers for the in-process inference stack."""

from __future__ import annotations

# Default MLX model IDs (HF / mlx-community repos) — what the in-process backends
# actually load. Kept here for display/status; each backend also defines its own
# candidate chain. See AAR-001..003.
MLX_MODELS = {
    "embed_model": "mlx-community/embeddinggemma-300m-bf16",
    "summary_model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "vision_model": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    "embed_dimensions": None,
}

# Summary model by RAM tier (owner decision 2026-07-17, benchmarked on a paired
# 9-doc corpus + adversarial memo — see docs/testing and the data/golden set):
# Qwen3-4B is production (~2.3s/doc, 0 parse failures, best fact yield);
# Qwen3-1.7B serves ≤9 GB Macs (~1.5s/doc, ~1.2 GB resident) and REQUIRES
# enable_thinking=False at the chat template or its <think> preamble breaks
# parse_summary (8/9 silent fallbacks without it). Older Qwen2.5/gemma-2 entries
# retired; anyone with only those cached re-fetches on first run.
_MLX_SUMMARY_LOW = "mlx-community/Qwen3-1.7B-4bit"
_MLX_SUMMARY_FULL = [
    "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "mlx-community/Qwen3-1.7B-4bit",
]


def mlx_summary_candidates(low_ram: bool | None = None) -> list[str]:
    """Ordered MLX summary repos to try, smallest-first on low-RAM machines."""
    from find_and_seek.config.settings import LOW_RAM

    low = LOW_RAM if low_ram is None else low_ram
    return [_MLX_SUMMARY_LOW, *_MLX_SUMMARY_FULL] if low else list(_MLX_SUMMARY_FULL)


def resolve_models(profile_name: str) -> dict[str, str | int | None]:
    """Embed/summary/vision model identifiers for the active profile.

    On Macs the production backend is MLX, so this is just the MLX repo ids —
    no daemon query, no subprocess, no network. ``profile_name`` is accepted for
    call-site compatibility; the MLX repos are the same across tiers (the summary
    tier split lives in :func:`mlx_summary_candidates`). Off-Mac the production
    backend is a local Ollama daemon + tesseract OCR, so report those ids
    (env overrides included) for honest status/diagnostics display.
    """
    from find_and_seek.config.settings import IS_MAC

    if IS_MAC:
        return dict(MLX_MODELS)
    import os

    return {
        "embed_model": os.environ.get("FINDANDSEEK_OLLAMA_EMBED_MODEL", "embeddinggemma"),
        "summary_model": os.environ.get("FINDANDSEEK_OLLAMA_CHAT_MODEL", "qwen3:4b"),
        "vision_model": "tesseract",
        "embed_dimensions": None,
    }
