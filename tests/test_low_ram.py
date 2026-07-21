"""RAM-tier behaviour: 8 GB Macs get smaller models/batches, a reachable
memory-pause threshold, and OCR off by default. Pure functions — no models load."""

from __future__ import annotations

from find_and_seek.config import models, settings


def test_min_free_gb_scales_with_ram():
    assert settings.default_min_free_gb(8) == 1.5     # 8 GB: never has 5 GB free
    assert settings.default_min_free_gb(16) == 2.5
    assert settings.default_min_free_gb(48) == 5.0


def test_batch_size_smaller_on_low_ram():
    assert settings.default_batch_size(8) == 4
    assert settings.default_batch_size(32) == 16


def test_ocr_off_by_default_only_on_low_ram():
    assert settings.ocr_enabled_default(low_ram=True) is False
    assert settings.ocr_enabled_default(low_ram=False) is True


def test_summary_model_smaller_on_low_ram():
    low = models.mlx_summary_candidates(low_ram=True)
    full = models.mlx_summary_candidates(low_ram=False)
    assert "1.7B" in low[0]                      # small model first on 8 GB
    assert "4B" in full[0]                       # full-size elsewhere
    assert low[-1] == full[-1]                   # full chain kept as fallback
    # Pure-Qwen3 ladder (owner decision 2026-07-17): no retired generations.
    assert not any("Qwen2.5" in m or "gemma" in m for m in low + full)


def test_summary_backend_platform_default(monkeypatch):
    """MLX Qwen3 is the production summariser on every Mac (owner decision
    2026-07-17: AFM dropped from the summary path); off-Mac the production
    summariser is a local Ollama daemon serving the same model family. Only an
    explicit env override may change it."""
    monkeypatch.delenv("FINDANDSEEK_TEST", raising=False)
    monkeypatch.delenv("FINDANDSEEK_SUMMARY_BACKEND", raising=False)
    expected = "mlx" if settings.IS_MAC else "ollama"
    assert settings.role_backend("summary") == expected
    monkeypatch.setenv("FINDANDSEEK_SUMMARY_BACKEND", "vllm")
    assert settings.role_backend("summary") == "vllm"


def test_first_run_never_fetches_unused_vision_vlm(monkeypatch):
    """Production OCR is macOS Vision (role_backend('vision') == 'macos'), so the
    MLX vision VLM (~2.9 GB) is bench/eval-only and must not be pulled on any RAM
    tier — it was costing every user a download for a model nothing loads."""
    monkeypatch.delenv("FINDANDSEEK_ENABLE_OCR", raising=False)
    monkeypatch.delenv("FINDANDSEEK_FETCH_MLX_VISION", raising=False)
    from find_and_seek import setup_weights

    monkeypatch.setattr(settings, "LOW_RAM", True)
    low = setup_weights.models_to_fetch()
    assert not any("VL" in m for m in low)       # vision VLM never pulled
    assert any("embeddinggemma" in m for m in low)
    assert any("1.7B" in m for m in low)         # small summary on 8 GB

    monkeypatch.setattr(settings, "LOW_RAM", False)
    full = setup_weights.models_to_fetch()
    assert not any("VL" in m for m in full)      # still not pulled on 16 GB+
    assert any("4B" in m for m in full)          # full summary elsewhere


def test_summary_weights_always_required(monkeypatch):
    """The MLX summariser is production on every machine — first run must fetch
    it unconditionally (no AFM branch left to skip it)."""
    monkeypatch.delenv("FINDANDSEEK_FETCH_MLX_VISION", raising=False)
    from find_and_seek import setup_weights

    monkeypatch.setattr(settings, "LOW_RAM", False)
    fetched = setup_weights.models_to_fetch()
    assert any("Qwen3-4B" in m for m in fetched)


def test_mlx_vision_is_opt_in(monkeypatch):
    """The VLM can still be fetched for bench/eval via an explicit flag."""
    monkeypatch.delenv("FINDANDSEEK_ENABLE_OCR", raising=False)
    from find_and_seek import setup_weights

    monkeypatch.setattr(settings, "LOW_RAM", False)
    monkeypatch.setenv("FINDANDSEEK_FETCH_MLX_VISION", "1")
    assert any("VL" in m for m in setup_weights.models_to_fetch())
