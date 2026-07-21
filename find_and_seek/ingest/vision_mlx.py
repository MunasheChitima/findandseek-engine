"""MLX vision/OCR backend — in-process VLM via mlx-vlm (the production OCR/caption
model).

Implements ``florence.process_image`` and reuses its shared prompt + chunk
builder. Like the summariser, the VLM is a *heavy* model, so ``load()`` /
``unload()`` here implement the §5.7 residency discipline.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from io import BytesIO

from PIL import Image

from find_and_seek.ingest.chunk import Chunk
from find_and_seek.ingest.florence import VISION_PROMPT, build_vision_chunks

logger = logging.getLogger(__name__)

# Qwen2.5-VL — strong document OCR, already cached on the dev machine.
_MODEL_CANDIDATES = [
    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
]

_model = None
_processor = None
_config = None
_model_id: str | None = None
_load_seconds: float = 0.0


def active_memory_gb() -> float:
    import mlx.core as mx

    getter = getattr(mx, "get_active_memory", None) or getattr(getattr(mx, "metal", None), "get_active_memory", None)
    return (getter() / (1024**3)) if getter else 0.0


def load() -> tuple:
    global _model, _processor, _config, _model_id, _load_seconds
    if _model is not None:
        return _model, _processor

    from mlx_vlm import load as vlm_load
    from mlx_vlm.utils import load_config

    explicit = os.environ.get("FINDANDSEEK_MLX_VISION_MODEL")
    candidates = [explicit] if explicit else _MODEL_CANDIDATES

    last_err: Exception | None = None
    for repo in candidates:
        try:
            t0 = time.perf_counter()
            model, processor = vlm_load(repo)
            _config = load_config(repo)
            _load_seconds = time.perf_counter() - t0
            _model, _processor, _model_id = model, processor, repo
            logger.info("MLX vision model loaded: %s (%.1fs)", repo, _load_seconds)
            return _model, _processor
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.debug("MLX vision model %s failed: %s", repo, e)
    raise RuntimeError(f"No MLX vision model could be loaded; last error: {last_err}")


def unload() -> None:
    global _model, _processor, _config
    _model = _processor = _config = None
    try:
        import gc

        import mlx.core as mx

        gc.collect()
        clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
        if clear:
            clear()
    except Exception:  # noqa: BLE001
        pass


def model_info() -> dict:
    return {"model_id": _model_id, "load_seconds": round(_load_seconds, 2)}


def _bytes_to_temp_jpg(data: bytes) -> str:
    img = Image.open(BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    img.save(path, format="JPEG")
    return path


def process_image(data: bytes, location_ref: str) -> list[Chunk]:
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load()
    img_path = _bytes_to_temp_jpg(data)
    try:
        formatted = apply_chat_template(processor, _config, VISION_PROMPT, num_images=1)
        result = vlm_generate(
            model, processor, formatted, image=[img_path],
            max_tokens=512, temperature=0.1, verbose=False,
        )
        text = result.text if hasattr(result, "text") else str(result)
        return build_vision_chunks(text, location_ref)
    finally:
        try:
            os.unlink(img_path)
        except OSError:
            pass
