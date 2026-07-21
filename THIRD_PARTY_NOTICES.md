# Third-Party Notices

FindandSeek Engine (`find-and-seek`) is licensed under BUSL-1.1 (see
[LICENSE.md](LICENSE.md)). That license covers FindandSeek's own code only.

The engine depends on third-party software, and at first run it downloads
third-party model weights. Each of those components is licensed by its own
authors under its own terms, listed below. Nothing here changes those terms.

The per-package license list should be regenerated from installed metadata
before each release rather than maintained by hand:

```bash
uv run --with pip-licenses pip-licenses --format=markdown \
  --with-urls --with-license-file --order=license
```

## Model weights (downloaded at first run, not bundled)

These are fetched by `findandseek-setup` from Hugging Face (MLX) or Ollama.
They are **not** redistributed as part of this repository; they are pulled
from their upstream source onto the user's machine at install time.

| Model | Repo / tag | License |
| --- | --- | --- |
| Embeddings (search vectors) | `mlx-community/embeddinggemma-300m-bf16`, Ollama `embeddinggemma` | **Gemma Terms of Use** (Google). Not OSI-approved. Commercial use permitted subject to the Gemma Prohibited Use Policy, which must be passed through to downstream users. https://ai.google.dev/gemma/terms |
| Summary LLM (4B) | `mlx-community/Qwen3-4B-Instruct-2507-4bit`, Ollama `qwen3:4b` | Apache-2.0 |
| Summary LLM (low-RAM, 1.7B) | `mlx-community/Qwen3-1.7B-4bit` | Apache-2.0 |
| Vision VLM (optional; benchmark only, `FINDANDSEEK_FETCH_MLX_VISION=1`) | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | **Qwen Research License**. Verify scope before any commercial use; the 3B VL weights are published under research terms, not Apache-2.0. |

OCR uses Apple's Vision framework on macOS (part of the operating system) and
Tesseract (Apache-2.0) on Linux.

## Python dependencies

Strong-copyleft dependencies are listed first because they carry obligations
that matter for redistribution.

| Package | License | Notes |
| --- | --- | --- |
| `pymupdf` | **AGPL-3.0** or Artifex commercial | PDF text/layout extraction. AGPL includes a network-use disclosure obligation. |
| `extract-msg` | **GPL-3.0** | Outlook `.msg` parsing. |
| `chardet` | LGPL-2.1 | Character-encoding detection. |
| `python-docx` | MIT | |
| `python-pptx` | MIT | |
| `openpyxl` | MIT | |
| `striprtf` | BSD-3-Clause | |
| `mail-parser` | Apache-2.0 | |
| `spacy` (+ `en_core_web_*` model) | MIT | |
| `onnxruntime` | MIT | |
| `numpy` | BSD-3-Clause | |
| `sqlite-vec` | Apache-2.0 / MIT | |
| `fastapi` | MIT | |
| `uvicorn` | BSD-3-Clause | |
| `pydantic` | MIT | |
| `watchdog` | Apache-2.0 | |
| `mcp` | MIT | |
| `psutil` | BSD-3-Clause | |
| `httpx` | BSD-3-Clause | |
| `Pillow` | HPND (MIT-style) | |
| `tiktoken` | MIT | |
| `tokenizers` | Apache-2.0 | |
| `mlx`, `mlx-lm`, `mlx-vlm`, `mlx-embeddings` | MIT | macOS only |
| `huggingface-hub` | Apache-2.0 | |

Licenses above are recorded to the best current knowledge and should be
confirmed against installed package metadata (see the `pip-licenses` command
above) before publication. Where a package is dual-licensed, the copyleft
option is listed; a commercial option may also be available from the author.
