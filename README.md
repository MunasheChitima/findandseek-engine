# FindandSeek Engine

[![tests](https://github.com/MunasheChitima/findandseek-engine/actions/workflows/test.yml/badge.svg)](https://github.com/MunasheChitima/findandseek-engine/actions/workflows/test.yml)
[![license: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-blue)](LICENSE.md)
[![python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

A local-first file intelligence engine. It indexes the documents, code, and
email on your machine — understanding them once at ingest time with on-device
models — and exposes that memory to any AI assistant over
[MCP](https://modelcontextprotocol.io). Nothing leaves your machine: models run
locally (in-process MLX on Apple silicon; a local [Ollama](https://ollama.com)
daemon elsewhere), the index is a local SQLite database, and there is no cloud
component.

Point it at your folders, connect Claude Desktop (or any MCP client), and ask
questions about everything you've made.

## What that looks like

> **You:** What did the plumber charge me in March?
>
> **Assistant:** *(calls `search_files`, gets one result card back)*
> $485 — `invoice_plumber_march.pdf`, 15 March, Joe's Plumbing Services,
> for an emergency pipe repair.

The assistant didn't read your folders. It got back a compact card — filename,
document type, a one-line anchor, the headline fact, a confidence band — and
that was enough. When a card *isn't* enough it calls `get_file_context` for the
relevant passages, not the whole file.

That's the design in one sentence: **understand each file once at ingest, so
retrieval is cheap forever after.** A conventional agent greps and reads whole
files on every question, and its context — and your bill — grows with the size
of the job. Measured against that baseline over a 2,500-file corpus — three full
repeats, every arm run to completion — this engine read **56% fewer input
tokens** and completed the same tasks for **about half the cost** (49% less;
range 26–61% across repeats). The saving grows with the size of the job: it is
near zero on a single-file lookup and ~49% on six-department synthesis. See
[docs/benchmarks/](docs/benchmarks/) for the raw per-turn transcripts, the
per-task spread, and the caveats.

For how this compares to classic RAG, knowledge graphs, cloud vector-DB search,
and plain agentic grep — and where it is *not* the right tool — see
[docs/COMPARISON.md](docs/COMPARISON.md).

## What it can read

| | |
|---|---|
| Documents | `.pdf` `.docx` `.doc` `.odt` `.rtf` `.txt` `.md` |
| Spreadsheets | `.xlsx` `.xls` `.ods` `.csv` `.tsv` |
| Presentations | `.pptx` `.ppt` `.odp` |
| Email | `.eml` `.msg` |
| Images (OCR) | `.png` `.jpg` `.jpeg` `.tiff` `.heic` |

Scanned PDFs and photos go through OCR (Apple Vision on macOS, tesseract on
Linux). Legacy `.doc`/`.odt`/`.ppt` need LibreOffice installed on Linux.

macOS OCR runs through a small pre-built Swift helper. If you'd rather not run a
binary you didn't compile, build it from the source beside it — see
[`tools/ocr/README.md`](tools/ocr/README.md).

## Requirements

- macOS on Apple silicon (inference is in-process MLX), **or** Linux with
  [Ollama](https://ollama.com) installed and running (`ollama serve`) plus
  `tesseract-ocr` for image/scan OCR and optionally LibreOffice for legacy
  `.doc`/`.odt` formats
- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- ~4 GB of disk for model weights

Windows isn't a supported target for inference, but the watcher and supervisor
are cross-platform — if you have Ollama running you can drive it with
`findandseek-up` (see [Run it on boot](#run-it-on-boot)).

## Quickstart

```bash
git clone https://github.com/MunasheChitima/findandseek-engine
cd findandseek-engine
uv sync
uv run findandseek-setup        # downloads model weights (Qwen3, embeddings)
uv run findandseek-api          # starts the local engine on 127.0.0.1:8775
```

On macOS `findandseek-setup` fetches the MLX weights; on Linux it pulls the
equivalent Ollama models (`qwen3:4b` for summaries, `embeddinggemma` for
embeddings — the same model families, so an index built on one platform stays
compatible with the other).

Tell the engine what to index (repeat for each folder):

```bash
curl -X POST http://127.0.0.1:8775/folders \
  -H 'Content-Type: application/json' \
  -d '{"path": "'"$HOME"'/Documents"}'
```

Indexing runs in the background; check progress any time:

```bash
curl http://127.0.0.1:8775/health
```

### Background ingest (keep the index live)

Ingest is split across three long-lived processes: the **worker** does the
indexing, the **watcher** detects file changes and enqueues them, and the
**API** serves control/query. `findandseek-api` alone enqueues folders but does
not index them — you need the worker running too.

The simplest way is one command that runs and supervises all three (restarts a
crashed one, stops them cleanly on Ctrl-C):

```bash
uv run findandseek-up            # worker + watcher + API
uv run findandseek-up --no-api   # background ingest only (worker + watcher)
```

The watcher is cross-platform (it uses `watchdog`: inotify on Linux, FSEvents on
macOS, ReadDirectoryChangesW on Windows), so new and changed files in your
watched folders are re-indexed automatically.

#### Run it on boot

On macOS use `findandseek-service` (launchd). On Linux, a systemd user unit is
provided at [`deploy/findandseek.service`](deploy/findandseek.service) — edit
the two marked paths, then `systemctl --user enable --now findandseek` (plus
`loginctl enable-linger "$USER"` to keep it running while logged out). On
Windows, point Task Scheduler at `uv run findandseek-up`.

## Connect your AI assistant

Add the MCP server to Claude Desktop (`claude_desktop_config.json`) or any
other MCP client:

```json
{
  "mcpServers": {
    "findandseek": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/findandseek-engine", "findandseek-engine-mcp"]
    }
  }
}
```

Your assistant now has these tools against your local index:

| Tool | What it does |
| --- | --- |
| `search_files` | Hybrid semantic + keyword search over everything indexed |
| `get_summary` | On-device summary of a file |
| `get_file_context` | A file's summary plus its most relevant chunks |
| `get_chunk` | One chunk of a file, with optional neighbours |
| `find_entity` | Look up people, orgs, emails, and other extracted entities |
| `query_typed_facts` / `aggregate_typed_facts` | Query structured facts (amounts, dates, …) extracted at ingest |
| `list_recent` | Recently created or modified files |
| `index_status` | What's indexed, what's pending |
| `propose_organize_plan` | Draft a cleanup/reorganisation plan for a folder |

**Nothing here moves your files.** `propose_organize_plan` writes a proposal to
the index and returns a plan id; applying it is user-gated in the FindandSeek
app, journaled, and reversible. There is deliberately no MCP tool that touches
the filesystem.

**Building an agent on this?** [docs/FOR_AGENTS.md](docs/FOR_AGENTS.md) is the
case for why it suits agents; [docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md) is the
operating manual — the tools, the escalation ladder, and the anti-patterns to
put in your agent's system prompt.

## REST API

Everything the MCP tools do is also available over plain HTTP on
`127.0.0.1:8775`, if you'd rather not use MCP:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Index and ingest status — start here |
| `GET /search?q=…&scope=…` | Hybrid search; same ranking as `search_files` |
| `GET /browse?type=…&scope=…` | List indexed files by document type |
| `GET /recent?days=7` | Recently modified files |
| `GET /entity?type=org&value=…` | Entity lookup |
| `GET /facts` | Typed facts (amounts, dates) with filters |
| `POST /folders` | Add a folder to the index |
| `DELETE /folders` | Stop watching a folder and purge it |

Most read endpoints take a `scope` parameter — a folder path that restricts
results to that folder and everything beneath it. The full route list is in
[`find_and_seek/api/server.py`](find_and_seek/api/server.py).

**There is no authentication.** The API binds to `127.0.0.1` and trusts every
caller, which is fine for a local single-user tool and unsafe the moment you
change it — `FINDANDSEEK_API_HOST=0.0.0.0` would expose search over everything
you've indexed, to anyone who can reach the port. See
[SECURITY.md](SECURITY.md).

## Running tests

The suite is hermetic — no model weights or network needed, and it runs in
about 15 seconds:

```bash
export FINDANDSEEK_TEST=1
export FINDANDSEEK_PERFORMANCE_MODE=full   # required on battery — smooth mode pauses ingest
uv run pytest tests -q
```

One thing to know: in test mode the embedding backend returns deterministic
pseudo-vectors, so ranking tests exercise the keyword and fusion paths rather
than real semantic recall. If you're changing retrieval quality, validate
against a real index with [`find_and_seek/eval/`](find_and_seek/eval/) too.

## Configuration

Behaviour is tunable via `FINDANDSEEK_*` environment variables (ports, OCR,
search weights, file-size caps, battery pausing). See
[`find_and_seek/config/settings.py`](find_and_seek/config/settings.py) — every
knob is documented where it's read.

## Relationship to the FindandSeek product

This repository is the open engine that powers
[FindandSeek](https://findandseek.app). The commercial product adds a native
macOS app, media understanding (images, video, speech), the organisational
Intelligence Engine layer (HCP triage cards, governed execution, fleet
deployment), and a **permission-aware agent-deployment layer** — role-based
access control that scopes every deployed agent to the files the acting user is
already allowed to see, so agents inherit their user's permissions instead of
seeing the whole index. The engine here is complete and useful on its own; the
product is where the enterprise machinery lives.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). In short: tests run on every PR, new
behaviour needs a test, and first-time contributors are asked to sign a CLA
(the project keeps a single copyright holder so it can offer the commercial
license and the future Apache-2.0 release).

## License

[Business Source License 1.1](LICENSE.md) (BUSL-1.1). This is a
source-available license, not an open-source one: you can read, run, and modify
the engine freely, including inside a company for your own use, but you can't
sell it to others or run it as a service for them. Each release converts to
Apache 2.0 on the second anniversary of its publication.

"FindandSeek" is a trademark of the licensor. The license covers the code, not
the name.

**What that means in practice** (plain-language summary, not legal advice, the
[license text](LICENSE.md) governs):

- **Free to run inside your own team or company.** Using the engine for your
  own operations, on your own machines, is free, and it stays free no matter how
  commercial your company is. Personal use, learning, and research are free too.
- **You can't sell it to others or run it for them.** You may not host it as a
  service for third parties, and you may not be paid to deploy, operate, or
  provide it to someone else's business. Turning the engine into a product or a
  managed service for others needs a commercial license. Licensing at
  [findandseek.app](https://findandseek.app), and it comes with a purpose-built
  agent-deployment layer: role-based access control that scopes every agent to
  the permissions of the user it acts for.
- **It opens up eventually.** Every release becomes Apache 2.0 two years after
  it ships, at which point these restrictions fall away for that version.

## Acknowledgements

To Brittany, and to my sunshine — thank you for your love and support. This
exists because of you.
