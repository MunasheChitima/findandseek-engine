# Contributing

Thanks for your interest. A few ground rules to know before opening a PR:

## Contributor License Agreement

FindandSeek is licensed under FSL-1.1-ALv2, which requires the project to
retain a single copyright holder. Before your first PR can be merged you'll be
asked to agree to a CLA assigning copyright in your contribution to the
licensor. This is currently a manual step — expect to be asked in a PR comment,
not by a bot. If that's not acceptable to you, that's completely fair — please
open an issue describing the change instead, and we may implement it
independently.

## Development

```bash
uv sync
export FINDANDSEEK_TEST=1
export FINDANDSEEK_PERFORMANCE_MODE=full
uv run pytest tests -q
```

The test suite is hermetic — pseudo-embeddings, no model weights, no network —
so it runs anywhere in well under a minute. It runs on every PR via
[`.github/workflows/test.yml`](.github/workflows/test.yml) (Python 3.11 and
3.13, plus an import check over every console-script entry point) and needs to
be green before merge. New behaviour needs a test alongside it.

Note what the hermetic setup does *not* cover: under `FINDANDSEEK_TEST=1` the
embedding backend returns pseudo-vectors and query expansion is disabled, so
ranking tests exercise the keyword and fusion paths rather than real semantic
recall. If you're changing retrieval quality, validate it against a real index
with [`find_and_seek/eval/`](find_and_seek/eval/) as well.

## Invariants worth knowing before you change things

Five properties in this codebase are load-bearing and cheap to break by
accident. Most have a test guarding them; all have been broken at least once.

**No MCP tool may write to the filesystem.** An assistant can search, summarise,
and read. It cannot move, rename, or delete. `propose_organize_plan` records a
proposal and returns an id; applying is user-gated, journaled, and reversible.
If you add a tool, this is the boundary to preserve — see
[SECURITY.md](SECURITY.md).

**Plans that were applied or undone must never be garbage-collected.** `undo`
replays `migration_journal` rows that belong to the plan; delete the plan and an
apply becomes irreversible, with the user's files already moved. Only
`PRUNABLE_STATUSES` in `organize/plan_store.py` are eligible, and callers choose
how many to keep, never which statuses qualify.

**Folder scoping goes through `db/scope.py`.** It is a trust boundary, not just
a filter — it's what an agent passes to say "only look in Downloads". Hand-rolled
`LIKE 'scope%'` over-matches two ways: sibling directories sharing a prefix, and
`_`/`%` in folder names acting as wildcards. `tests/test_scope.py` covers both.

**Logs and error records must never contain document text.** `safe_log.py`
exists to keep file contents out of `ingest_queue.last_error` and the
diagnostics log. That's a privacy promise enforced in code, not a style choice.

**The test corpus is generated, not committed.** Edit
`tests/generate_corpus.py`; conftest rebuilds the fixtures automatically. Never
hand-edit the binaries — committed artifacts drift from their generator, and
that drift once hid real personal data inside a PDF where no text search could
find it.

## Scope

This repository is the document/search/MCP engine only. Issues about the
macOS app, media understanding, licensing/activation, or enterprise
deployment belong at support@findandseek.app rather than this tracker.

## Pace

This project is maintained by one person alongside the commercial product.
Issues and PRs are read, but responses may take a while. Steady beats fast.
