# Benchmarks — token economics

How much does bounded retrieval actually save an AI agent working over a real
document set? We put a live Claude Sonnet 5 agent on the same three tasks twice
— once with generic filesystem tools, once with this engine — and measured the
model's own reported token usage on every turn. Three full repeats; all 18 arms
ran to a natural finish.

Full write-up: **[TOKEN_ECONOMICS_REPORT.md](TOKEN_ECONOMICS_REPORT.md)**.

## Headline

Mean of 3 repeats, all three tasks, no-cache:

| | Read-everything agent | This engine | Saving |
|---|---|---|---|
| **Cost** | $4.14 | **$2.12** | **49%** |
| **Input tokens** | 1,768,723 | **776,754** | **56%** |
| **Tool calls** | 176 | **123** | **30%** |

The engine cost less in every repeat — 55%, 26%, 61%. Deliverables were not
thinner for it: on the reconciliation task the engine's memo was *longer*
(10,201 vs 8,251 characters) at half the cost.

**The saving scales with the size of the job.** On the smallest task (a
three-file lookup) it is statistically indistinguishable from zero; on
cross-campus reconciliation and the six-department audit it is ~50%. Quote "about
half" for broad synthesis work — not for a workload of short questions. The
per-task table and its variance are in
[§3.2](TOKEN_ECONOMICS_REPORT.md#32-per-task).

The **56% token reduction** is the most reproducible figure here — it holds at
56–57% across all three tasks despite their costs differing by 6×. If you quote
one number, quote that one.

## Reading the numbers

**"Stripped engine"** in the data files means the engine in this repository —
the open engine on its own, without the commercial product's layers. It's the
label the harness writes; there is nothing removed from it.

**No-cache.** Prompt caching changes price, not which tokens get processed. A
customer can't be forced to configure it, so every figure bills input at the
full rate. This is also the conservative choice: caching discounts the
read-everything incumbent *more* (long re-read prefixes are exactly what caching
rewards), so no-cache understates the engine's advantage rather than inflating
it.

**Priced at $2/$10 per Mtok** — Sonnet 5's introductory rate, which runs through
2026-08-31. List pricing is $3/$15; that scales every dollar figure by 1.5× on
both sides and leaves the percentages unchanged. The token counts are the
durable result.

**Variance is large — quote the range, not a point.** A live agent varies
between runs, on both sides. Total savings measured 26%, 55%, and 61% across
three repeats of the identical config; the incumbent's own cost moved 23%
between runs. Say "about half" or "26–61%", never a two-decimal figure. The
smallest task's cost saving (±49 points on a mean of 34) is not a measurement at
all and is flagged as such in the report.

**Two ways this comparison flatters the engine**, both detailed in the report's
caveats: the incumbent's `grep` can't paginate (inflating its turn count), and
the engine's ingest-time precompute isn't costed (these are query-time figures,
not total cost of ownership).

## Layout

```
docs/benchmarks/
├── TOKEN_ECONOMICS_REPORT.md   # the report — read this first
├── harness/
│   └── generation_savings_benchmark.py   # the benchmark driver
├── results/
│   └── threerun_uncapped.json   # THE DATASET. 3 repeats, both sides at
│                                # 40 iters, 18/18 arms finished (end_turn)
└── transcripts/                 # raw per-turn usage for every run (audit trail)
    └── v2_*_run{0,1,2}.jsonl    # the 3-repeat dataset behind the report
```

**An earlier run was discarded, not shipped.** It records a real mistake worth
knowing about: the harness used to default to an *asymmetric* iteration ceiling
(15 for the incumbent, 8 for the engine), and the engine cannot finish a
six-department audit in eight turns. It hit the cap, returned an empty answer, and
the accounting scored that as a large cost *saving* — because failing early is
cheap. That is how a 66% headline came out of a run where the engine had failed a
task. Defaults are now 40/40 and only the corrected dataset ships here. **In an
agent benchmark, a truncated arm looks cheap: check `finish == "end_turn"` before
quoting anything.** See
[§6](TOKEN_ECONOMICS_REPORT.md#6-a-note-on-an-earlier-run-that-was-discarded).

Each transcript line is one agent turn:
`{turn, stop_reason, usage:{in,out,cache_read,cache_write}, blocks}`.
`stop_reason: "end_turn"` means the agent produced a finished deliverable;
`"tool_use"` on the last line means it was still working when it hit its
iteration budget. Every figure in the report can be recomputed from these files
using the `PRICE` table in the harness.

## What was compared

- **Incumbent** — generic filesystem tools: `list_dir`, `grep`, `read_file`.
  The conventional "point an agent at the files and let it read" approach.
- **This engine** — bounded retrieval: `search_files` returns ranked result
  cards (filename, type, one-line anchor, headline fact, confidence);
  `get_evidence` fetches full detail only when a card isn't enough.

The corpus is `company-2500`, a synthetic enterprise document set of ~2,500
files. Synthetic by design — no real documents are used anywhere in this
repository.

## Reproducing

Requires an `ANTHROPIC_API_KEY`, a running Ollama daemon for query embeddings,
and the `company-2500` index.

```bash
# from the engine repo root:
ANTHROPIC_API_KEY=...  \
GENSAV_CORPUS_ROOT=/path/to/company-2500  \
FINDANDSEEK_DB_PATH=/path/to/company-2500/index.db  \
GENSAV_INC_ITERS=40 GENSAV_SYS_ITERS=40 GENSAV_REPEATS=3 GENSAV_MAX_COST=3.0  \
uv run --with anthropic python docs/benchmarks/harness/generation_savings_benchmark.py
```

That is the exact config behind `threerun_uncapped.json`. It costs roughly **$7**
and takes about **45 minutes** — both sides now run every task to completion.

Two things to keep, if you change anything:

- **Same iteration budget on both sides.** The old 40/8 default is what
  invalidated the earlier data.
- **A spend guard high enough never to be reached.** `GENSAV_MAX_COST` truncates
  mid-answer when it trips. The old $0.75 default sat right on top of the audit's
  real incumbent cost (~$0.72), cutting off the task that mattered most.

After any run, check that every arm reports `finish: "end_turn"`. Anything that
finished on `cap` is void.

Other knobs: `GENSAV_REPEATS`, `GENSAV_MAX_COST` (per-side USD circuit breaker),
`GENSAV_TASKS` / `GENSAV_SIDES` (filter to specific tasks or sides), `GENSAV_TAG`
(output filename prefix). The harness prints both cached and no-cache cost per
side; **quote the no-cache figure.**
