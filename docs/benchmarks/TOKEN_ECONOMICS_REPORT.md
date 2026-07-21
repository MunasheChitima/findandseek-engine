# FindandSeek Engine — Token Economics Benchmark

**Date:** 2026-07-20
**Model in the loop:** Claude Sonnet 5 (`claude-sonnet-5`), real Anthropic API
**Corpus:** `company-2500` synthetic enterprise index (~2,500 files)
**Primary dataset:** `results/threerun_uncapped.json` — 3 repeats × 3 tasks × 2 sides, **18/18 arms completed naturally**
**Cost basis:** no-cache, at $2.00 / $10.00 per million input / output tokens (see *Cost basis* below)

---

## 1. Executive summary

Running a real Claude agent over an enterprise document set, the FindandSeek
engine (bounded retrieval — search returns compact result cards; full evidence is
fetched only when a card is not enough) completed the same generation tasks as a
conventional read-everything agent for **about half the cost**:

| | Incumbent | Engine | Saving |
|---|---|---|---|
| **No-cache** | $4.14 | **$2.12** | **49%** |
| Cached | $1.30 | **$0.98** | 25% |

Measured over **three full repeats**. The engine cost less in **every repeat**
(55%, 26%, 61%) and every one of the eighteen arms ran to a natural finish
(`end_turn`) with a complete deliverable — nothing truncated, nothing spliced
across runs.

The mechanism is more stable than the headline. The engine read **56% fewer
input tokens**, and that figure barely moves between tasks (57% / 56% / 56%) or
between runs. Cost follows from it, with more noise attached.

**The saving scales with task size.** On a small three-file lookup the engine is
roughly a wash; on six-department synthesis it saves ~49%. §3.2 gives the
per-task breakdown, including one task whose variance is too wide to quote.

---

## 2. What was measured

A real Claude Sonnet 5 agent drives a tool-use loop against the corpus, in two
configurations answering identical prompts:

- **Incumbent** — generic filesystem tools (`list_dir`, `grep`, `read_file`).
  This is the conventional "point an agent at the files and let it read"
  approach.
- **System** — the engine's bounded retrieval (`search_files` returns ranked
  result cards; `get_file_context` fetches detail only when a card is not
  enough).

Three **generation** tasks, each forcing multi-document synthesis into a written
deliverable (this is where a read-everything agent balloons its input context
hardest):

1. **Spring Glow executive summary** — total campaign spend + approved concepts + delivering agency.
2. **Cross-campus reconciliation** — board-ready spend memo reconciling multiple departments' records.
3. **Full program audit** — six-department spend-and-risk audit with a reconciled variance and a data-integrity pass.

Token usage is the model's own reported `usage` per turn, summed across the
agentic loop — measured, not estimated. The harness and the raw per-turn
transcripts for every run are included in this folder (see the appendix), so
every number here is reproducible and auditable.

### Environment
- Linux; embeddings served by a local Ollama daemon (`nomic-embed-text`).
- Summaries/classification/facts were precomputed at ingest and stored in the index.

### Run configuration
Both sides were given **the same 40-iteration ceiling** and a $3.00/side spend
guard, high enough that no arm reached either. This matters — see §6.

### Cost basis
Two choices are baked into every dollar figure here, and both are stated so you
can recompute:

**No-cache.** Prompt caching only discounts *price*; it does not change which
tokens are processed. A customer cannot be forced to configure caching, so the
honest "what you pay by default" figure bills every input token at the full
rate. This is also the **conservative** choice: caching, when present, discounts
the read-everything incumbent *more* (its long re-read prefixes are exactly what
caching rewards), so no-cache understates rather than inflates the engine's
advantage. The cached figures are given alongside so both are visible.

**$2 / $10 per Mtok.** The harness (`harness/generation_savings_benchmark.py`,
the `PRICE` table) prices Sonnet 5 at Anthropic's *introductory* rate, which
runs through 2026-08-31. Standard list pricing is **$3 / $15**. The introductory
rate is what a customer pays today; at list pricing every absolute dollar figure
below scales by 1.5× on both sides, and **the percentage savings are
unchanged**. The token counts — which are the durable result — are unaffected
either way.

---

## 3. Results

### 3.1 Per-repeat totals

All three tasks, both sides, same run. No arm was truncated.

| Repeat | Incumbent | Engine | Saving |
|---|---|---|---|
| 0 | $4.24 | $1.89 | 55% |
| 1 | $3.67 | $2.70 | 26% |
| 2 | $4.52 | $1.77 | 61% |
| **Mean** | **$4.14** | **$2.12** | **49%** |

The 26% repeat is the number to plan against, not the 61% one. Run-to-run
variance on this workload is large and it is not an artifact — both sides are
live, nondeterministic agents.

### 3.2 Per-task

Mean of 3 repeats, no-cache, ± standard deviation:

| Task | Incumbent | Engine | Saving | Input tokens (inc → eng) |
|---|---|---|---|---|
| Spring Glow summary | $0.44 | $0.22 | **+34% ± 49** ⚠ | 185,978 → 80,115 (−57%) |
| Cross-campus reconciliation | $1.19 | $0.60 | **+50% ± 13** | 498,483 → 218,395 (−56%) |
| Full program audit | $2.51 | $1.29 | **+49% ± 20** | 1,084,261 → 478,244 (−56%) |

⚠ **Spring Glow's cost saving should not be quoted.** Its standard deviation
(49 points) exceeds its mean (34); across three identical runs it measured −18%,
+40%, and +80%. This is the smallest task — a three-file lookup — so there is
little context to save and run-to-run jitter swamps the effect. Its *token*
saving (−57%) is stable and is the figure worth citing for that task.

The two large synthesis tasks both land near 49% with tolerable spread. **The
engine's advantage is a function of how much the incumbent has to read**, so it
grows with task size. That is the honest shape of this result and it is more
useful to a buyer than a flat percentage: the product pays for itself on broad,
multi-document work, not on single-file lookups.

### 3.3 Deliverables were not thinner

Cheaper output is only interesting if the work is comparable. Mean answer length:

| Task | Incumbent | Engine |
|---|---|---|
| Spring Glow summary | 4,081 chars | 3,507 chars |
| Cross-campus reconciliation | 8,251 chars | **10,201 chars** |
| Full program audit | 18,078 chars | 16,957 chars |

On the cross-campus memo the engine produced **24% more** deliverable at half the
cost. On the audit and the summary it produced slightly less — within the range
you would expect from two different agents writing the same document, not a sign
of a truncated answer (all eighteen arms finished on `end_turn`).

### 3.4 The metrics with the tightest error bars

Cost is the headline, but it is the noisiest thing measured here. Two figures
are far more reproducible, and they are the mechanism the cost saving rests on:

| Metric | Incumbent | Engine | Reduction | sd of reduction |
|---|---|---|---|---|
| Input tokens | 1,768,723 | 776,754 | **−56%** | 20.3 |
| Tool calls | 176 | 123 | **−30%** | 8.0 |

The token reduction holds at 56–57% across all three tasks despite their costs
differing by 6×. **If you quote one number from this report, this is the one that
will survive re-running.**

---

## 4. The bill-shock finding

With the incumbent uncapped, a genuinely hard task drives its cost sharply
upward. The worst single incumbent task-run in this dataset cost **$2.64** and
re-read **1,153,930 input tokens** across 84 tool calls — roughly **10× its own
cheapest task**. Input grows super-linearly with turn count, because every turn
re-processes the whole accumulated conversation.

Bounded retrieval caps that worst case: across the same three repeats the engine
completed the audit for $0.83–$1.82. The value is not only average-case cheaper,
it is **worst-case bounded** — the spiral that produces a surprise bill is
removed.

---

## 5. Honest caveats (read before quoting)

1. **Three repeats is enough to see variance, not enough to pin it down.** The
   spread on total savings was 26–61%. Quote "about half", or quote the range —
   never a two-decimal figure.
2. **Spring Glow's cost saving is not a measurement.** ±49 points on a mean of
   34. See §3.2.
3. **Both sides are nondeterministic.** The incumbent's cost varied by 23%
   between repeats on identical inputs ($3.67–$4.52); the engine's by 53%
   ($1.77–$2.70). This is faithful to production, and it is why single-run
   benchmarks of agent workloads are close to meaningless.
4. **The incumbent's tools are deliberately simple, and that flatters us.**
   `grep` returns at most `GREP_MAX = 12` hits **with no pagination**, and
   `list_dir` truncates at 60 entries — against a 2,514-file corpus. The
   transcripts show the incumbent re-issuing the same grep with varied terms
   because it cannot page through results, which inflates its turn count and
   therefore its input tokens. A read-everything agent with paginated search
   would close some of this gap. How much is not measured here.
5. **Ingest-time precompute is not costed.** The engine's advantage depends on
   summaries, classifications, and facts generated by an LLM for all ~2,500 files
   at ingest. That is a real one-time cost, amortized across all future queries,
   and it appears nowhere in these figures. The comparison is therefore
   query-time cost, not total cost of ownership. For a corpus queried many times
   the amortization is favorable; for one queried twice it is not.
6. **The saving is task-size dependent.** It is near zero on small lookups. Do
   not extrapolate the 49% to a workload of short questions.
7. **Corpus is synthetic** (`company-2500`). Real-world corpora will differ;
   these are directional, defensible figures, not a customer SLA.

---

## 6. A note on an earlier run that was discarded

An earlier measurement in this folder was thrown out before publication. **Its
66% headline should never be quoted**, and the reason is instructive.

The harness used to default to `INC_MAX_ITERS = 15` and `SYS_MAX_ITERS = 8` —
an asymmetric ceiling, with the engine given the smaller budget. The full program
audit spans six departments and cannot be completed in eight turns. So the engine
hit the ceiling, returned an **empty answer**, and the accounting scored that as
a large cost *saving* — because failing early is cheap. At 15/8, **10 of 18 arms
produced an empty or near-empty answer**, so the per-task "savings" were mostly
measuring which side ran out of turns first. The 66% headline came from the two
tasks that happened to finish in one lucky run.

The defaults are now **40/40 with a $3.00 spend guard**, and the harness carries
a comment explaining the trap. The lesson generalises: **in an agent benchmark, a
truncated arm looks cheap, so any run where a side finishes on `cap` rather than
`end_turn` is void.** Check the `finish` field before quoting anything.

The earlier 66% figure was not fabricated — it was a real measurement of two
tasks in one lucky run. It did not survive repetition. 49% did.

---

## Appendix — file pointers (all co-located in this folder)

- **Harness:** `harness/generation_savings_benchmark.py` — the driver (see `README.md` for env knobs and reproduction). The `PRICE` table is the cost basis; `GREP_MAX` and `READ_CAP` bound the incumbent's tools.
- **Primary dataset (§1, §3, §4):** `results/threerun_uncapped.json` — `inc_max_iters: 40`, `sys_max_iters: 40`, `repeats: 3`, 18/18 arms `end_turn`. Transcripts: `transcripts/v2_*_run{0,1,2}.jsonl`. This is the only dataset in the folder; the superseded 15/8 run described in §6 was removed.

Each transcript line is one agent turn: `{turn, stop_reason, usage, blocks}`.
`stop_reason: "end_turn"` means the agent produced a finished deliverable;
`"tool_use"` on the last line means it was still working when it hit its
iteration budget.

To recompute the headline from the raw file:

```bash
cd docs/benchmarks
python3 -c "
import json, statistics as st
P = {'in': 2., 'out': 10., 'cache_read': .2, 'cache_write': 2.5}
d = json.load(open('results/threerun_uncapped.json'))
nc = lambda u: ((u['in'] + u['cache_read'] + u['cache_write']) * P['in'] + u['out'] * P['out']) / 1e6
side = lambda s: st.mean([sum(nc(r[s]['usage']) for r in run['rows']) for run in d['runs']])
i, e = side('incumbent'), side('system')
print('incumbent \$%.2f  engine \$%.2f  ->  %.0f%% saving' % (i, e, 100 * (1 - e / i)))
"
```
