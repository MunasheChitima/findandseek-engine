"""Run the search-quality gold set and report hit@k / MRR / latency.

Usage:
    python -m find_and_seek.eval.run_eval [--db PATH] [--k 10] [--report DIR]

Scores the hybrid search() against eval/gold.jsonl. A returned hit counts as
relevant if its path/filename contains any `expect` substring (case-insensitive)
or its file_id is in `expect_ids`. Per-query crashes are caught and recorded as
errors (a miss) so an index defect surfaces as a number instead of aborting the
run. Always point --db at a COPY of the index, never the live DB.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from find_and_seek.db.connection import init_db
from find_and_seek.organize.taxonomy import canonicalize_type
from find_and_seek.search.hybrid import search

GOLD = Path(__file__).parent / "gold.jsonl"
DEFAULT_DB = Path.home() / ".findandseek" / "index.eval.db"
KS = (1, 3, 5, 10)


def load_gold() -> list[dict]:
    return [json.loads(line) for line in GOLD.read_text().splitlines() if line.strip()]


def is_relevant(hit, item: dict) -> bool:
    if hit.file_id in item.get("expect_ids", []):
        return True
    hay = f"{hit.path} {hit.filename}".lower()
    return any(sub.lower() in hay for sub in item.get("expect", []))


def eval_query(conn, item: dict, k: int) -> dict:
    t0 = time.perf_counter()
    try:
        hits, _ = search(conn, item["query"], "all", k, None)
        error = None
    except Exception as e:  # an index defect must not abort the whole run
        hits, error = [], f"{type(e).__name__}: {e}"
    latency_ms = (time.perf_counter() - t0) * 1000

    first_rank = None
    relevant_count = 0
    relevant_hits = []
    for i, h in enumerate(hits, start=1):
        if is_relevant(h, item):
            relevant_count += 1
            relevant_hits.append(h)
            if first_rank is None:
                first_rank = i

    # Type precision: of the RELEVANT hits this query retrieved, what fraction
    # carry the expected type? Scoring over *all* k results is wrong — a
    # known-item search legitimately surfaces related docs of other types at
    # lower ranks, so demanding type purity across k caps precision artificially
    # and even credits an irrelevant doc that happens to match the type. We also
    # compare canonical types on both sides so taxonomy drift (resume↔cv,
    # slide_deck↔slide-deck) and the raw document_type the search path returns
    # line up with the gold's canonical expectation.
    type_prec = None
    if item.get("expect_type") and relevant_hits:
        want = canonicalize_type(item["expect_type"])
        type_prec = sum(
            1 for h in relevant_hits if canonicalize_type(h.document_type) == want
        ) / len(relevant_hits)

    return {
        "id": item["id"],
        "intent": item.get("intent", "?"),
        "query": item["query"],
        "error": error,
        "first_rank": first_rank,
        "relevant_in_topk": relevant_count,
        "rr": (1.0 / first_rank) if first_rank else 0.0,
        "type_precision": type_prec,
        "latency_ms": round(latency_ms, 1),
        "top": [
            {"rank": i, "file_id": h.file_id, "filename": h.filename,
             "document_type": h.document_type, "relevant": is_relevant(h, item)}
            for i, h in enumerate(hits[:5], start=1)
        ],
    }


def summarize(results: list[dict], k: int) -> dict:
    n = len(results)
    hit_at = {kk: sum(1 for r in results if r["first_rank"] and r["first_rank"] <= kk) / n for kk in KS if kk <= k}
    mrr = sum(r["rr"] for r in results) / n
    errors = [r["id"] for r in results if r["error"]]
    lat = sorted(r["latency_ms"] for r in results)
    tp = [r["type_precision"] for r in results if r["type_precision"] is not None]
    by_intent: dict[str, dict] = {}
    for r in results:
        b = by_intent.setdefault(r["intent"], {"n": 0, "hit": 0})
        b["n"] += 1
        b["hit"] += 1 if r["first_rank"] else 0
    return {
        "n": n, "k": k, "mrr": round(mrr, 3),
        "hit_at": {f"hit@{kk}": round(v, 3) for kk, v in hit_at.items()},
        "errors": errors,
        "latency_ms": {"p50": lat[n // 2], "max": lat[-1]},
        "type_precision_mean": round(sum(tp) / len(tp), 3) if tp else None,
        "by_intent": {it: {**b, "hit_rate": round(b["hit"] / b["n"], 3)} for it, b in by_intent.items()},
    }


def render_md(summary: dict, results: list[dict], db: Path) -> str:
    L = [f"# Search eval — {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
         f"\nDB: `{db}` · queries: {summary['n']} · k={summary['k']}\n",
         "## Aggregate",
         f"- **MRR**: {summary['mrr']}",
         "- " + " · ".join(f"**{kk}**: {v}" for kk, v in summary["hit_at"].items()),
         f"- **Latency** p50/max: {summary['latency_ms']['p50']} / {summary['latency_ms']['max']} ms",
         f"- **Errors**: {len(summary['errors'])} {summary['errors'] or ''}",
         f"- **Type precision (mean)**: {summary['type_precision_mean']}",
         "\n### By intent",
         "| intent | n | hit-rate |", "|---|---|---|"]
    for it, b in summary["by_intent"].items():
        L.append(f"| {it} | {b['n']} | {b['hit_rate']} |")
    L += ["\n## Per query", "| id | intent | first_rank | rel@k | lat ms | result |", "|---|---|---|---|---|---|"]
    for r in results:
        if r["error"]:
            res = f"❌ {r['error'][:50]}"
        elif r["first_rank"]:
            res = f"✅ rank {r['first_rank']}"
        else:
            top = r["top"][0]["filename"][:32] if r["top"] else "—"
            res = f"⚠️ miss (top: {top})"
        L.append(f"| {r['id']} | {r['intent']} | {r['first_rank'] or '—'} | {r['relevant_in_topk']} | {r['latency_ms']} | {res} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--report", type=Path, default=Path(__file__).parent / "reports")
    args = ap.parse_args()

    if not args.db.exists():
        raise SystemExit(f"DB not found: {args.db}\nCopy the live index first: cp ~/.findandseek/index.db {args.db}")

    conn = init_db(args.db)
    gold = load_gold()
    print(f"Running {len(gold)} gold queries against {args.db} (k={args.k}) ...\n")
    results = [eval_query(conn, item, args.k) for item in gold]
    summary = summarize(results, args.k)

    for r in results:
        mark = "❌" if r["error"] else ("✅" if r["first_rank"] else "⚠️ ")
        detail = r["error"][:60] if r["error"] else (f"rank {r['first_rank']}" if r["first_rank"] else "MISS")
        print(f"  {mark} {r['id']:<26} {detail:<22} {r['latency_ms']:>7.1f}ms")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    args.report.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    (args.report / f"eval-{stamp}.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    md = args.report / f"eval-{stamp}.md"
    md.write_text(render_md(summary, results, args.db))
    print(f"\nReport: {md}")


if __name__ == "__main__":
    main()
