"""Search-recall eval against a hand-labeled gold set on a REAL index.

The existing eval/run_eval.py scores ranking on the synthetic gold corpus. This
one measures the thing F-1.2 is about: does an OBLIQUE query (whose words don't
overlap the document's) still reach the right file, without the confidence gate
either over-suppressing it OR fabricating a match for a query about nothing.

Three query kinds:
  recall          — oblique; a target id must appear in top-k (the F-1.2 target)
  control_recall  — aligned vocabulary; must keep working (regression guard)
  negative        — no real target; must return [] / not match (precision guard)

Gold (jsonl): {"q", "want": [file_id...], "type"}. Real filenames ⇒ gold lives
OUTSIDE the repo (~/findandseek-audit-tmp/search_gold.jsonl); never commit it.
Run serially (loads embed + reranker; +classifier if expansion is on).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_GOLD = "~/findandseek-audit-tmp/search_gold.jsonl"


def run(db_path: str, gold_path: str, k: int = 5) -> dict:
    from find_and_seek.db.connection import get_connection
    from find_and_seek.search.hybrid import search

    conn = get_connection(os.path.expanduser(db_path))
    gold = [json.loads(l) for l in Path(os.path.expanduser(gold_path)).read_text().splitlines() if l.strip()]

    rows, by_kind = [], {}
    for g in gold:
        hits, _ = search(conn, g["q"], limit=k)
        ids = [h.file_id for h in hits]
        want = set(g["want"])
        kind = g["type"]
        if kind == "negative":
            ok = len(ids) == 0
        else:
            ok = bool(want & set(ids))
        rank = next((i + 1 for i, fid in enumerate(ids) if fid in want), None)
        rows.append({"q": g["q"], "type": kind, "ok": ok, "rank": rank,
                     "n_hits": len(ids), "top_ids": ids[:5]})
        b = by_kind.setdefault(kind, {"n": 0, "ok": 0})
        b["n"] += 1
        b["ok"] += ok

    summary = {
        "k": k,
        "by_kind": {kk: {"n": v["n"], "passed": v["ok"], "rate": round(v["ok"] / v["n"], 3)}
                    for kk, v in sorted(by_kind.items())},
        "overall": round(sum(r["ok"] for r in rows) / len(rows), 3),
    }
    print(json.dumps(summary, indent=2))
    for r in rows:
        mark = "ok " if r["ok"] else "FAIL"
        rk = f"#{r['rank']}" if r["rank"] else f"{r['n_hits']} hits"
        print(f"  [{mark}] {r['type']:14} {rk:8} {r['q']}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="~/.findandseek/index.db")
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    run(a.db, a.gold, a.k)


if __name__ == "__main__":
    main()
