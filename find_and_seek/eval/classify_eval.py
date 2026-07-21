"""Classification accuracy + calibration eval against a hand-labeled gold set.

This is the engine-quality gauge the pipeline was missing: the MCP/search tests
assert the engine RESPONDS; the search gold set scores ranking. Neither says
whether `document_type` is RIGHT, or whether the stored confidence means
anything. This eval measures both, so any engine change (prompt, taxonomy,
extraction, a different model) becomes a scored experiment instead of vibes.

Gold format (jsonl): {"file_id", "path", "filename", "label", "alts": [...]}
`alts` are acceptable alternates for genuinely gray documents. The gold file and
the reports contain REAL filenames, so both live OUTSIDE the repo by default
(~/findandseek-audit-tmp/) — never commit them.

Run (serially — never alongside another model job):
    ./.venv/bin/python -m find_and_seek.eval.classify_eval [--db PATH] [--gold PATH]
        [--model HF_REPO]      # try a different classifier model, same gauge
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_GOLD = "~/findandseek-audit-tmp/classify_gold.jsonl"
DEFAULT_REPORTS = "~/findandseek-audit-tmp/classify-reports"


def _file_text(conn, file_id: int, limit: int = 6) -> str:
    rows = conn.execute(
        "SELECT text FROM file_chunks WHERE file_id=? ORDER BY chunk_index LIMIT ?",
        (file_id, limit),
    ).fetchall()
    return "\n".join((r[0] if isinstance(r, tuple) else r["text"]) for r in rows)


def run(db_path: str, gold_path: str, model: str | None = None) -> dict:
    if model:
        os.environ["FINDANDSEEK_MLX_CLASSIFY_MODEL"] = model
    from find_and_seek.db.connection import get_connection
    from find_and_seek.organize.classify import classify_document

    conn = get_connection(os.path.expanduser(db_path))
    gold = [json.loads(l) for l in Path(os.path.expanduser(gold_path)).read_text().splitlines() if l.strip()]

    results = []
    t0 = time.perf_counter()
    for g in gold:
        row = conn.execute(
            "SELECT id, path, filename FROM files WHERE id=? OR path=?",
            (g.get("file_id", -1), g["path"]),
        ).fetchone()
        if not row:
            results.append({**g, "pred": None, "conf": None, "outcome": "missing"})
            continue
        text = _file_text(conn, row["id"])
        res = classify_document(row["path"], row["filename"], text, conn=conn)
        ok_set = {g["label"], *g.get("alts", [])}
        if res.slug == "needs-review":
            outcome = "abstain_correct" if "needs-review" in ok_set else "abstain"
        else:
            outcome = "correct" if res.slug in ok_set else "wrong"
        results.append({**g, "pred": res.slug, "conf": res.confidence,
                        "suggested": res.suggested, "outcome": outcome,
                        "body_chars": len(text.strip())})
    elapsed = time.perf_counter() - t0

    placed = [r for r in results if r["outcome"] in ("correct", "wrong")]
    abstains = [r for r in results if r["outcome"].startswith("abstain")]
    correct = [r for r in placed if r["outcome"] == "correct"]
    # Calibration: accuracy within each confidence bucket. A calibrated engine
    # is MORE accurate at 'high' than at 'medium' — if the buckets are flat, the
    # confidence carries no information and the hedging is theatre.
    buckets: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in placed:
        b = buckets[r["conf"] or "none"]
        b["n"] += 1
        b["correct"] += r["outcome"] == "correct"
    calibration = {k: {"n": v["n"], "accuracy": round(v["correct"] / v["n"], 3)}
                   for k, v in sorted(buckets.items())}
    misses = [{"filename": r["filename"], "label": r["label"], "pred": r["pred"],
               "conf": r["conf"], "body_chars": r.get("body_chars")}
              for r in results if r["outcome"] == "wrong"]
    confusion = Counter((r["label"], r["pred"]) for r in results if r["outcome"] == "wrong")

    summary = {
        "n": len(results),
        "model": model or os.environ.get("FINDANDSEEK_MLX_CLASSIFY_MODEL", "default(Qwen3-4B)"),
        "placed": len(placed),
        "abstained": len(abstains),
        "abstain_correct": sum(1 for r in abstains if r["outcome"] == "abstain_correct"),
        "missing": sum(1 for r in results if r["outcome"] == "missing"),
        "accuracy_on_placed": round(len(correct) / len(placed), 3) if placed else None,
        "strict_accuracy": round(
            (len(correct) + sum(1 for r in abstains if r["outcome"] == "abstain_correct"))
            / len(results), 3),
        "calibration": calibration,
        "top_confusions": [{"gold": a, "pred": b, "n": n} for (a, b), n in confusion.most_common(8)],
        "elapsed_s": round(elapsed, 1),
    }

    reports = Path(os.path.expanduser(DEFAULT_REPORTS))
    reports.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (reports / f"classify-{stamp}.json").write_text(
        json.dumps({"summary": summary, "results": results, "misses": misses}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"report: {reports}/classify-{stamp}.json")
    return summary


def test_filename_bias() -> bool:
    """Smoke test: verify that content signals dominate over filename keywords in
    the heuristic and fast-classify paths.  No DB or LLM required — exercises the
    deterministic code only.  Returns True when all assertions pass.

    P-12 regression: a file named 'cover-letter-template.docx' whose body is a
    contract must NOT be typed 'letter' by the heuristic or fast-classify paths.
    Similarly 'my-invoice-tracker.docx' that contains a report body must NOT be
    typed 'invoice'.

    Run:
        ./.venv/bin/python -m find_and_seek.eval.classify_eval --test-bias
    """
    from find_and_seek.ingest.chunk import Chunk
    from find_and_seek.ingest.summarise import _heuristic_summary, fast_classify

    def make_chunk(text: str) -> Chunk:
        return Chunk(text=text, source_type="text", location_ref="p1",
                     chunk_index=0, token_estimate=len(text.split()))

    failures: list[str] = []

    # ── 1. Heuristic: filename says "letter", content says "contract" ────────
    contract_body = (
        "THIS AGREEMENT is entered into as of 1 January 2024 between Party A and "
        "Party B. The parties hereby agree to the following terms and conditions of "
        "this binding contract. Signed and executed by both parties."
    )
    h = _heuristic_summary([make_chunk(contract_body)], "cover-letter-template.docx")
    if h["document_type"] == "letter":
        failures.append(
            "FAIL heuristic: 'cover-letter-template.docx' with contract body → "
            f"got 'letter', want 'contract' (content must dominate)"
        )
    else:
        print(f"  PASS heuristic P-12a: got '{h['document_type']}' (not 'letter')")

    # ── 2. Heuristic: filename says "invoice", content is a report ───────────
    report_body = (
        "QUARTERLY PERFORMANCE REPORT — Q2 2024\n"
        "Executive Summary: Sales across all regions increased 12% year-on-year. "
        "This report covers performance metrics, risks, and recommendations."
    )
    h2 = _heuristic_summary([make_chunk(report_body)], "my-invoice-tracker.docx")
    if h2["document_type"] == "invoice":
        failures.append(
            "FAIL heuristic: 'my-invoice-tracker.docx' with report body → "
            f"got 'invoice', want 'report' (content must dominate)"
        )
    else:
        print(f"  PASS heuristic P-12b: got '{h2['document_type']}' (not 'invoice')")

    # ── 3. fast_classify: filename says "letter", title says "CONTRACT" ──────
    contract_title_body = "CONTRACT OF EMPLOYMENT\n\nThis agreement is between Employer Co and Employee."
    fc = fast_classify("docx", [make_chunk(contract_title_body)], "cover-letter-template.docx")
    if fc is not None and fc["document_type"] == "letter":
        failures.append(
            "FAIL fast_classify: 'cover-letter-template.docx' titled 'CONTRACT' → "
            f"got 'letter', want 'contract' or None"
        )
    else:
        pred = fc["document_type"] if fc else "None (deferred to LLM)"
        print(f"  PASS fast_classify P-12c: got '{pred}' (not 'letter')")

    # ── 4. fast_classify: "invoice" in filename, no title hit → should fire ──
    # "invoice" is in _FNAME_ONLY_SIGNALS — this is a case we WANT to keep.
    plain_body = "Summary of work completed for Client X in November 2024."
    fc2 = fast_classify("pdf", [make_chunk(plain_body)], "invoice_2024_client_x.pdf")
    if fc2 is None:
        print("  PASS fast_classify P-12d: 'invoice' filename with no title → deferred to LLM (acceptable)")
    elif fc2["document_type"] == "invoice":
        print(f"  PASS fast_classify P-12d: 'invoice' filename → correctly kept as 'invoice'")
    else:
        # Not a hard failure — just note it.
        print(f"  INFO fast_classify P-12d: 'invoice' filename → '{fc2['document_type']}' (unexpected but not a bias failure)")

    # ── 5. fast_classify: "agreement" in filename, no title → must NOT fire ──
    # "agreement" is excluded from _FNAME_ONLY_SIGNALS; without a title hit, it
    # should return None so the LLM decides.
    generic_body = "This document outlines the scope of work for the project."
    fc3 = fast_classify("docx", [make_chunk(generic_body)], "partnership-agreement-template.docx")
    if fc3 is not None and fc3["document_type"] == "contract":
        failures.append(
            "FAIL fast_classify P-12e: 'partnership-agreement-template.docx' with generic body → "
            f"got 'contract' from filename alone; 'agreement' must not be a filename-only trigger"
        )
    else:
        pred = fc3["document_type"] if fc3 else "None (deferred to LLM)"
        print(f"  PASS fast_classify P-12e: got '{pred}' (not 'contract' from filename alone)")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" ", f)
        return False
    print("\nAll filename-bias checks passed.")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="~/.findandseek/index.db")
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--model", default=None, help="HF repo to test as the classifier")
    ap.add_argument("--test-bias", action="store_true",
                    help="Run the filename-bias smoke test (no DB or LLM required)")
    a = ap.parse_args()
    if a.test_bias:
        ok = test_filename_bias()
        raise SystemExit(0 if ok else 1)
    run(a.db, a.gold, a.model)


if __name__ == "__main__":
    main()
