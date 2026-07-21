#!/usr/bin/env python3
"""Generation-savings benchmark for the FindandSeek engine.

Drives a real Claude agent over the company-2500 corpus in two configurations
answering identical prompts: an "incumbent" side with generic filesystem tools
(list_dir / grep / read_file), and a "system" side using the engine's own MCP
retrieval surface (search_files / get_file_context). Token usage, tool calls,
and no-cache cost are recorded per side. See ../TOKEN_ECONOMICS_REPORT.md.

Run from the repo root:  uv run --with anthropic python generation_savings_benchmark.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRATCH = Path(__file__).resolve().parent
DB_PATH = SCRATCH / "company2500_index.db"
CORPUS_ROOT = Path(os.environ["GENSAV_CORPUS_ROOT"])
REPEATS = int(os.environ.get("GENSAV_REPEATS", "3"))
# Spend guard per side, per task. Same warning as the iteration ceilings below:
# it truncates mid-answer, so set it high enough to be a runaway stop and not a
# limit the run actually reaches. The old 0.75 default sat right on top of the
# full-program audit's real incumbent cost (~$0.72), so the task was being cut
# off at the exact moment it mattered most.
MAX_COST_PER_SIDE = float(os.environ.get("GENSAV_MAX_COST", "3.0"))
MODEL = os.environ.get("GENSAV_MODEL", "claude-sonnet-5")
# Iteration ceilings, per side. These are a runaway guard, NOT a handicap: both
# sides get the same generous budget so the comparison is decided by how much
# each has to read, not by who ran out of turns.
#
# They used to default to 15/8, and that asymmetry silently invalidated results.
# The full-program audit spans six departments; the engine cannot finish it in
# eight turns, so it hit the ceiling and returned an *empty* answer — which the
# accounting then scored as a large cost "saving" because failing is cheap. Any
# run where a side finishes on "cap" rather than "end_turn" is void; check the
# `finish` field before quoting a number from it.
INC_MAX_ITERS = int(os.environ.get("GENSAV_INC_ITERS", "40"))
SYS_MAX_ITERS = int(os.environ.get("GENSAV_SYS_ITERS", "40"))
TAG = os.environ.get("GENSAV_TAG", "stripped")
# Comma-sep task ids to include (default all); comma-sep sides to run (default both)
ONLY_TASKS = [t for t in os.environ.get("GENSAV_TASKS", "").split(",") if t]
ONLY_SIDES = [s for s in os.environ.get("GENSAV_SIDES", "incumbent,system").split(",") if s]

# Engine env — must be set before importing find_and_seek
os.environ["FINDANDSEEK_DB_PATH"] = str(DB_PATH)
os.environ.setdefault("FINDANDSEEK_EMBED_BACKEND", "ollama")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("FINDANDSEEK_OLLAMA_EMBED_MODEL", "nomic-embed-text")
os.environ.setdefault("FINDANDSEEK_SEARCH_RELEVANCE_FLOOR", "0.0")
os.environ.setdefault("FINDANDSEEK_SEARCH_LEXICAL_MIN_REL", "0.0")
os.environ.setdefault("FINDANDSEEK_SKIP_FTS_CHECK", "1")

import anthropic  # noqa: E402
from find_and_seek.mcp import server  # noqa: E402

PRICE = {"claude-sonnet-5": {"in": 2.0, "out": 10.0, "cache_read": 0.2, "cache_write": 2.5}}
READ_CAP = 12000
GREP_MAX = 12
TEXT_EXTS = {".txt", ".csv", ".md", ".eml", ".json", ".yaml", ".yml", ".log"}

SYSTEM_PROMPT = (
    "You are an enterprise assistant. Answer the user's question using ONLY the tools provided "
    "to locate information in the company's data. Work efficiently: use as few tool calls as you "
    "can. When you have the answer, state it directly and concisely. If the information genuinely "
    "is not available through your tools, say so plainly rather than guessing."
)

GEN_TASKS = [
    {"id": "spring_glow_exec_summary", "prompt": (
        "Draft a one-page executive summary of the Spring Glow creative campaign. "
        "Include: the total spend across all Spring Glow creative work, the creative "
        "concepts that were approved, and which agency or team delivered them. "
        "Ground every figure and claim in the company's documents and note the source.")},
    {"id": "cross_campus_reconciliation", "prompt": (
        "Produce a board-ready reconciliation memo of total Spring Glow campaign spend. "
        "Search across Campaign, Finance, and Legal records — do not rely on Campaign's "
        "documents alone. Resolve every conflicting spend figure you find: state which "
        "document is authoritative and why. Flag any spend or commitment visible in "
        "Finance or Legal records that Campaign's own budget documents do not account "
        "for. Note anything not yet signed off or approved by Legal. Cite every figure "
        "and claim to its source document.")},
    {"id": "full_program_audit", "prompt": (
        "Produce a comprehensive spend-and-risk audit memo for the Spring Glow (SGX) "
        "program, covering every department that touched it: Campaign, Finance, Legal, "
        "HR, Operations, and IT. (1) State total program cost across ALL departments — "
        "not just creative spend — including staffing/contractor time, IT infrastructure, "
        "logistics, and legal/compliance costs. Reconcile every conflicting figure you "
        "find and state which source is authoritative and why. (2) Produce a risk "
        "register: every invoice, contract, or commitment that is unsigned, unpaid, "
        "pending approval, or flagged non-compliant, with department and dollar amount. "
        "(3) Cross-reference vendors across departments — flag any vendor whose billed "
        "amount differs between departments' records, or who appears in one department's "
        "records but not in Campaign's own approved vendor list. "
        "(4) Produce a single reconciled program-total variance: the dollar gap between "
        "the highest and lowest total figures found across all six departments, and that "
        "gap as a percentage of the highest figure. Show your work — list every candidate "
        "total you found and which document it came from before collapsing to the final "
        "variance. "
        "(5) Cross-reference every person named as a sender, signer, or approver across "
        "all six departments' documents. Flag any name that appears with inconsistent "
        "details between departments (different title, different contact info, mismatched "
        "dates, or a signature that looks pasted/templated rather than department-specific) "
        "as a possible data-integrity or fraud flag. "
        "(6) Cite every figure and claim to its source document.")},
]


# ── incumbent tools: generic filesystem over the corpus ──────────────
def _all_files():
    return [p for p in CORPUS_ROOT.rglob("*") if p.is_file() and p.name != "index.db"]


def _db_text_for(path: Path) -> str:
    conn = sqlite3.connect(str(DB_PATH))
    try:
        row = conn.execute("SELECT id FROM files WHERE path = ?", (str(path),)).fetchone()
        if not row:
            row = conn.execute("SELECT id FROM files WHERE filename = ?", (path.name,)).fetchone()
        if not row:
            return ""
        r = conn.execute(
            "SELECT COALESCE(fs.summary_text,''), COALESCE(GROUP_CONCAT(fc.text, char(10)),'') "
            "FROM files f LEFT JOIN file_summaries fs ON fs.file_id=f.id "
            "LEFT JOIN file_chunks fc ON fc.file_id=f.id WHERE f.id=? GROUP BY f.id",
            (row[0],),
        ).fetchone()
        summ, chunks = (r[0], r[1]) if r else ("", "")
        return (summ + "\n" + chunks).strip()
    finally:
        conn.close()


def _read_text(path: Path) -> str:
    if path.suffix.lower() in TEXT_EXTS:
        try:
            return path.read_bytes().decode("utf-8", errors="replace")[:READ_CAP]
        except OSError:
            return ""
    return _db_text_for(path)[:READ_CAP]


def inc_list_dir(path: str = ".") -> str:
    base = (CORPUS_ROOT / path).resolve()
    if not str(base).startswith(str(CORPUS_ROOT.resolve())) or not base.exists():
        return "No such directory."
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in base.iterdir() if p.name != "index.db")
    return "\n".join(entries[:60]) or "(empty)"


def inc_grep(query: str) -> str:
    q = query.lower().strip()
    hits = []
    for p in _all_files():
        text = _read_text(p)
        idx = text.lower().find(q)
        if idx != -1:
            rel = str(p.relative_to(CORPUS_ROOT))
            snippet = text[max(0, idx - 40):idx + 80].replace("\n", " ")
            hits.append(f"{rel}: ...{snippet}...")
        if len(hits) >= GREP_MAX:
            break
    return "\n".join(hits) if hits else f"No files contain '{query}'."


def inc_read_file(path: str) -> str:
    p = (CORPUS_ROOT / path).resolve()
    if not str(p).startswith(str(CORPUS_ROOT.resolve())) or not p.is_file():
        return "No such file."
    return _read_text(p) or "(file has no extractable text)"


TOOLS_INCUMBENT = [
    {"name": "list_dir", "description": "List files and folders in the company filesystem under a directory (relative path; '.' = top level).",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}},
    {"name": "grep", "description": "Search the full text of every file for a keyword or phrase. Returns matching file paths with a short snippet.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "read_file", "description": "Read the full text content of a file by its path.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
]


# ── system tools: the stripped engine's own surface ──────────────────
def _fn(tool):
    return tool.fn if hasattr(tool, "fn") else tool


def sys_search_files(query: str) -> str:
    res = _fn(server.search_files)(query, limit=5)
    cards = []
    for h in res.get("results", []):
        cards.append({
            "file_id": h.get("file_id"),
            "filename": h.get("filename"),
            "document_type": h.get("document_type"),
            "one_line_anchor": h.get("one_line_anchor"),
            "summary": (h.get("summary") or "")[:400],
            "match_confidence": h.get("match_confidence"),
            "headline_fact": h.get("headline_fact"),
        })
    return json.dumps(cards, ensure_ascii=False) if cards else "No results."


def sys_get_evidence(file_id: int) -> str:
    try:
        res = _fn(server.get_file_context)(file_id=int(file_id), max_chunks=3)
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"
    parts = []
    s = res.get("summary") or {}
    if s.get("summary_text"):
        parts.append(f"SUMMARY: {s['summary_text']}")
    if s.get("key_facts"):
        parts.append(f"KEY FACTS: {json.dumps(s['key_facts'], ensure_ascii=False)}")
    for c in res.get("chunks", []):
        if isinstance(c, dict):
            loc = c.get("location_ref")
            body = c.get("text") or c.get("preview") or json.dumps(c, ensure_ascii=False)
            parts.append(f"[{loc}] {body}" if loc else str(body))
        else:
            parts.append(str(c))
    return "\n".join(parts)[:READ_CAP] or json.dumps(res, ensure_ascii=False)[:READ_CAP]


TOOLS_SYSTEM = [
    {"name": "search_files", "description": "Semantic + keyword hybrid search across the company knowledge base. Returns ranked result cards (filename, summary, one-line anchor, match_confidence, file_id) for the most relevant documents.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "get_evidence", "description": "Fetch the detailed context for a specific result by its file_id when the summary card is not enough: the file's stored summary, key facts, and first content chunks.",
     "input_schema": {"type": "object", "properties": {"file_id": {"type": "integer"}}, "required": ["file_id"]}},
]


# ── agent loop (tool-use loop with prompt-cache breakpoints) ─────────
def _block_to_dict(b):
    return b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else dict(b)


def cost_of(usage, model):
    p = PRICE[model]
    return round((usage["in"] * p["in"] + usage["out"] * p["out"]
                  + usage["cache_read"] * p["cache_read"] + usage["cache_write"] * p["cache_write"]) / 1e6, 4)


def run_agent(client, model, tools, dispatch, question, max_iters, transcript_path=None, max_cost=None):
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    tools = [dict(t) for t in tools]
    tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
    system_blocks = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    usage = {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0}
    tool_calls = 0
    t0 = time.perf_counter()
    final_text = ""
    xf = open(transcript_path, "w") if transcript_path else None
    cache_marked_at = 0
    finish = "cap"  # overwritten to "end_turn" or "budget" below
    for i in range(max_iters):
        messages[cache_marked_at]["content"][-1].pop("cache_control", None)
        messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        cache_marked_at = len(messages) - 1
        with client.messages.stream(
            model=model, max_tokens=32000, system=system_blocks, tools=tools, messages=messages,
        ) as stream:
            resp = stream.get_final_message()
        u = resp.usage
        usage["in"] += u.input_tokens
        usage["out"] += u.output_tokens
        usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
        final_text = " ".join(b.text for b in resp.content if b.type == "text") or final_text
        if xf:
            xf.write(json.dumps({"turn": i + 1, "stop_reason": resp.stop_reason,
                                 "usage": {"in": u.input_tokens, "out": u.output_tokens,
                                           "cache_read": u.cache_read_input_tokens or 0,
                                           "cache_write": u.cache_creation_input_tokens or 0},
                                 "blocks": [_block_to_dict(b) for b in resp.content]}) + "\n")
            xf.flush()
        if max_cost is not None and cost_of(usage, model) >= max_cost:
            finish = "budget"
            break
        if resp.stop_reason != "tool_use":
            finish = "end_turn"
            break
        messages.append({"role": "assistant", "content": [_block_to_dict(b) for b in resp.content]})
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                tool_calls += 1
                try:
                    out = dispatch(b.name, b.input)
                except Exception as e:  # noqa: BLE001
                    out = f"tool error: {e}"
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": str(out)[:READ_CAP]})
        messages.append({"role": "user", "content": results})
    if xf:
        xf.close()
    return {"answer": final_text, "usage": usage, "tool_calls": tool_calls,
            "finish": finish, "answer_chars": len(final_text),
            "elapsed_s": round(time.perf_counter() - t0, 2)}


def main():
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        print("No ANTHROPIC_API_KEY", file=sys.stderr)
        return 1
    client = anthropic.Anthropic(api_key=key)

    inc_dispatch = lambda n, a: {"list_dir": lambda: inc_list_dir(a.get("path", ".")),
                                 "grep": lambda: inc_grep(a["query"]),
                                 "read_file": lambda: inc_read_file(a["path"])}[n]()
    sys_dispatch = lambda n, a: {"search_files": lambda: sys_search_files(a["query"]),
                                 "get_evidence": lambda: sys_get_evidence(a["file_id"])}[n]()

    all_runs = []
    for r in range(REPEATS):
        print(f"\n{'#'*80}\nRUN {r+1}/{REPEATS} (model={MODEL}, stripped engine)", flush=True)
        rows = []
        tasks = [t for t in GEN_TASKS if not ONLY_TASKS or t["id"] in ONLY_TASKS]
        for task in tasks:
            jobs = {}
            with ThreadPoolExecutor(max_workers=2) as pool:
                if "incumbent" in ONLY_SIDES:
                    jobs["incumbent"] = pool.submit(run_agent, client, MODEL, TOOLS_INCUMBENT, inc_dispatch, task["prompt"],
                                                    INC_MAX_ITERS, str(SCRATCH / f"{TAG}_{task['id']}_incumbent_run{r}.jsonl"), MAX_COST_PER_SIDE)
                if "system" in ONLY_SIDES:
                    jobs["system"] = pool.submit(run_agent, client, MODEL, TOOLS_SYSTEM, sys_dispatch, task["prompt"],
                                                 SYS_MAX_ITERS, str(SCRATCH / f"{TAG}_{task['id']}_system_run{r}.jsonl"), MAX_COST_PER_SIDE)
                res = {side: fut.result() for side, fut in jobs.items()}
            row = {"id": task["id"]}
            parts = []
            for side, x in res.items():
                x["cost"] = cost_of(x["usage"], MODEL)
                row[side] = x
                parts.append(f"{side} {x['tool_calls']}calls ${x['cost']:.3f} [{x['finish']},{x['answer_chars']}ch]")
            rows.append(row)
            print(f"  {task['id']}: " + " | ".join(parts), flush=True)

        def side(name):
            xs = [x[name] for x in rows if name in x]
            return {"calls": sum(x["tool_calls"] for x in xs),
                    "in": sum(x["usage"]["in"] for x in xs),
                    "out": sum(x["usage"]["out"] for x in xs),
                    "cache_read": sum(x["usage"]["cache_read"] for x in xs),
                    "cache_write": sum(x["usage"]["cache_write"] for x in xs),
                    "cost_usd": round(sum(x["cost"] for x in xs), 4),
                    "elapsed_s": round(sum(x["elapsed_s"] for x in xs), 1)}
        run = {"rows": rows}
        for nm in ONLY_SIDES:
            run[nm] = side(nm)
        all_runs.append(run)
        for nm in ONLY_SIDES:
            t = run[nm]
            ti = t["in"] + t["cache_read"] + t["cache_write"]
            nocache = (ti * PRICE[MODEL]["in"] + t["out"] * PRICE[MODEL]["out"]) / 1e6
            print(f"  {nm:10}: {t['calls']} calls | total_input {ti:,} out {t['out']:,} | cached ${t['cost_usd']:.4f} | NO-CACHE ${nocache:.4f} | {t['elapsed_s']}s", flush=True)

    out = SCRATCH / f"gensav_{TAG}_results.json"
    out.write_text(json.dumps({"model": MODEL, "engine": "stripped-findandseek-engine",
                               "inc_max_iters": INC_MAX_ITERS, "sys_max_iters": SYS_MAX_ITERS,
                               "max_cost_per_side": MAX_COST_PER_SIDE,
                               "repeats": REPEATS, "runs": all_runs}, indent=2))
    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
