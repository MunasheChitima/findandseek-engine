"""End-to-end MCP smoke test: spawn findandseek-mcp over stdio and exercise every
tool against a target DB, asserting result shapes and isError behaviour.

Usage:
    FINDANDSEEK_DB_PATH=/path/to/index.db python -m find_and_seek.eval.mcp_smoke
    (defaults to ~/.findandseek/index.eval.db)

Exits non-zero if any check fails. Use a COPY of the index for production tests.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

os.environ.setdefault("FINDANDSEEK_DB_PATH", os.path.expanduser("~/.findandseek/index.eval.db"))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED_TOOLS = {
    "search_files", "get_summary", "find_entity", "get_chunk", "list_recent",
    "query_typed_facts", "aggregate_typed_facts", "index_status", "get_file_context",
    "propose_organize_plan",
}

results: list[tuple[bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((ok, name))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


async def main() -> int:
    params = StdioServerParameters(command=sys.executable, args=["-m", "find_and_seek.mcp.server"], env={**os.environ})
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            t0 = time.time()
            await s.initialize()
            check("initialize", True, f"{time.time()-t0:.2f}s")

            tools = {t.name for t in (await s.list_tools()).tools}
            check("tool set", tools == EXPECTED_TOOLS, f"{len(tools)} tools")
            check("apply is NOT agent-exposed (user-gated)", "apply_organize_plan" not in tools)

            async def call(name, **kw):
                t = time.time()
                res = await s.call_tool(name, kw)
                return res, (time.time() - t) * 1000

            res, ms = await call("index_status")
            st = res.structuredContent or {}
            check("index_status", not res.isError and st.get("files_indexed", 0) > 0, f"{st.get('files_indexed')} files, {ms:.0f}ms")
            check("index_status.facts_available", st.get("facts_available") is True, f"{st.get('total_facts_indexed')} facts")
            fid = None

            res, ms = await call("search_files", query="invoice", limit=3)
            sc = res.structuredContent or {}
            hits = sc.get("results", [])
            check("search_files", not res.isError and len(hits) > 0, f"{len(hits)} hits, {ms:.0f}ms")
            if hits:
                fid = hits[0]["file_id"]
                ids = [h["file_id"] for h in hits]
                check("search_files dedup (distinct files)", len(ids) == len(set(ids)))

            res, _ = await call("search_files", query="invoice", limit=2, offset=2)
            check("search_files pagination", not res.isError and (res.structuredContent or {}).get("offset") == 2)

            res, ms = await call("get_summary", file_id=fid)
            check("get_summary by id", not res.isError, f"{ms:.0f}ms")

            res, _ = await call("get_summary", file_id=99999999)
            check("get_summary bad id -> isError", res.isError is True)

            res, ms = await call("get_file_context", file_id=fid)
            sc = res.structuredContent or {}
            check("get_file_context", not res.isError and "summary_text" in sc, f"{ms:.0f}ms")

            res, _ = await call("get_file_context", path="/no/such/path.pdf")
            check("get_file_context unindexed path -> isError", res.isError is True)

            # Any value exercises the tool — the check is that it answers
            # without erroring, not that this index contains the entity. Set
            # FINDANDSEEK_SMOKE_ORG to something in your index to also eyeball hits.
            res, ms = await call(
                "find_entity", entity_type="org",
                value=os.environ.get("FINDANDSEEK_SMOKE_ORG", "acme"),
            )
            check("find_entity", not res.isError, f"{ms:.0f}ms")

            res, ms = await call("list_recent", days=3650)
            check("list_recent", not res.isError and len((res.structuredContent or {}).get("files", [])) > 0, f"{ms:.0f}ms")

            res, ms = await call("query_typed_facts", fact_type="money", min_number=500, limit=3)
            sc = res.structuredContent or {}
            check("query_typed_facts (money>500)", not res.isError and len(sc.get("facts", [])) > 0, f"{len(sc.get('facts', []))} facts, {ms:.0f}ms")

            res, ms = await call("aggregate_typed_facts", op="count", fact_type="money")
            sc = res.structuredContent or {}
            check("aggregate_typed_facts", not res.isError and sc.get("value", 0) > 0, f"count={sc.get('value')}, {ms:.0f}ms")

            res, ms = await call("get_chunk", chunk_id=hits[0]["top_chunk_id"]) if hits else (None, 0)
            check("get_chunk + structured location", res is not None and not res.isError and (res.structuredContent or {}).get("chunk", {}).get("location") is not None)

            # Organize: propose is preview/DB-only (safe). There is no apply tool —
            # applying is user-gated (done by the human in the app), verified above.
            res, ms = await call("propose_organize_plan", scope="`<local-db-path>`")
            sc = res.structuredContent or {}
            check("propose_organize_plan (preview)", not res.isError and "plan_id" in sc, f"{sc.get('total_actions')} actions, {ms:.0f}ms")
            check("propose includes user-gated apply handoff", "to_apply" in sc)

            # ── Agent-contract regression probes (ENGINE-FINAL-PUSH-AUDIT-2026-06-11) ──
            # Assert the *contract* an agent relies on — not just liveness.

            # Probe 1: nonsense query must return empty, not noise.
            res, _ = await call("search_files", query="zorbulated quantum framjet licensing", limit=3)
            sc2 = res.structuredContent or {}
            check("agent-contract: nonsense query → []",
                  not res.isError and sc2.get("count", -1) == 0,
                  f"got {sc2.get('count')} results")

            # Probe 2+3: oblique salary query → payslip in top-3 with non-weak confidence.
            res, ms2 = await call("search_files", query="how much do I get paid", limit=3)
            sc2 = res.structuredContent or {}
            agent_hits = sc2.get("results", [])
            payslip_hit = next(
                (h for h in agent_hits
                 if "payslip" in (h.get("document_type") or "").lower()
                 or "payslip" in (h.get("filename") or "").lower()),
                None,
            )
            check("agent-contract: oblique salary → payslip top-3",
                  payslip_hit is not None, f"{ms2:.0f}ms")
            check("agent-contract: payslip match_confidence not weak",
                  payslip_hit is not None
                  and payslip_hit.get("match_confidence") in ("strong", "moderate"),
                  f"got {payslip_hit.get('match_confidence') if payslip_hit else 'no hit'}")
            check("agent-contract: match_confidence on all hits",
                  all(h.get("match_confidence") in ("strong", "moderate", "weak")
                      for h in agent_hits),
                  f"{len(agent_hits)} hits checked")

            # Probe 4: empty find_entity → honesty payload, never silent empty.
            res, _ = await call("find_entity", entity_type="org",
                                value="__definitely_absent_xyz__")
            sc2 = res.structuredContent or {}
            check("agent-contract: find_entity empty → entity_layer_note",
                  not res.isError and "entity_layer_note" in sc2
                  and sc2.get("files") == [],
                  f"files={len(sc2.get('files', []))}, "
                  f"note={'yes' if 'entity_layer_note' in sc2 else 'missing'}")

            # Probe 5: find_entity result includes path field.
            res, _ = await call("find_entity", entity_type="org", value="a")
            sc2 = res.structuredContent or {}
            first_entity_hit = (sc2.get("files") or [None])[0]
            check("agent-contract: find_entity result has path",
                  first_entity_hit is None or "path" in first_entity_hit,
                  f"keys={list(first_entity_hit.keys()) if first_entity_hit else 'no hits'}")

            # Probe 6: unindexed path → isError (already tested above, re-assert explicitly).
            res, _ = await call("get_file_context", path="/no/such/__absent__.pdf")
            check("agent-contract: unindexed path → isError", res.isError is True)

            # Probe 7: list_recent small window → empty state has newest_file_modified_at.
            res, _ = await call("list_recent", days=1)
            sc2 = res.structuredContent or {}
            if not sc2.get("files"):
                check("agent-contract: list_recent empty → newest_file_modified_at",
                      "newest_file_modified_at" in sc2,
                      f"keys: {list(sc2.keys())}")
            else:
                check("agent-contract: list_recent empty → newest_file_modified_at",
                      True, "skipped (result non-empty)")

            # Probe 8: aggregate distinct_files dedupe field present.
            res, _ = await call("aggregate_typed_facts", op="sum", fact_type="money")
            sc2 = res.structuredContent or {}
            check("agent-contract: aggregate has distinct_files",
                  "distinct_files" in sc2, f"keys: {list(sc2.keys())}")

            # Probe 9: structured location on a real chunk.
            sample_hits = agent_hits or (hits or [])
            if sample_hits:
                cid = sample_hits[0].get("top_chunk_id")
                if cid:
                    res, _ = await call("get_chunk", chunk_id=cid)
                    sc2 = res.structuredContent or {}
                    loc = sc2.get("chunk", {}).get("location")
                    check("agent-contract: chunk location is structured dict",
                          isinstance(loc, dict) and "kind" in loc, f"location={loc}")
                else:
                    check("agent-contract: chunk location is structured dict", True,
                          "skipped (no chunk_id)")
            else:
                check("agent-contract: chunk location is structured dict", True,
                      "skipped (no hits)")

    passed = sum(1 for ok, _ in results if ok)
    print(f"\n{'='*50}\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
