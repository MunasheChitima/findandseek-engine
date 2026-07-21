# Why this is built for agents

Most retrieval tools were designed for a human typing a query into a search box.
This one is designed for an **agent** doing multi-step work, where the scarce
resource is the context window and the bill is the tokens you ship every turn.

If you build agents, here is the case in one screen. For *how* to drive it, see
the [agent guide](AGENT_GUIDE.md); for how it compares to RAG, GraphRAG,
knowledge graphs, and the rest, see [COMPARISON.md](COMPARISON.md).

---

## The problem it removes

An agent with generic file tools (`grep`, `read_file`) answers a question by
reading files — then re-reads them for the next question. Its context, latency,
and cost grow with the size of the job. On a broad task over a real document set,
most of the tokens an agent spends are re-reading things it has already seen.

## The fix

Every file is read and understood **once**, at ingest, into a compact card. At
query time the agent gets **cards, not raw text**, and only escalates to full
passages when a card isn't enough. Context per question stays small and roughly
*constant* instead of scaling with the corpus.

Measured over a 2,500-file corpus, three full repeats: **~56% fewer input
tokens** and about **half the cost** versus the read-everything agent, with no
loss in deliverable quality — the saving grows with the size of the job. Numbers,
transcripts, and caveats in [docs/benchmarks/](benchmarks/).

## What specifically helps an agent

- **Bounded, near-constant context.** Cards are small and precomputed, so a long
  task doesn't blow the window. This is the whole cost thesis.
- **A real escalation protocol.** `search_files → get_file_context → get_chunk`.
  The agent reads cards first and raw text only on demand — and the MCP server's
  own initialize instructions tell the agent to do exactly that.
- **A token budget on every card.** `estimated_tokens_if_opened` lets the agent
  decide whether opening a file is worth the context *before* it spends it.
- **Verified facts, so it hallucinates less.** Facts are extracted and
  containment-checked against the source at ingest; each card reports how many
  verified. The agent quotes provable facts with citations instead of
  paraphrasing from a chunk.
- **Honest empties.** When nothing clears the confidence bar, the engine returns
  an empty result instead of the top-k-regardless. "Not found" is a real answer,
  so the agent stops fabricating.
- **Structured queries, not eyeballing.** `query_typed_facts` /
  `aggregate_typed_facts` answer "total invoiced in Q1" directly from a typed
  fact table — no reading chunks and adding numbers by hand.
- **MCP-native.** It's a set of tools, not another chatbot. It drops into any
  agent stack (Claude Desktop, or any MCP client) as callable memory.
- **Local and private.** Models run on-device; the index is a local SQLite file;
  nothing leaves the machine. No per-query cloud cost, no data residency
  question.

## What it deliberately doesn't do

No graph traversal, no cross-document entity resolution, and long documents are
read through a bounded window (the card reports coverage). If your agent needs
multi-hop relationship reasoning, pair it with a knowledge graph. The honest
limits are listed in [COMPARISON.md](COMPARISON.md#where-this-is-not-the-right-tool).

---

**Next:** [AGENT_GUIDE.md](AGENT_GUIDE.md) — the operating manual, including the
anti-patterns to keep out of your agent's system prompt.
