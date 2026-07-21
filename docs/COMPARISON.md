# How this differs from RAG, knowledge graphs, and agentic search

Short version: **classic RAG retrieves passages at query time. This retrieves
*understanding* that was computed once at ingest time.** It is not a replacement
for vector search — it *uses* hybrid vector + keyword + cross-encoder reranking
under the hood. What it adds is the layer most retrieval stacks skip: a verified,
compact card for every document, produced once when the file is ingested, and an
escalation protocol so an agent works from cards and only pulls raw text when a
card genuinely isn't enough.

If you only remember one thing: **an agent shouldn't re-read your files on every
question.** That is the cost this design removes.

---

## The core idea: triage cards, not raw chunks

When a file is ingested, an on-device model reads it **once** and writes a
compact **card** to a local SQLite row. The card holds:

- `document_type` — constrained to a fixed enum (invoice, contract, report,
  letter, receipt, cv, …), decided by a dedicated classifier, not the filename
- `one_line_anchor` — the single most decision-useful line (≤25 words), never
  letterhead or boilerplate
- `key_facts` — the stated facts, typed and normalised (money → number +
  currency, dates → ISO 8601)
- `headline_fact` — the largest amount, or the most confident date
- `confidence_note` — **mechanically rebuilt**, not self-reported: how many facts
  were verified verbatim against the source, and how much of the document was
  actually read

At query time, `search_files` returns these cards — filename, type, anchor,
headline fact, a confidence band, and `estimated_tokens_if_opened` — **not the
raw text.** A card is usually enough to answer or to decide which file to open.
Only when it isn't does the agent escalate:

```
search_files      → ranked triage cards          (cheap, usually sufficient)
get_file_context  → card + top 3 chunk previews   (mid-tier, on demand)
get_chunk         → one chunk's full text         (deepest, only when forced)
```

This escalation is baked into the MCP server's own instructions, so any client
(Claude Desktop, etc.) is told to work from cards first. The result: agent
context stays small and roughly *constant* per question, instead of growing with
the size of the job.

---

## vs. classic RAG (chunk → embed → top-k → stuff into context)

We are not against vector retrieval — the search pipeline is hybrid **vector
(0.7) + BM25 keyword (0.3)**, fused with Reciprocal Rank Fusion, then reranked by
a bundled ONNX **cross-encoder**, with a relevance floor that returns *nothing*
rather than pad with noise. That part is solid, conventional IR.

The difference is what surrounds it:

| | Classic RAG | This engine |
|---|---|---|
| **When the document is "understood"** | Never, really — it's chunked and embedded. Understanding happens at query time, inside the LLM, from raw text. | Once, at ingest, into a stored card. |
| **What a query returns** | Raw top-k chunks, stuffed into the prompt. | Compact cards; raw text only on explicit escalation. |
| **Context cost per question** | Grows with k and with the size of the job — every question re-ships passages. | Roughly constant — cards are small and precomputed. |
| **Retrieval signals** | Usually vector similarity alone. | Hybrid vector + BM25 + cross-encoder rerank + filename/summary boosts. |
| **"I don't know"** | Rare — returns the top-k regardless, so it always hands back *something*. | A relevance gate returns an empty result when nothing clears the bar. |
| **Fact grounding** | The LLM reads chunks live and may paraphrase or hallucinate. | Facts are extracted and **containment-verified** against the source at ingest; unverified facts are flagged on the card. |

Classic RAG is simpler and perfectly good for open-ended passage Q&A over a huge
corpus. This design wins when an **agent** is doing multi-step work and its
context budget (and your bill) is the bottleneck — see
[the benchmark](benchmarks/): on broad synthesis tasks it read **~56% fewer input
tokens** than an agent using generic file tools, at about half the cost, without
thinner deliverables.

---

## vs. knowledge graphs

Knowledge graphs are the honest opposite trade-off, and it's worth being clear
about it.

| | Knowledge graph | This engine |
|---|---|---|
| **Setup** | Design an ontology/schema, do entity resolution, build the graph. | Point it at a folder. No schema. |
| **On messy real-world docs** | Brittle — extraction and entity linking are the hard, failure-prone part. | Degrades gracefully; a card that can't be verified says so. |
| **Maintenance** | The graph must be kept consistent as files change. | Incremental — a file watcher re-indexes only what changed. |
| **Multi-hop relationship queries** ("who is connected to X through Y") | **This is what graphs are for.** | **Not supported.** No entity-to-entity edges, no triples, no traversal. |

We extract **typed facts** (money, dates, quantities, people, orgs, refs) into a
normalised table you can filter and aggregate with SQL — `query_typed_facts`,
`aggregate_typed_facts`. But that is a *flat, citable fact store*, not a graph.
Entity linking is string-normalised, so "J. Smith" and "John Smith" are distinct
rows — there is no cross-document coreference.

**If your problem is relationship traversal, use a knowledge graph.** If it's
"understand and find things across a pile of documents cheaply," this is lighter
and far less to maintain.

---

## vs. GraphRAG (Microsoft GraphRAG and friends)

GraphRAG is the interesting middle ground: an LLM reads the whole corpus, extracts
entities and relationships into a graph, clusters them into "communities," and
writes community summaries — so it can answer *global, thematic* questions ("what
are the main themes across all these documents").

- **What it's for:** corpus-wide sensemaking. If you need "summarise the whole
  archive's recurring themes," GraphRAG is purpose-built and this is not.
- **Indexing cost:** GraphRAG runs an LLM over the corpus to build *and* summarise
  the graph — expensive, and usually against a cloud LLM. This also reads each
  file once with an LLM, but produces per-document cards plus a flat, citable
  facts table — cheaper, on-device, and incremental per file.
- **Query shape:** GraphRAG answers by map-reducing over graph communities; this
  answers by returning the few right cards and the decision-useful fact, with a
  provable citation.

Different jobs. GraphRAG is global synthesis via a graph; this is cheap,
verifiable per-document retrieval with no graph.

---

## vs. an agent with generic file tools (grep / read_file)

This is the baseline the [benchmark](benchmarks/) measures against — the
"point an agent at the files and let it read" approach (Claude Code, most
filesystem-tool agents).

The failure mode is context growth: the agent greps, reads whole files, re-reads
on the next question, and its input tokens scale with the size of the task. On a
2,500-file corpus over three full repeats, bounded retrieval cut input tokens by
**~56%** and cost by **~half** on synthesis work. The saving is near zero on a
single-file lookup and grows with the job — so the honest claim is "about half on
broad synthesis," not a universal number. The raw per-turn transcripts and the
caveats are all in [docs/benchmarks/](benchmarks/).

---

## vs. cloud RAG / vector-DB SaaS / enterprise search

| | Cloud RAG (Pinecone, hosted RAG, Glean-style search) | This engine |
|---|---|---|
| **Where your data goes** | Uploaded; embeddings and often the text live in someone else's cloud. | Nowhere. Models run on-device; the index is a local SQLite file. |
| **Cost model** | Ongoing per-query / per-vector / seat pricing. | Local compute you already own. |
| **Network at query time** | Required. | None. The reranker ships as a bundled ONNX model — zero runtime egress. |
| **Data residency** | Their infrastructure. | Your machine. |

If you need a cloud-scale index shared across a whole company, hosted search is
built for that. If you want your own documents understood **without them leaving
your machine**, that is this engine's entire premise.

---

## vs. local "chat with your documents" apps (AnythingLLM, Khoj, PrivateGPT, GPT4All LocalDocs, Onyx)

These are the closest neighbours — also local, also "your files + an LLM." So the
difference is architectural, not "we're private and they aren't":

- **They are closed apps with their own chat box.** This is a **headless engine**
  exposed over **MCP and REST** to whatever assistant you already use. It doesn't
  add another chatbot — it gives the agent you already have a memory tool.
- **Most do query-time RAG:** retrieve chunks, stuff them into the model on every
  question. This does **ingest-time cards plus escalation**, so an agent's context
  stays small across a long, multi-step task.
- **The trust layer** — containment-verified facts, mechanical confidence, honest
  empties, injection scrubbing — is not standard in those stacks.

They're good turnkey apps. If you want an all-in-one local chatbot over your
files, use one. If you want your existing agent stack to *have* that memory as a
callable tool, that's this.

---

## vs. cloud document assistants (NotebookLM, Claude Projects, ChatGPT file upload)

Upload files, chat grounded in them, get citations — convenient and well-made.
But your documents leave your machine and live in the provider's system; there are
per-provider limits on how much you can attach; and the memory is welded to that
one product. This keeps everything on disk, has no upload step or cap beyond your
own storage, and is available to *any* MCP client rather than a single assistant.

---

## vs. agent memory layers (mem0, Zep, Letta/MemGPT)

Easy to conflate because both are "memory for an agent," but they store different
things. Those layers remember the **conversation** — what the user said, facts
learned in chat, sometimes as a temporal graph. This remembers the **documents**
you already have on disk. They're complementary: you could run a conversational
memory layer *and* point the agent at this for document memory. (Note that Zep's
graph is a graph of interactions — there is still no document graph here.)

---

## A few more it gets mistaken for

- **Desktop / OS search (Spotlight, Everything, Recoll, DocFetcher).** Keyword,
  filename, and metadata matching — no semantic understanding, no extracted facts,
  no confidence, nothing structured to hand an agent. This does hybrid
  semantic + keyword and returns *understanding*, not just paths. (Spotlight is a
  fine complement when you already know the filename.)
- **"Just use a long context window."** With million-token models you *can* stuff
  everything in — but latency and cost scale with the tokens you ship every turn,
  which is precisely the bill this removes. Retrieval keeps each turn small; long
  context pays for the whole pile on every question.
- **"Just fine-tune a model on my docs."** Fine-tuning bakes behaviour into
  weights; it's unreliable for factual recall, can't cite a source, and has to be
  redone when your files change. This keeps facts in a queryable, citable index
  that updates incrementally as files change.

---

## The trust layer (the part most retrieval stacks don't have)

Because understanding happens once, up front, it can afford checks that are too
expensive to run on every query:

- **Containment verification** — each precise fact on a card is checked to appear
  verbatim in the source. The card reports `verified N/M facts`, and lists the
  ones it couldn't confirm.
- **Mechanical confidence** — the model's *self-reported* confidence is
  discarded. The confidence band is rebuilt from provable signals: how many facts
  verified, and what fraction of the document was actually read.
- **Injection scrubbing** — prompt-injection text embedded in a document
  ("ignore all previous instructions…") is stripped so a card never re-serves it
  to your agent from a trusted surface.
- **Honest empties** — the relevance gate returns `[]` when nothing is a
  confident match, so "couldn't find it" is a real answer instead of a plausible
  wrong one.

---

## Where this is *not* the right tool

Stated plainly, because credibility depends on it:

- **You need multi-hop relationship queries.** Use a knowledge graph — there is
  no graph traversal here.
- **You need cross-document entity resolution.** Entity linking is string-match;
  aliases and coreference are not resolved.
- **Your documents are very long and you need every detail.** Summary cards read
  a bounded window of each file (head/middle/tail sampling); a card can miss
  content buried deep in a large document. The coverage percentage is reported on
  the card, but it is a real limit.
- **You want frontier-model answer quality regardless of cost.** Ingest uses
  small on-device models (Qwen3-4B class for summaries, a 300M embedding model).
  That is the price of local-first and zero-egress.
- **Ingest is not free.** Reading every file once with a local model is the bulk
  of ingest time; the benchmark measures query-time savings, not total cost of
  ownership. The win compounds only if you query the corpus more than once.
- **Single short lookups.** If you only ever ask one-file questions, a plain grep
  is fine — the savings here are for repeated, broad work.

---

## In one line

Vector search finds passages. Knowledge graphs model relationships. This finds
**verified, precomputed understanding** — and hands an agent small cards instead
of making it re-read your files on every question.
