# Security Policy

FindandSeek's core promise is that indexed file data never leaves your
machine. If you find a way to make the engine violate that — any network
egress of file content, metadata, or derived data — or any other
vulnerability, please report it privately.

**Email:** support@findandseek.app (subject line starting with `SECURITY`)

Please do not open public issues for vulnerabilities. You'll get an
acknowledgement within a few days, and credit in the release notes once a fix
ships, if you'd like it.

## What the engine assumes about its environment

Two design decisions are worth stating explicitly, because both are safe by
default and unsafe if you change them.

**The HTTP API has no authentication.** It binds to `127.0.0.1:8775` and trusts
every caller, on the assumption that reaching it already means having a local
account. That is a reasonable trade for a single-user local tool — but it means
`FINDANDSEEK_API_HOST=0.0.0.0` exposes unauthenticated full-text search, summaries,
and typed facts over **every document you have indexed** to anyone who can reach
the port. There is no credential to get wrong because there is no credential.
Don't bind it to a non-loopback interface, and don't port-forward it. If you
need remote access, put it behind something that authenticates.

The MCP server (`127.0.0.1`, stdio or port) follows the same model.

**MCP tools are read-only by design, and that is load-bearing.** An assistant
connected to the index can search, summarise, and read your files. It cannot
move, rename, or delete them: there is deliberately no MCP tool that writes to
the filesystem. `propose_organize_plan` records a *proposal* in the index and
returns an id; applying it is user-gated in the FindandSeek app, journaled, and
reversible. If you extend the tool surface, preserving that boundary is the
single most important invariant in this codebase.

## Out of scope

- Anything requiring an attacker to already have local code execution as your
  user. At that point they can read the indexed files directly.
- The `tools/ocr/ocrtool` binary is committed pre-built. If you'd rather not run
  a binary you didn't compile, build it from the source beside it and set
  `FINDANDSEEK_OCR_BIN` — see [`tools/ocr/README.md`](tools/ocr/README.md).
- Model weights are downloaded from Hugging Face and Ollama at setup time and
  are trusted as those registries serve them.
