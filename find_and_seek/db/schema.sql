-- PRAGMAS (set on every connection in connection.py, not here):
--   PRAGMA journal_mode=WAL;
--   PRAGMA foreign_keys=ON;
--   PRAGMA synchronous=NORMAL;
--   PRAGMA busy_timeout=5000;

-- ── files ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS files (
  id            INTEGER PRIMARY KEY,
  path          TEXT UNIQUE NOT NULL,
  filename      TEXT NOT NULL,
  extension     TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  size_bytes    INTEGER,
  modified_at   TEXT,
  indexed_at    TEXT,
  index_version INTEGER NOT NULL DEFAULT 1,
  file_type     TEXT,
  status        TEXT NOT NULL DEFAULT 'pending',
  page_count    INTEGER,
  language      TEXT
);
CREATE INDEX IF NOT EXISTS idx_files_hash   ON files(content_hash);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_type   ON files(file_type);

-- ── file_chunks ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS file_chunks (
  id            INTEGER PRIMARY KEY,
  file_id       INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  chunk_index   INTEGER NOT NULL,
  text          TEXT NOT NULL,
  source_type   TEXT NOT NULL,
  location_ref  TEXT,
  token_estimate INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON file_chunks(file_id);

-- ── file_summaries ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS file_summaries (
  file_id           INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
  summary_text      TEXT,
  one_line_anchor   TEXT,
  document_type     TEXT,
  key_facts         TEXT,
  suggested_filename TEXT,
  confidence_note   TEXT,
  section_anchors   TEXT,  -- JSON {page: anchor} for composite (multi-document) files; NULL otherwise
  classification_confidence TEXT,  -- 'high'|'medium'|'low'|'none' — the classifier's confidence in document_type; drives UI hedging + the agent (MCP) contract
  suggested_category TEXT  -- the model's own short name for this kind of doc (e.g. 'tender response'); seed for EMERGENT user categories, never a stored type
);

-- ── summary_edits ─────────────────────────────────────────────────
-- User corrections to a card, stored as a FIELD-LEVEL OVERLAY beside the model
-- output, never in file_summaries itself: re-summarise/model upgrades rewrite
-- the model layer and MUST NOT touch what a human fixed. content_hash records
-- what the user saw — if the file changes later, the read path reports drift
-- ("document changed since your correction") instead of silently keeping or
-- discarding the edit. field is a card field name, or 'key_facts.<key>' for a
-- single fact; value NULL means the user deleted that field/fact.
CREATE TABLE IF NOT EXISTS summary_edits (
  file_id      INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  field        TEXT NOT NULL,
  value        TEXT,
  edited_at    TEXT NOT NULL,
  content_hash TEXT,
  PRIMARY KEY (file_id, field)
);

-- ── file_entities ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS file_entities (
  id           INTEGER PRIMARY KEY,
  file_id      INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  chunk_id     INTEGER REFERENCES file_chunks(id) ON DELETE CASCADE,
  entity_type  TEXT NOT NULL,
  entity_value TEXT NOT NULL,
  entity_raw   TEXT
);
CREATE INDEX IF NOT EXISTS idx_entities_lookup ON file_entities(entity_type, entity_value);
CREATE INDEX IF NOT EXISTS idx_entities_file   ON file_entities(file_id);

-- ── ingest_queue ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ingest_queue (
  path        TEXT PRIMARY KEY,
  event_type  TEXT NOT NULL,
  queued_at   TEXT NOT NULL,
  status      TEXT NOT NULL DEFAULT 'pending',
  attempts    INTEGER NOT NULL DEFAULT 0,
  last_error  TEXT,
  -- Processing order: higher priority first (user-flagged folders), then most
  -- recently modified/opened file first, so the files the user actually uses get
  -- understood before the long tail. recency = max(mtime, atime) epoch seconds.
  priority    INTEGER NOT NULL DEFAULT 0,
  recency     REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_queue_status ON ingest_queue(status);
-- NB: the (status, priority, recency) ordering index is created in migrations.py
-- _ensure_columns AFTER the columns exist — on a live DB those columns are added
-- by ALTER after this script runs, so the index can't live here.

-- ── vectors (sqlite-vec) ──────────────────────────────────────────
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
  chunk_id INTEGER PRIMARY KEY,
  embedding FLOAT[768]
);
CREATE VIRTUAL TABLE IF NOT EXISTS summary_vectors USING vec0(
  file_id INTEGER PRIMARY KEY,
  embedding FLOAT[768]
);

-- ── keyword (FTS5) ─────────────
-- External-content FTS over file_chunks.text ONLY. The previous schema also
-- declared a `filename` column, but file_chunks has no such column — so native
-- 'rebuild'/'integrity-check' could not read it, and the delete/update triggers
-- fed FTS a mismatched filename ('') on every edit. That desynced the bm25
-- shadow tables (docsize/data) and surfaced as "database disk image is malformed"
-- on multi-term ranking, with no way to self-repair. Filename matching is handled
-- separately by the live filename-token boost in search.hybrid, so dropping the
-- column here costs no recall and makes the index correct + rebuildable.
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
  text,
  content='file_chunks',
  content_rowid='id',
  tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON file_chunks BEGIN
  INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON file_chunks BEGIN
  INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON file_chunks BEGIN
  INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES ('delete', old.id, old.text);
  INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
END;

-- ════════════════════════════════════════════════════════════════════
-- Organize feature (ORGANIZE_FEATURE_DESIGN.md §5.3)
-- All additive (CREATE TABLE IF NOT EXISTS) so appending here upgrades an
-- existing index on next connection with no migration script.
-- migration_journal is created now but only written by Phase 2 (Apply).
-- ════════════════════════════════════════════════════════════════════

-- ── tags ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tags (
  id     INTEGER PRIMARY KEY,
  name   TEXT UNIQUE NOT NULL,
  kind   TEXT,            -- type|project|party|year|status|custom
  color  TEXT,            -- maps to a Finder tag colour when synced (Phase 2)
  source TEXT             -- auto|user
);

CREATE TABLE IF NOT EXISTS file_tags (
  file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  tag_id     INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
  source     TEXT,        -- auto|user
  confidence REAL,        -- for auto tags
  PRIMARY KEY (file_id, tag_id)
);
CREATE INDEX IF NOT EXISTS idx_file_tags_tag ON file_tags(tag_id);

-- ── plans ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS plans (
  id           INTEGER PRIMARY KEY,
  created_at   TEXT,
  applied_at   TEXT,
  undone_at    TEXT,
  status       TEXT,        -- draft|previewed|applying|applied|undone|failed
  strategy     TEXT,
  scope_json   TEXT,        -- roots/filters this plan covers
  summary_json TEXT         -- counts: N moves, N renames, N tags, conflicts
);

CREATE TABLE IF NOT EXISTS plan_actions (
  id          INTEGER PRIMARY KEY,
  plan_id     INTEGER NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
  seq         INTEGER,                       -- apply order
  action_type TEXT,                          -- move|rename|create_dir|add_tag|quarantine_duplicate
  file_id     INTEGER,                       -- nullable (create_dir)
  payload_json TEXT,                         -- from/to paths, names, tag, reason, etc.
  decision    TEXT DEFAULT 'pending',        -- pending|accepted|rejected|edited
  status      TEXT DEFAULT 'staged',         -- staged|done|skipped|failed|undone
  applied_at  TEXT,
  error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_plan_actions_plan ON plan_actions(plan_id);

-- ── migration journal (the undo ledger; written by Phase 2) ───────
CREATE TABLE IF NOT EXISTS migration_journal (
  id          INTEGER PRIMARY KEY,
  plan_id     INTEGER,
  action_id   INTEGER,
  op          TEXT,          -- move|rename|create_dir|tag|quarantine
  before_path TEXT,
  after_path  TEXT,
  before_hash TEXT,          -- verify file is unchanged before undo
  ts          TEXT
);
CREATE INDEX IF NOT EXISTS idx_journal_plan ON migration_journal(plan_id);

-- ════════════════════════════════════════════════════════════════════
-- Typed facts (HANDOVER.md #1): normalize the opaque key_facts JSON + NER
-- entities into citable, queryable rows. Additive (IF NOT EXISTS) so it
-- materialises on the next connection with no migration script.
-- Populated at ingest (ingest/facts.py) and back-fillable for the existing
-- catalog (db.store.backfill_facts).
-- ════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS facts (
  id           INTEGER PRIMARY KEY,
  file_id      INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  chunk_id     INTEGER REFERENCES file_chunks(id) ON DELETE SET NULL,
  fact_type    TEXT NOT NULL,   -- money|date|quantity|person|org|location|email|phone|ref|attribute
  key          TEXT,            -- source key (key_facts key, or the entity type)
  value_text   TEXT,            -- normalized human-readable value
  value_number REAL,            -- money/quantity numeric value
  value_date   TEXT,            -- ISO 8601 date (YYYY-MM-DD)
  unit         TEXT,            -- currency code (USD/GBP/EUR/AUD) or measurement unit
  confidence   REAL,
  source       TEXT             -- key_facts|ner
);
CREATE INDEX IF NOT EXISTS idx_facts_file   ON facts(file_id);
CREATE INDEX IF NOT EXISTS idx_facts_type   ON facts(fact_type);
CREATE INDEX IF NOT EXISTS idx_facts_number ON facts(value_number);
CREATE INDEX IF NOT EXISTS idx_facts_date   ON facts(value_date);

-- ════════════════════════════════════════════════════════════════════
-- User-extensible classification taxonomy. Built-in categories live in code
-- (organize/categories.py); this table holds USER-ADDED categories so the
-- classifier (organize/classify.py via load_taxonomy) reasons against the
-- current, extended set. Adding a row is what makes the app re-engage —
-- a reclassify pass re-types documents against the new category.
-- ════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS categories (
  slug        TEXT PRIMARY KEY,
  label       TEXT NOT NULL,
  definition  TEXT NOT NULL,
  excludes    TEXT,
  builtin     INTEGER NOT NULL DEFAULT 0,
  created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
