-- Internal Linking Assistant — Postgres schema (TRD §3).
-- Rule: every query filters by client_id. No cross-client results, ever.
-- Idempotent: safe to re-apply (IF NOT EXISTS everywhere).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS clients (
  id          SERIAL PRIMARY KEY,
  name        TEXT UNIQUE NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pages (
  id           SERIAL PRIMARY KEY,
  client_id    INT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
  url          TEXT NOT NULL,
  title        TEXT,
  char_count   INT,
  ingested_at  TIMESTAMPTZ DEFAULT now(),
  UNIQUE (client_id, url)
);

CREATE TABLE IF NOT EXISTS chunks (
  id           SERIAL PRIMARY KEY,
  client_id    INT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,  -- denormalized for fast tenant filter
  page_id      INT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  chunk_index  INT NOT NULL,
  content      TEXT NOT NULL,
  embedding    vector(768),
  tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  UNIQUE (page_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx       ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_client_idx    ON chunks (client_id);
