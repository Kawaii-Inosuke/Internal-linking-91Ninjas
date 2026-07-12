# Internal Linking Assistant

Recommends **internal links for a new blog post before it goes live**. Given a
target keyword, the new post's Google Doc, and a client name, it searches the
client's existing published blog library (Postgres + pgvector) and returns ranked
link suggestions — each with the blog to link to, verbatim anchor text, and the
paragraph where the link belongs. See [`PRD.md`](PRD.md) and [`TRD.md`](TRD.md).

Stack: **Python 3.11+**, **Postgres 16 + pgvector**, **local embeddings**
(`BAAI/bge-base-en-v1.5`, 768-d, via sentence-transformers), **Gemini API**
(Flash — LLM relevance gate, M3+), **Google Docs API** (read). Free-tier
friendly, single client to start.

## Build status

| Milestone | Scope | Status |
|---|---|---|
| **M1** | DB schema + ingestion of the blog corpus | ✅ implemented |
| **M2** | Exact-keyword matching (`suggest`) + thin web UI | ✅ implemented |
| M3 | Semantic + hybrid + RRF + LLM gate | ⬜ next |
| M4 / M4b | Google Docs read + auto-save the processed post | ⬜ |
| M6 | Write links back into the Doc (optional) | ⬜ |

> The FastAPI UI (TRD's Phase-2 stack) is pulled forward into M2 as a thin front
> door over the same matcher; M3 deepens the matching behind it.

## Repository layout

```
.
├── cli.py                 # CLI entry point (`ingest`, `suggest`, `serve`)
├── app.py                 # FastAPI web UI (thin wrapper over the matcher)
├── schema.sql             # Postgres schema (idempotent)
├── requirements.txt       # runtime deps
├── requirements-dev.txt   # + pytest
├── .env.example           # config template (copy to .env)
├── templates/
│   └── index.html         # single-page suggest form + results table
├── linker/
│   ├── config.py          # env-based configuration
│   ├── chunking.py        # split blog content into chunks (TRD §5)
│   ├── embeddings.py      # local embedder (sentence-transformers, bge)
│   ├── db.py              # Postgres + pgvector access layer
│   ├── matcher.py         # matching engine — exact-keyword pass (TRD §6 A)
│   └── ingest.py          # ingestion pipeline (TRD §4)
└── tests/
    ├── test_chunking.py   # chunking unit tests (no DB/model needed)
    ├── test_config.py     # config unit tests
    ├── test_embeddings.py # embedder task-split tests (fake model)
    ├── test_ingest.py     # ingest planning / blank-row filter tests
    └── test_matcher.py    # exact-pass ranking + first-occurrence anchor tests
```

## Prerequisites

- **Python 3.11+**
- **Postgres 16** with the **pgvector** extension
- A **Gemini API key** — https://aistudio.google.com/apikey (needed only from M3
  onwards, for the LLM relevance gate; ingestion embeds locally and needs no key)

### Install Postgres 16 + pgvector

**Option A — local apt (Debian/Ubuntu):**

```bash
sudo apt-get update
sudo apt-get install -y postgresql-16 postgresql-16-pgvector
sudo -u postgres psql -c "CREATE ROLE linker LOGIN PASSWORD 'linker';"
sudo -u postgres psql -c "CREATE DATABASE linker OWNER linker;"
# -> DATABASE_URL=postgresql://linker:linker@localhost:5432/linker
```

**Option B — Docker:**

```bash
docker run -d --name linker-pg -p 5432:5432 \
  -e POSTGRES_USER=linker -e POSTGRES_PASSWORD=linker -e POSTGRES_DB=linker \
  pgvector/pgvector:pg16
# -> DATABASE_URL=postgresql://linker:linker@localhost:5432/linker
```

**Option C — hosted free tier** (Supabase / Neon): create a database, enable the
`vector` extension, and copy its connection string into `DATABASE_URL`.

The `vector` extension and all tables/indexes are created automatically on first
run (`schema.sql` is applied idempotently), so no manual `psql` step is required
beyond creating the database.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env
# then edit .env: set DATABASE_URL (GEMINI_API_KEY is only needed from M3)
```

Config lives entirely in `.env` (loaded via python-dotenv). Never commit `.env`.

> **First run downloads the embedding model.** `EMBED_MODEL` defaults to the local
> `BAAI/bge-base-en-v1.5` (768-d). The first ingest downloads it (~hundreds of MB)
> and caches it under `~/.cache/huggingface`; this is a **one-time** cost — every
> later run loads it from disk and embeds fully offline, with no API key or rate
> limit. Ingestion runs a one-off embedding smoke test that fails fast if the
> configured model/dimension is wrong.
>
> **Model names change.** `LLM_MODEL` (M3+) defaults to `gemini-flash-latest`;
> verify the current free-tier Flash model at build time.

## Run: ingest the blog corpus (M1)

```bash
python cli.py ingest --client gokwik --file "Gokwik content.xlsx"
```

This reads each row (`link`, `title`, `content`), **drops content-less rows**
(the corpus has ~835 blank trailing rows), chunks the content, embeds the chunks
locally with `bge-base-en-v1.5` (document task, no query prefix), and stores
pages + chunks. Re-running is idempotent (each page's chunks are
delete-then-inserted).

Expected summary (~2,312 chunks across the 167 content-bearing posts):

```
Pages ingested  : 167
Chunks created  : ~2312
Skipped rows    : 0
Blank rows      : 835 (empty spreadsheet rows, ignored)
```

> **Switched embedding models?** Vectors from different models are not comparable.
> If you previously ingested with another model, wipe first so the two never mix:
> `TRUNCATE chunks;` (or drop/recreate the table) and re-run the ingest.

### Verify the ingest (M1 acceptance — TRD §11)

```sql
-- 167 pages for gokwik
SELECT count(*) FROM pages p JOIN clients c ON c.id = p.client_id
WHERE c.name = 'gokwik';

-- every page has at least one chunk
SELECT count(*) FROM pages p
WHERE NOT EXISTS (SELECT 1 FROM chunks ch WHERE ch.page_id = p.id);  -- expect 0

-- no null embeddings
SELECT count(*) FROM chunks WHERE embedding IS NULL;                 -- expect 0
```

## Suggest internal links (M2 — exact-keyword pass)

Given a keyword, the new post's text, and its final URL, M2 runs the exact-keyword
pass (TRD §6 Step A): it finds the best existing post to link the keyword to
(keyword-in-title first, then chunk match count; the post is never linked to
itself) and the first paragraph of the new post where the keyword appears. The
matching logic lives in `linker/matcher.py`; the CLI and the web UI are two thin
front doors over the **same** function.

The new post is supplied as pasted text or a `.txt`/`.md` file (the TRD §8
fallback — the Google Docs reader arrives in M4). M2 is **read-only**: it does not
save the post to the DB (that auto-save is M4b).

### CLI

```bash
python cli.py suggest --client gokwik --keyword "cart abandonment" \
                      --file new_post.txt --url https://gokwik.co/blog/new-post
# or pass text inline instead of --file:
python cli.py suggest --client gokwik --keyword "cart abandonment" \
                      --text "…post body…" --url https://gokwik.co/blog/new-post
```

Prints a table and writes `suggestions.json`. `--url` is the post's final live URL
(excluded from its own target selection). Example:

```
PARA  CONF  ANCHOR                    TARGET
   1  1.00  cart abandonment          https://www.gokwik.co/blog/how-to-avoid-and-overcome-cart-abandonment-losses
```

### Web UI

```bash
python cli.py serve            # -> http://127.0.0.1:8000  (or: uvicorn app:app)
```

Open http://localhost:8000, paste the post, enter the keyword and URL, and submit.
The page renders the same suggestions as a table (target URL, title, anchor,
paragraph index, confidence). `GET /health` returns `{"status":"ok"}`.

## Tests

```bash
pytest                    # runs tests/ — all pure-logic, no DB/model/API needed
```

Current suites: chunking boundaries, config parsing, the embedder document/query
task split (fake model), the ingest blank-row filter, and the M2 matcher
(exact-pass ranking + first-occurrence anchor, with `suggest` orchestration over a
fake DB layer). More tests (RRF fusion, anchor validation, threshold) land with
M3, per TRD §12.

## Security

- Secrets are read from `.env` only; `.env`, `service-account.json`, and
  `*.credentials.json` are git-ignored. Never commit credentials.
- Every database query filters by `client_id` — no cross-client leakage.
