# Internal Linking Assistant

Recommends **internal links for a new blog post before it goes live**. Given a
target keyword, the new post's Google Doc, and a client name, it searches the
client's existing published blog library (Postgres + pgvector) and returns ranked
link suggestions — each with the blog to link to, verbatim anchor text, and the
paragraph where the link belongs. See [`PRD.md`](PRD.md) and [`TRD.md`](TRD.md).

Stack: **Python 3.11+**, **Postgres 16 + pgvector**, **Gemini API** (embeddings +
Flash), **Google Docs API** (read). Free-tier friendly, single client to start.

## Build status

| Milestone | Scope | Status |
|---|---|---|
| **M1** | DB schema + ingestion of the blog corpus | ✅ implemented |
| M2 | Exact-keyword matching + `suggest` CLI | ⬜ next |
| M3 | Semantic + hybrid + RRF + LLM gate | ⬜ |
| M4 / M4b | Google Docs read + auto-save the processed post | ⬜ |
| M5 / M6 | FastAPI web UI / write-back (optional) | ⬜ |

## Repository layout

```
.
├── cli.py                 # CLI entry point (M1: `ingest`)
├── schema.sql             # Postgres schema (idempotent)
├── requirements.txt       # runtime deps
├── requirements-dev.txt   # + pytest
├── .env.example           # config template (copy to .env)
├── linker/
│   ├── config.py          # env-based configuration
│   ├── chunking.py        # split blog content into chunks (TRD §5)
│   ├── embeddings.py      # Gemini embedding client (batch + backoff)
│   ├── db.py              # Postgres + pgvector access layer
│   └── ingest.py          # ingestion pipeline (TRD §4)
└── tests/
    └── test_chunking.py   # chunking unit tests (no DB/API needed)
```

## Prerequisites

- **Python 3.11+**
- **Postgres 16** with the **pgvector** extension
- A **Gemini API key** — https://aistudio.google.com/apikey

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
# then edit .env: set GEMINI_API_KEY and DATABASE_URL
```

Config lives entirely in `.env` (loaded via python-dotenv). Never commit `.env`.

> **Model names change.** `EMBED_MODEL` defaults to `gemini-embedding-001` (768-d)
> and `LLM_MODEL` to `gemini-flash-latest`. Verify the current models at build
> time; ingestion runs a one-off embedding smoke test that fails fast if the
> configured model/key/dimension is wrong.

## Run: ingest the blog corpus (M1)

```bash
python cli.py ingest --client gokwik --file "Gokwik content.xlsx"
```

This reads each row (`link`, `title`, `content`), chunks the content, embeds the
chunks with Gemini (`task_type=RETRIEVAL_DOCUMENT`), and stores pages + chunks.
Re-running is idempotent (each page's chunks are delete-then-inserted).

Expected summary:

```
Pages ingested  : 167
Chunks created  : <several per page>
Skipped rows    : 0
```

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

## Tests

```bash
pytest                    # runs tests/ (chunking tests need no DB/API)
```

More tests (RRF fusion, anchor validation, threshold, client isolation) land with
their milestones (M2/M3), per TRD §12.

## Security

- Secrets are read from `.env` only; `.env`, `service-account.json`, and
  `*.credentials.json` are git-ignored. Never commit credentials.
- Every database query filters by `client_id` — no cross-client leakage.
