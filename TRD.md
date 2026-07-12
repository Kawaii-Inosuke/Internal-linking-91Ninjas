# TRD — Internal Linking Assistant

This is the implementation spec for the tool described in `PRD.md`. It is written to be handed to a coding agent (Claude Code). Follow the build order in §11.

## 1. Architecture
Four components:
1. **Ingestion** (`ingest.py`) — one-off / re-runnable: reads blog content (xlsx/CSV) → chunks → embeds via Gemini → stores in Postgres + pgvector.
2. **Matching engine** (`matcher.py`) — takes `(client, keyword, doc paragraphs)` → hybrid retrieval → LLM relevance gate + anchor extraction → ranked suggestions.
3. **Google Docs reader** (`gdocs.py`) — reads a doc into ordered paragraphs (with char offsets); optional write-back later.
4. **Interface** — CLI (`cli.py`) for v1; FastAPI + minimal HTML (`app.py`) for v2.

**Data flow:** spreadsheet → ingestion → Postgres (pages + chunks). At query time: Google Doc → paragraphs → for each paragraph, hybrid search over that client's chunks → fuse → threshold → Gemini gate/anchor → merge with exact-keyword pass → suggestions JSON + table.

## 2. Tech Stack
- **Python 3.11+**
- **Postgres 16 + pgvector** (local via Docker, or Supabase/Neon free tier)
- **Gemini API** via the `google-genai` SDK:
  - Embeddings: `gemini-embedding-001`, `output_dimensionality=768`
  - LLM (relevance gate + anchor): current Flash model (e.g. `gemini-flash-latest` / `gemini-2.5-flash` — confirm the current free-tier Flash model at build time)
- **Google Docs API** via `google-api-python-client` (read; optional write-back)
- **DB access:** `psycopg[binary]` + `pgvector` (register vector type) or SQLAlchemy
- **FastAPI + uvicorn** (Phase 2 UI)
- **python-dotenv** for config

> Note: Gemini model names and free-tier limits change. At build time, verify the current embedding model, the current free Flash model, and rate limits, rather than hardcoding assumptions.

## 3. Data Model (Postgres) — `schema.sql`
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE clients (
  id          SERIAL PRIMARY KEY,
  name        TEXT UNIQUE NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE pages (
  id           SERIAL PRIMARY KEY,
  client_id    INT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
  url          TEXT NOT NULL,
  title        TEXT,
  char_count   INT,
  ingested_at  TIMESTAMPTZ DEFAULT now(),
  UNIQUE (client_id, url)
);

CREATE TABLE chunks (
  id           SERIAL PRIMARY KEY,
  client_id    INT NOT NULL REFERENCES clients(id) ON DELETE CASCADE,  -- denormalized for fast tenant filter
  page_id      INT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  chunk_index  INT NOT NULL,
  content      TEXT NOT NULL,
  embedding    vector(768),
  tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  UNIQUE (page_id, chunk_index)
);

CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_idx       ON chunks USING gin (tsv);
CREATE INDEX chunks_client_idx    ON chunks (client_id);
```
**Rule:** every query filters by `client_id`. No cross-client results, ever.

## 4. Ingestion Pipeline (`ingest.py`)
Input: path to xlsx/CSV with columns `link, title, content`; a `client` name.

1. Upsert the client row; get `client_id`.
2. For each row:
   a. Upsert `pages` (by `client_id, url`); set `title`, `char_count`. On re-ingest, delete existing chunks for that page first (idempotent).
   b. **Chunk** `content` (see §5) → list of `(chunk_index, text)`.
3. **Embed** all chunks in batches: `gemini-embedding-001`, `output_dimensionality=768`, `task_type="RETRIEVAL_DOCUMENT"`. Send multiple texts per request (~100) to stay under the free-tier daily request cap; throttle to respect RPM; retry with backoff on 429.
4. Insert `chunks` (embedding + content; `tsv` is generated). 
5. Log counts: pages ingested, chunks created, any rows that produced 0 chunks.

## 5. Chunking (§ used by ingestion)
`content` is newline-separated blocks (headings, paragraphs, bullets). Do **not** embed whole 11K-char articles (too diffuse) and do **not** embed every one-line bullet alone (too sparse).

Strategy:
- Split on newlines into blocks.
- Greedily merge consecutive blocks into chunks of roughly **150–250 tokens** (~600–1200 chars), never splitting mid-block, preserving order.
- Keep `chunk_index` (document order).
- Prepend the page `title` to each chunk's text **only for the embedding input** (adds topical context, improves retrieval); store `content` without the prepended title.
- Skip empty/whitespace chunks.

## 6. Matching Engine (`matcher.py`)
Input: `client_id`, `keyword`, `doc_paragraphs: list[{index, text}]`.

**Config (env-tunable):** `TOP_K=10`, `SIM_THRESHOLD=0.75` (cosine), `MAX_LINKS_PER_PARAGRAPH=2`, `MAX_TOTAL_LINKS=15`, `RRF_K=60`.

### Step A — Exact-keyword pass
- Find the best **target page** for the keyword:
  ```sql
  SELECT p.id, p.url, p.title, count(*) AS hits,
         bool_or(p.title ILIKE '%' || :kw || '%') AS in_title
  FROM chunks c JOIN pages p ON p.id = c.page_id
  WHERE c.client_id = :cid
    AND p.url IS DISTINCT FROM :current_url   -- never link a post to itself
    AND c.content ILIKE '%' || :kw || '%'
  GROUP BY p.id, p.url, p.title
  ORDER BY in_title DESC, hits DESC
  LIMIT 3;
  ```
  Pick the top page as the exact target.
- Find doc paragraphs containing the keyword (case-insensitive, word-boundary). Take the **first** occurrence (SEO best practice: link a keyword once).
- Emit `{doc_paragraph_index, anchor_text = matched keyword, target_url, match_type="exact", confidence=1.0}`.

### Step B — Semantic pass (per doc paragraph)
For each paragraph:
1. Embed it: `task_type="RETRIEVAL_QUERY"`, 768-d.
2. **Vector search:**
   ```sql
   SELECT c.page_id, p.url, p.title, c.content,
          1 - (c.embedding <=> :qvec) AS sim
   FROM chunks c JOIN pages p ON p.id = c.page_id
   WHERE c.client_id = :cid AND p.url IS DISTINCT FROM :current_url
   ORDER BY c.embedding <=> :qvec
   LIMIT :top_k;
   ```
3. **Keyword search** on salient terms of the paragraph (`websearch_to_tsquery`), ranked by `ts_rank`.
4. **Fuse** the two ranked lists with Reciprocal Rank Fusion: `score(page) = Σ 1/(RRF_K + rank_in_list)`. Aggregate to page level.
5. **Filter:** keep candidates with vector `sim >= SIM_THRESHOLD`; drop the exact-pass target; drop the current post itself (the `:current_url` filter is already in the SQL above).
6. **LLM gate + anchor** (Gemini Flash) for each surviving top candidate — see §7. Validate the returned anchor actually appears in the paragraph; if not, drop the suggestion.
7. Keep `should_link=true`; cap at `MAX_LINKS_PER_PARAGRAPH`.

### Step C — Merge & rank
- Combine exact + semantic. **De-dupe by `target_url`** (a given post is linked at most once across the doc — keep the highest-confidence occurrence). Enforce `MAX_TOTAL_LINKS`.
- Sort (exact first, then semantic by confidence). Return JSON + render a table.

## 7. LLM Prompt (relevance gate + anchor extraction)
Call the Flash model with JSON-only output. Template:

```
System: You place internal links for SEO. You are precise and conservative:
only link when the target article is genuinely relevant to the paragraph.

User:
NEW ARTICLE PARAGRAPH:
"""{paragraph_text}"""

CANDIDATE ARTICLE TO LINK TO:
Title: {candidate_title}
Excerpt: {candidate_excerpt}   # first ~300 chars of the candidate

Decide whether linking this paragraph to the candidate article is
editorially helpful for a reader. If yes, choose an anchor phrase of
2–6 words that appears VERBATIM in the paragraph and best describes the
target. If no natural verbatim anchor exists, answer no.

Return ONLY JSON:
{"should_link": true|false, "anchor": "<verbatim phrase or empty>", "reason": "<short>"}
```
Parse JSON defensively (strip code fences). **Reject** any result where `anchor` is not a case-insensitive substring of the paragraph.

## 8. Google Docs Integration (`gdocs.py`)
- **Auth:** service account with Docs read scope is simplest for a tool (share the doc with the SA email); OR OAuth if reading the user's own docs. Put creds path in env.
- **Read:** `documents.get(documentId)` → walk `body.content` → for each paragraph element, concatenate its `textRun` contents → produce `{index, text, start_index, end_index}`. Keep the offsets for future write-back.
- **Doc URL → ID:** extract the ID from `https://docs.google.com/document/d/<ID>/edit`.
- **v1 fallback:** if Docs API setup blocks progress, also accept a pasted `.txt`/`.md` or raw text input, but keep the Docs reader as the primary path.
- **(Phase 3) Write-back:** `documents.batchUpdate` with an `updateTextStyle` request setting a `link.url` over the anchor's `[start,end)` range.

## 8b. Auto-saving the processed post (closing the loop)
Every doc processed is treated as a final post, so `suggest` ingests it into the same library automatically — there is no separate command and no draft/published distinction. Within a single `suggest` run:

1. Read the Doc and generate the link suggestions (§6) **first**.
2. **Then** ingest the post itself: chunk → embed (`task_type=RETRIEVAL_DOCUMENT`) → upsert a `pages` row + its `chunks`, exactly like the bulk ingest (§4/§5), keyed by the post's final URL.

Two rules make this safe:
- **Save after matching, never before** — so the post can't match against itself within its own run.
- **Exclude the current post's URL from target selection** — both matching queries filter `p.url IS DISTINCT FROM :current_url`, so even on a re-run (when the post is already in the DB) it is never suggested as a link to itself.

Upsert is idempotent: delete-then-insert the page's chunks by `page_id`, so re-running the same post refreshes that row instead of duplicating it. This keeps the corpus self-maintaining — today 167 posts; every post you run adds one more link target, hands-off, with no scraping step.

## 9. Interface
**v1 CLI (`cli.py`):**
```
python cli.py ingest  --client gokwik --file Gokwik_content.xlsx
python cli.py suggest --client gokwik --keyword "cart abandonment" \
                      --doc-url <google-doc-url> --url <final-live-url>
```
`suggest` prints the table, writes `suggestions.json`, and then auto-ingests the post into the library (§8b). `--url` is the post's final live URL: used both to exclude the post from its own target selection and as the upsert key when saving it.

**v2 API (`app.py`, FastAPI):**
- `POST /ingest` — `{client, file}`
- `POST /suggest` — `{client, keyword, doc_url, url}` → `suggestions[]` (also auto-saves the post)
- `GET /health`
- A single HTML form (Jinja or static) posts to `/suggest` and renders the results table.

## 10. Config & Secrets (`.env.example`)
```
GEMINI_API_KEY=
DATABASE_URL=postgresql://user:pass@localhost:5432/linker
GOOGLE_APPLICATION_CREDENTIALS=./service-account.json
EMBED_MODEL=gemini-embedding-001
EMBED_DIM=768
LLM_MODEL=gemini-flash-latest
TOP_K=10
SIM_THRESHOLD=0.75
MAX_LINKS_PER_PARAGRAPH=2
MAX_TOTAL_LINKS=15
RRF_K=60
```
Never hardcode secrets. Load via `python-dotenv`.

## 11. Build Order (milestones)
1. **M1 — DB + ingestion.** `schema.sql`, DB connection, `ingest.py`. Ingest the 167-row xlsx. Verify: pages=167, chunks > 0 per page, embeddings non-null. This is the foundation — get it solid before matching.
2. **M2 — Exact-keyword matching + CLI.** Step A + `cli.py suggest`. Verify a known keyword returns the right target page and paragraph.
3. **M3 — Semantic + hybrid + LLM gate.** Steps B & C, RRF, threshold, prompt, anchor validation, dedupe/caps.
4. **M4 — Google Docs read integration.** Replace pasted-text input with real Docs reading.
5. **M4b — Auto-save the processed post.** After `suggest` produces results, ingest that post into the same DB (§8b), upserting by URL. Confirm the post is saved *after* matching, and that both target queries exclude the current post's own URL (no self-links, even on re-runs).
5. **M5 *(optional)* — FastAPI + minimal web UI.**
6. **M6 *(optional)* — write links back into the Doc.**

## 12. Testing
- **Unit:** chunking boundaries; RRF fusion; anchor-validation (anchor must be a substring of the paragraph); threshold filtering; client isolation (queries always include `client_id`).
- **Integration:** ingest a small sample; run `suggest` on a doc known to relate to a specific post; assert the exact-keyword target is found and no cross-client rows appear.
- **Manual eval:** run on a few real new posts; measure precision (accepted ÷ suggested); tune `SIM_THRESHOLD` accordingly.

## 13. Notes / Decisions
- The `keyword` is the **anchor intent**; the **target** is chosen by relevance (keyword-in-title / match count), not merely "any post that contains the keyword."
- **Precision over recall:** when in doubt, return fewer links. A wrong internal link is worse than a missing one — hence the similarity threshold *and* the LLM gate *and* verbatim-anchor validation.
- `SIM_THRESHOLD` is a starting guess; tune empirically on real data.
- Future: add a `priority`/`pillar` flag on `pages` to bias targets toward cornerstone content; add bidirectional linking (old → new).
