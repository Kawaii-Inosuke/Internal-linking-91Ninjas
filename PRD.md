# PRD — Internal Linking Assistant

## 1. Summary
A tool that recommends **internal links for a new blog post before it goes live**. Given a target keyword, the new post's Google Doc, and a client name, it searches the client's existing published blog library and returns a ranked list of internal-link suggestions — each with the existing blog to link to, the anchor text to use, and the exact paragraph in the new doc where the link belongs.

The first client is **GoKwik** (167 blog posts already scraped into a spreadsheet with `link`, `title`, `content`).

## 2. Problem
Internal linking is high-value for SEO but done manually and inconsistently. Writers can't remember which of 150+ existing posts are relevant to a new article, so link opportunities are missed and anchor text is chosen ad hoc. We want to automate finding the *right* internal links for each new post, with a strong bias toward **precision** — a wrong or irrelevant link is worse than a missing one.

## 3. Goals
- Input: `keyword`, `Google Doc URL` (new post), `client name`. Output: ranked internal-link suggestions.
- Support **two matching modes**:
  1. **Exact-keyword** (user-directed): find the best existing post to link for the given keyword, and the paragraph in the new doc where that keyword appears (the anchor location).
  2. **Contextual/semantic** (discovery): for each paragraph of the new doc, surface topically related existing posts even when they share no exact keyword.
- Every suggestion includes anchor text that **appears verbatim** in the new doc's paragraph.
- Suggestions are filtered for precision (confidence threshold + LLM relevance check), de-duplicated, and capped.
- Runs on a **free stack** (Postgres + pgvector, Gemini free tier), **single client** to start.

## 4. Non-Goals (v1)
- Multi-client at scale (schema supports it; v1 targets GoKwik only).
- Automatically writing links back into the Google Doc (stretch, Phase 3).
- Bidirectional linking (adding links from *old* posts to the *new* one) — future.
- A polished multi-user web app — v1 is a CLI/script; a minimal web form is Phase 2.
- Re-scraping content — the corpus is already scraped; ingestion reads the existing spreadsheet/CSV.

## 5. Users & Stories
**User:** an SEO / content team member preparing a post for publication.

- As a writer, I paste my new post's Google Doc link + a target keyword + the client, and get a list of existing posts to link to, with suggested anchor text and the paragraph to place each link.
- As an editor, each suggestion shows a confidence score and the paragraph context, so I can accept/reject quickly.
- As a user, I never want a link suggested to a different client's content, or to a page that isn't genuinely relevant.

## 6. Functional Requirements
- **FR1** Ingest existing blog content (`link`, `title`, `content`) into a searchable store (full-text + vector).
- **FR2** Accept input: `keyword` (string), `google_doc_url`, `client`, and the post's final live `url`.
- **FR3** Read the Google Doc and split it into ordered paragraphs (retain paragraph index + text; retain character offsets for future write-back).
- **FR4** Exact-keyword matching: find the best target post for the keyword, and the doc paragraph(s) containing the keyword (link the first prominent occurrence only).
- **FR5** Contextual matching: for each doc paragraph, retrieve topically related existing posts via hybrid search.
- **FR6** Anchor text for every suggestion must be a phrase that appears verbatim in the corresponding doc paragraph.
- **FR7** Filter by a confidence threshold; de-duplicate (one link per target across the doc); cap links per paragraph and total.
- **FR8** Exclude self-links and the target already chosen by the exact pass.
- **FR9** Output suggestions as structured JSON **and** a human-readable table.
- **FR10** *(Stretch)* Write accepted links back into the Google Doc at the correct text range.
- **FR11** After producing suggestions, automatically save the processed post into the library (chunk → embed → upsert by URL) so future posts can link to it. Every doc is treated as final; the save happens *after* matching, and the post is excluded from its own target selection so it never self-links. No separate command and no manual step.

## 7. Inputs & Outputs
**Input:** `{ client, keyword, google_doc_url }`

**Output — one object per suggested link:**
| field | meaning |
|---|---|
| `doc_paragraph_index` | which paragraph in the new doc |
| `doc_paragraph_excerpt` | short snippet for context |
| `anchor_text` | phrase to hyperlink (verbatim from the paragraph) |
| `target_url` | existing blog to link to |
| `target_title` | its title |
| `match_type` | `exact` or `semantic` |
| `confidence` | 0–1 score |

## 8. Success Metrics
- **Precision** (accepted ÷ suggested) ≥ ~70% in manual review — this is the primary metric.
- **Coverage:** when a valid target for the keyword exists, the exact-keyword link is always found.
- **Latency:** results for a typical post (~30 paragraphs) in under ~2 minutes.
- **Zero cross-client leakage.**

## 9. Phasing
- **Phase 1 (MVP):** ingestion + exact + semantic matching + CLI output. Single client, free stack.
- **Phase 2:** minimal web UI (form → results table); empirical threshold tuning.
- **Phase 3:** write links back into the Google Doc; multi-client; bidirectional linking.

## 10. Constraints
- **Free stack:** Postgres + pgvector (local Docker or Supabase/Neon free tier); Gemini free tier for embeddings and the LLM step. Must respect Gemini rate limits (throttle + retry).
- **Data shape:** 167 rows, columns `link, title, content`. `content` is newline-separated blocks (headings, paragraphs, bullets) with clean article text (≈1.2K–31K chars, median ≈11K).
