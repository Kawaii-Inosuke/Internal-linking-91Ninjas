"""Matching engine (TRD §6).

M2 implements **Step A — the exact-keyword pass** only: given a client, a
keyword, the new post's text, and its final URL, find the best existing post to
link the keyword to and where in the new post the anchor belongs. The semantic /
hybrid / LLM-gate passes (Steps B & C) arrive in M3.

This module is the single source of matching logic: both front doors — the CLI
(``cli.py suggest``) and the web UI (``app.py``) — call :func:`suggest` /
:func:`suggest_with_config`. Neither re-implements any matching.

The public entry point takes clean text (``post_text``), so M4 can feed it either
pasted text or Google-Docs-API output without touching the matching code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import psycopg

from . import db
from .chunking import split_blocks
from .config import Config
from .db import ExactTarget

# Status codes for a suggest run. "ok" means a suggestion was produced; every
# other value is a clean "no match" outcome (never an exception).
STATUS_OK = "ok"
STATUS_EMPTY_KEYWORD = "empty_keyword"
STATUS_EMPTY_POST = "empty_post"
STATUS_UNKNOWN_CLIENT = "unknown_client"
STATUS_NO_TARGET = "no_target"
STATUS_KEYWORD_NOT_IN_POST = "keyword_not_in_post"


@dataclass(frozen=True)
class Paragraph:
    """A single paragraph of the new post, in document order."""

    index: int
    text: str


@dataclass(frozen=True)
class Suggestion:
    """One internal-link suggestion (TRD §6 Step A output shape)."""

    doc_paragraph_index: int
    anchor_text: str
    target_url: str
    target_title: str | None
    match_type: str  # "exact" for the Step A pass
    confidence: float


@dataclass(frozen=True)
class SuggestResult:
    """The outcome of a suggest run: any suggestions plus a status/message.

    ``suggestions`` is empty for every non-``ok`` status; ``message`` is a short,
    human-readable explanation suitable for the CLI or the web UI.
    """

    status: str
    message: str
    suggestions: list[Suggestion]


# --------------------------------------------------------------------------- #
# Pure helpers (no DB, no network) — unit-tested directly
# --------------------------------------------------------------------------- #
def split_paragraphs(text: str) -> list[Paragraph]:
    """Split post ``text`` into ordered, non-empty paragraphs.

    Paragraphs are newline-separated blocks (the same convention the corpus uses,
    see :func:`linker.chunking.split_blocks`): each stripped, non-empty line
    becomes one paragraph, numbered from 0. Blank lines are dropped, so the index
    counts only content-bearing paragraphs.
    """
    return [Paragraph(index=i, text=block) for i, block in enumerate(split_blocks(text or ""))]


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Compile a case-insensitive, word-boundary matcher for ``keyword``.

    Uses ``(?<!\\w) … (?!\\w)`` rather than ``\\b … \\b``: the keyword must not be
    flanked by word characters (so ``"art"`` does not match inside ``"cart"``),
    while still matching keywords that begin or end with a non-word character
    (e.g. ``"20%"``), which bare ``\\b`` would reject. ``keyword`` is escaped, so
    regex metacharacters are matched literally.
    """
    return re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)", re.IGNORECASE)


def find_first_keyword_paragraph(
    paragraphs: list[Paragraph], keyword: str
) -> Paragraph | None:
    """Return the FIRST paragraph containing ``keyword``, or None if absent.

    Matching is case-insensitive and word-boundary aware (so keyword ``"art"``
    does not match ``"cart"``). SEO best practice is to link a keyword once, so
    only the first occurrence is used as the anchor location (TRD §6 Step A).
    """
    if not keyword:
        return None
    pattern = _keyword_pattern(keyword)
    for paragraph in paragraphs:
        if pattern.search(paragraph.text):
            return paragraph
    return None


def select_top_target(candidates: list[ExactTarget]) -> ExactTarget | None:
    """Pick the best target from ranked ``candidates`` (keyword-in-title, then hits).

    Mirrors the SQL ``ORDER BY in_title DESC, hits DESC`` (TRD §6 Step A) as a
    pure, testable function: a keyword-in-title page always beats one without,
    and among equals the higher chunk-match count wins. Returns None if empty.
    """
    if not candidates:
        return None
    # A page whose title holds the keyword is a stronger target than a mere
    # body mention; break ties by how many chunks matched.
    return max(candidates, key=lambda t: (t.in_title, t.hits))


# --------------------------------------------------------------------------- #
# Exact-keyword pass (TRD §6 Step A)
# --------------------------------------------------------------------------- #
def suggest(
    conn: psycopg.Connection,
    *,
    client: str,
    keyword: str,
    post_text: str,
    current_url: str | None,
) -> SuggestResult:
    """Run the exact-keyword pass and return a :class:`SuggestResult`.

    Steps (TRD §6 Step A):
      1. Resolve ``client`` to its id (client isolation for all downstream SQL).
      2. Find the best target page for ``keyword`` among that client's chunks,
         excluding ``current_url`` (never link a post to itself).
      3. Find the first paragraph of ``post_text`` containing ``keyword``.
      4. Emit one exact suggestion (confidence 1.0).

    Never raises for the ordinary "no match" cases (empty/whitespace keyword,
    empty post, unknown client, no target page, keyword absent from the post):
    each returns an empty result with an explanatory status/message instead.
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return SuggestResult(STATUS_EMPTY_KEYWORD, "Enter a keyword to match.", [])

    paragraphs = split_paragraphs(post_text)
    if not paragraphs:
        return SuggestResult(STATUS_EMPTY_POST, "The post text is empty.", [])

    client_id = db.get_client_id(conn, client)
    if client_id is None:
        return SuggestResult(
            STATUS_UNKNOWN_CLIENT,
            f"No ingested corpus found for client {client!r}.",
            [],
        )

    candidates = db.find_exact_target_candidates(conn, client_id, keyword, current_url)
    target = select_top_target(candidates)
    if target is None:
        return SuggestResult(
            STATUS_NO_TARGET,
            f"No existing {client} post contains the keyword {keyword!r}.",
            [],
        )

    paragraph = find_first_keyword_paragraph(paragraphs, keyword)
    if paragraph is None:
        return SuggestResult(
            STATUS_KEYWORD_NOT_IN_POST,
            f"The keyword {keyword!r} does not appear in the post text.",
            [],
        )

    suggestion = Suggestion(
        doc_paragraph_index=paragraph.index,
        anchor_text=keyword,
        target_url=target.url,
        target_title=target.title,
        match_type="exact",
        confidence=1.0,
    )
    return SuggestResult(STATUS_OK, "1 exact-keyword suggestion.", [suggestion])


def suggest_with_config(
    config: Config,
    *,
    client: str,
    keyword: str,
    post_text: str,
    current_url: str | None,
) -> SuggestResult:
    """Open a (read-only) connection from ``config`` and run :func:`suggest`.

    Convenience wrapper so both the CLI and the web UI share one code path
    without each managing the connection lifecycle. The schema is assumed to
    exist (ingestion provisions it), so DDL is skipped.
    """
    conn = db.connect(config.database_url, apply_schema=False)
    try:
        return suggest(
            conn,
            client=client,
            keyword=keyword,
            post_text=post_text,
            current_url=current_url,
        )
    finally:
        conn.close()
