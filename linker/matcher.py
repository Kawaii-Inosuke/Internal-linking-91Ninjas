"""Matching engine (TRD §6).

M2 implements **Step A — the exact-keyword pass** only: given a client, a
keyword, the new post's text, and its final URL, find existing posts to link the
keyword to and where in the new post the anchor belongs. The semantic / hybrid /
LLM-gate passes (Steps B & C) arrive in M3.

Rather than a single winner, Step A now returns a **ranked, scored shortlist** of
the existing posts that contain the keyword, so a human can judge and pick. Every
candidate literally contains the keyword, so the ranking is NOT an ML confidence
— it is a *heuristic relevance score* that estimates how canonical each post is
for the keyword, blended from a few cheap signals (see :func:`rank_candidates`).

This module is the single source of matching logic: both front doors — the CLI
(``cli.py suggest``) and the web UI (``app.py``) — call :func:`suggest` /
:func:`suggest_with_config`. Neither re-implements any matching.

The public entry point takes clean text (``post_text``), so M4 can feed it either
pasted text or Google-Docs-API output without touching the matching code.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import psycopg

from . import db
from .chunking import split_blocks
from .config import Config
from .db import ExactTarget

# Status codes for a suggest run. "ok" and "no_anchor" both carry a shortlist;
# every other value is a clean "no shortlist" outcome (never an exception).
STATUS_OK = "ok"
STATUS_NO_ANCHOR = "no_anchor"  # shortlist found, but the keyword is not in the post
STATUS_EMPTY_KEYWORD = "empty_keyword"
STATUS_EMPTY_POST = "empty_post"
STATUS_UNKNOWN_CLIENT = "unknown_client"
STATUS_NO_TARGET = "no_target"

# --------------------------------------------------------------------------- #
# Ranking heuristic — tunable weights and knobs
# --------------------------------------------------------------------------- #
# The exact-keyword shortlist is ranked by how *canonical* a post is for the
# keyword, NOT by an ML/probability score: every candidate already contains the
# keyword, so the question is only "which of these is the most authoritative
# target?". We answer it with a weighted blend of cheap structural signals.
#
# Signal strengths (higher weight == stronger evidence of canonicalness):
#   * TITLE   — keyword in the page title. The single strongest signal: a post
#               titled for the keyword is almost always its canonical home.
#   * HEADING — keyword in an H1/H2-style heading in the body.
#   * EARLY   — the keyword appears early in the post (a definitional guide leads
#               with the term; a passing mention buries it).
#   * FREQUENCY — how many of the post's chunks contain the keyword. Deliberately
#               the WEAKEST signal and log-dampened (see below): frequency is not
#               authority. A comparison post that repeats "X vs Y" many times must
#               NOT outrank the canonical "What is X" guide on repetition alone.
# Tune these to reweight the ranking; they are the only numbers that matter.
WEIGHT_TITLE = 6.0
WEIGHT_HEADING = 3.0
WEIGHT_EARLY = 2.0
WEIGHT_FREQUENCY = 1.0

# Frequency is log-scaled and saturates: beyond this many matching chunks, extra
# repetition adds essentially nothing. Keeps a spammy post from buying rank with
# volume. Must be > 1 (log1p(1) is the denominator's floor).
FREQUENCY_SATURATION = 20

# The keyword counts as appearing "early" (and so lists the ``early`` signal)
# when its normalized first-occurrence position is within roughly the first
# chunk of the post. See :func:`_early_signal` for the position model.
EARLY_SIGNAL_THRESHOLD = 0.5

# Names reported in ``signals_present`` when a signal fires. Frequency is
# intentionally omitted: it is a weak tiebreaker, not a signal a human should
# lean on to trust the ranking.
SIGNAL_TITLE = "title"
SIGNAL_HEADING = "heading"
SIGNAL_EARLY = "early"

# Heading detection (best-effort). The ingested corpus stores no heading markup,
# so a "heading" is inferred structurally: a short line that reads like a title
# rather than a sentence. Deliberately conservative to limit false positives.
_HEADING_MAX_WORDS = 12
_HEADING_MAX_CHARS = 90

# How many matching pages to pull into the candidate pool before re-ranking. The
# pool is a superset of the returned shortlist so a strong-but-infrequent post
# (e.g. keyword-in-title, appears early, few chunks) can be promoted above a
# high-frequency one. A single keyword never matches anywhere near this many
# pages, so in practice the whole matching set is ranked.
_CANDIDATE_POOL_SIZE = 50


@dataclass(frozen=True)
class Paragraph:
    """A single paragraph of the new post, in document order."""

    index: int
    text: str


@dataclass(frozen=True)
class Suggestion:
    """One row of the ranked exact-keyword shortlist (TRD §6 Step A output).

    ``score_out_of_10`` is a heuristic RELEVANCE score (how canonical this post is
    for the keyword), not a probability or confidence. ``anchor_text`` /
    ``doc_paragraph_index`` locate the keyword's first occurrence in the *new*
    post and are the same across every row (SEO best practice: link a keyword
    once); they are ``None`` when the keyword does not appear in the new post.
    ``signals_present`` lists which structural signals fired (title/heading/early)
    so a human can see *why* a row ranks where it does.
    """

    rank: int
    target_url: str
    target_title: str | None
    anchor_text: str | None
    doc_paragraph_index: int | None
    match_type: str  # "exact" for the Step A pass
    score_out_of_10: int  # heuristic relevance 0-10, NOT a probability
    signals_present: list[str]


@dataclass(frozen=True)
class SuggestResult:
    """The outcome of a suggest run: any suggestions plus a status/message.

    ``suggestions`` is the ranked shortlist (possibly empty). It is non-empty for
    both ``STATUS_OK`` and ``STATUS_NO_ANCHOR`` (the latter meaning the shortlist
    was found but the keyword is absent from the new post, so no anchor location
    could be marked); it is empty for every other status. ``message`` is a short,
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


def _looks_like_heading(line: str) -> bool:
    """Heuristically decide whether ``line`` reads like a heading, not prose.

    The corpus stores no heading markup, so this is best-effort: a heading is a
    short line that does not end like a sentence. Conservative by design (short,
    no terminal period) to keep prose from being mistaken for a heading.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _HEADING_MAX_CHARS:
        return False
    if len(stripped.split()) > _HEADING_MAX_WORDS:
        return False
    # Headings rarely end in a full stop; questions ("What is X?") are fine.
    return not stripped.endswith(".")


def _keyword_in_heading(content: str, pattern: re.Pattern[str]) -> bool:
    """Whether ``pattern`` matches inside a heading-like line of ``content``."""
    return any(
        _looks_like_heading(line) and pattern.search(line)
        for line in (content or "").split("\n")
    )


def _early_signal(content: str, pattern: re.Pattern[str], first_chunk_index: int) -> float:
    """Score how early the keyword first appears, in ``[0, 1]`` (1.0 == very start).

    ``content`` is the page's keyword-bearing chunks joined in order, so the first
    match within it is the keyword's first occurrence in the post. Position is
    modeled as ``first_chunk_index + (offset fraction within that content)`` — i.e.
    "how many chunks deep" the first mention is — and mapped through ``1/(1+pos)``
    so a lead-sentence mention scores near 1.0 and a mention several chunks in
    decays toward 0. ``first_chunk_index`` anchors the coarse position even when
    the offset within the joined content is small.
    """
    if not content:
        return 0.0
    match = pattern.search(content)
    offset_fraction = (match.start() / len(content)) if match else 0.0
    position = first_chunk_index + offset_fraction
    return 1.0 / (1.0 + position)


def _frequency_signal(hits: int) -> float:
    """Log-dampened, saturating frequency score in ``[0, 1]``.

    ``hits`` is a proxy for keyword frequency (the number of the post's chunks
    that contain the keyword). Log-scaling plus saturation means 15 vs 12 hits is
    a hair's difference, not a landslide — frequency nudges ties, it does not
    decide the ranking (it also carries the lowest weight).
    """
    if hits <= 0:
        return 0.0
    return min(1.0, math.log1p(hits) / math.log1p(FREQUENCY_SATURATION))


def _score_candidate(
    target: ExactTarget, pattern: re.Pattern[str]
) -> tuple[float, int, list[str]]:
    """Score one candidate: return ``(raw_score, score_out_of_10, signals_present)``.

    ``raw_score`` is the weighted blend of the four signals (used for ordering);
    ``score_out_of_10`` is that blend rescaled to an integer 0-10 for display.
    ``signals_present`` lists the fired structural signals (title/heading/early);
    frequency is excluded on purpose (it is a weak tiebreaker, not trust-worthy
    evidence on its own).
    """
    signals: list[str] = []

    title = 1.0 if target.in_title else 0.0
    if title:
        signals.append(SIGNAL_TITLE)

    heading = 1.0 if _keyword_in_heading(target.matched_content, pattern) else 0.0
    if heading:
        signals.append(SIGNAL_HEADING)

    early = _early_signal(target.matched_content, pattern, target.first_chunk_index)
    if early >= EARLY_SIGNAL_THRESHOLD:
        signals.append(SIGNAL_EARLY)

    frequency = _frequency_signal(target.hits)

    raw = (
        WEIGHT_TITLE * title
        + WEIGHT_HEADING * heading
        + WEIGHT_EARLY * early
        + WEIGHT_FREQUENCY * frequency
    )
    max_raw = WEIGHT_TITLE + WEIGHT_HEADING + WEIGHT_EARLY + WEIGHT_FREQUENCY
    score_out_of_10 = round(10 * raw / max_raw)
    return raw, score_out_of_10, signals


def rank_candidates(
    candidates: list[ExactTarget],
    *,
    keyword: str,
    anchor_text: str | None,
    doc_paragraph_index: int | None,
    max_results: int,
) -> list[Suggestion]:
    """Rank ``candidates`` into a scored shortlist of at most ``max_results`` rows.

    Each candidate is scored by :func:`_score_candidate` (title / heading / early /
    frequency blend). The list is sorted by the raw blended score, descending; the
    original DB order (``in_title DESC, hits DESC``) breaks ties because the sort
    is stable. Because the score weights TITLE and EARLY far above (log-dampened)
    FREQUENCY, a canonical guide that leads with the keyword outranks a comparison
    post that merely repeats it — repetition alone cannot buy the top spot.

    ``anchor_text`` / ``doc_paragraph_index`` are the keyword's first-occurrence
    location in the *new* post (shared by every row, or ``None`` when the keyword
    is absent from the post) and are stamped onto each shortlist row.
    """
    pattern = _keyword_pattern(keyword)
    scored = [(target, *_score_candidate(target, pattern)) for target in candidates]
    scored.sort(key=lambda item: item[1], reverse=True)  # item[1] == raw_score

    return [
        Suggestion(
            rank=rank,
            target_url=target.url,
            target_title=target.title,
            anchor_text=anchor_text,
            doc_paragraph_index=doc_paragraph_index,
            match_type="exact",
            score_out_of_10=score_out_of_10,
            signals_present=signals,
        )
        for rank, (target, _raw, score_out_of_10, signals) in enumerate(
            scored[:max_results], start=1
        )
    ]


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
    max_results: int = 8,
) -> SuggestResult:
    """Run the exact-keyword pass and return a ranked-shortlist :class:`SuggestResult`.

    Steps (TRD §6 Step A):
      1. Resolve ``client`` to its id (client isolation for all downstream SQL).
      2. Find the client's pages whose chunks contain ``keyword``, excluding
         ``current_url`` (never link a post to itself).
      3. Rank them into a shortlist of at most ``max_results`` by heuristic
         relevance (title / heading / early-position / dampened frequency).
      4. Locate the keyword's first occurrence in ``post_text`` for the anchor.

    Never raises for the ordinary cases: an empty/whitespace keyword, an empty
    post, an unknown client, and "no page contains the keyword" each return an
    empty shortlist with an explanatory status/message. When the keyword IS found
    in the corpus but NOT in the new post, the shortlist is still returned
    (``STATUS_NO_ANCHOR``) with ``anchor_text``/``doc_paragraph_index`` set to
    ``None`` — the targets are useful even when the anchor location is unknown.
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

    pool_size = max(max_results, _CANDIDATE_POOL_SIZE)
    candidates = db.find_exact_target_candidates(
        conn, client_id, keyword, current_url, pool_size
    )
    if not candidates:
        return SuggestResult(
            STATUS_NO_TARGET,
            f"No existing {client} post contains the keyword {keyword!r}.",
            [],
        )

    paragraph = find_first_keyword_paragraph(paragraphs, keyword)
    anchor_text = keyword if paragraph is not None else None
    doc_paragraph_index = paragraph.index if paragraph is not None else None

    shortlist = rank_candidates(
        candidates,
        keyword=keyword,
        anchor_text=anchor_text,
        doc_paragraph_index=doc_paragraph_index,
        max_results=max_results,
    )

    if paragraph is None:
        return SuggestResult(
            STATUS_NO_ANCHOR,
            (
                f"{len(shortlist)} candidate post(s) contain {keyword!r}, but it does "
                f"not appear in the new post — no anchor location could be marked."
            ),
            shortlist,
        )
    return SuggestResult(
        STATUS_OK,
        f"{len(shortlist)} candidate post(s) ranked for {keyword!r}.",
        shortlist,
    )


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
    exist (ingestion provisions it), so DDL is skipped. ``max_results`` comes from
    ``config.exact_max_results`` (env ``EXACT_MAX_RESULTS``).
    """
    conn = db.connect(config.database_url, apply_schema=False)
    try:
        return suggest(
            conn,
            client=client,
            keyword=keyword,
            post_text=post_text,
            current_url=current_url,
            max_results=config.exact_max_results,
        )
    finally:
        conn.close()
