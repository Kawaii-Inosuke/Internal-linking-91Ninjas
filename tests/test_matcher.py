"""Unit tests for the exact-keyword pass (TRD §6 Step A / §12).

Pure-logic tests: no database and no network. They cover the pieces of M2 logic
that live outside SQL — first-occurrence anchor location and the ranked-shortlist
scoring/ordering — plus the ``suggest`` orchestration over a fake DB layer, so the
clean "no match" and "no anchor" outcomes are exercised without a live Postgres.
"""
from __future__ import annotations

import pytest

from linker import matcher
from linker.db import ExactTarget
from linker.matcher import (
    Paragraph,
    find_first_keyword_paragraph,
    rank_candidates,
    split_paragraphs,
)


# --------------------------------------------------------------------------- #
# First-occurrence anchor logic
# --------------------------------------------------------------------------- #
def test_split_paragraphs_indexes_non_empty_blocks():
    paras = split_paragraphs("First line.\n\n  Second line.  \n\nThird.")
    assert paras == [
        Paragraph(0, "First line."),
        Paragraph(1, "Second line."),
        Paragraph(2, "Third."),
    ]


def test_split_paragraphs_empty_text_yields_nothing():
    assert split_paragraphs("") == []
    assert split_paragraphs("   \n\t\n ") == []


def test_first_occurrence_is_returned():
    paras = split_paragraphs(
        "Intro with no match.\n"
        "Reducing cart abandonment boosts revenue.\n"
        "More on cart abandonment later."
    )
    hit = find_first_keyword_paragraph(paras, "cart abandonment")
    assert hit is not None
    assert hit.index == 1  # the FIRST paragraph mentioning it, not the third


def test_match_is_case_insensitive():
    paras = split_paragraphs("We study Cart Abandonment closely.")
    hit = find_first_keyword_paragraph(paras, "cart abandonment")
    assert hit is not None and hit.index == 0


def test_match_respects_word_boundaries():
    # "art" must not match inside "cart".
    paras = split_paragraphs("The cart is full.")
    assert find_first_keyword_paragraph(paras, "art") is None


def test_keyword_absent_returns_none():
    paras = split_paragraphs("Nothing relevant here.")
    assert find_first_keyword_paragraph(paras, "cart abandonment") is None


def test_regex_special_chars_in_keyword_are_literal():
    # A keyword with regex metacharacters must be matched literally, not as a pattern.
    paras = split_paragraphs("Save 20% today with checkout.")
    assert find_first_keyword_paragraph(paras, "20%") is not None
    assert find_first_keyword_paragraph(paras, "2.%") is None


# --------------------------------------------------------------------------- #
# Ranked-shortlist scoring & ordering
# --------------------------------------------------------------------------- #
def _target(
    page_id: int,
    *,
    hits: int,
    in_title: bool,
    first_chunk_index: int = 0,
    matched_content: str = "",
) -> ExactTarget:
    return ExactTarget(
        page_id=page_id,
        url=f"https://x/{page_id}",
        title=f"title {page_id}",
        hits=hits,
        in_title=in_title,
        first_chunk_index=first_chunk_index,
        matched_content=matched_content,
    )


def _rank(candidates, keyword="cash on delivery", max_results=8):
    return rank_candidates(
        candidates,
        keyword=keyword,
        anchor_text=keyword,
        doc_paragraph_index=0,
        max_results=max_results,
    )


def test_title_early_match_outranks_frequency_only_match():
    # A keyword-in-title page that leads with the keyword must beat a page that
    # merely repeats it many times in its body (frequency is the weakest signal).
    canonical = _target(
        1, hits=3, in_title=True, first_chunk_index=0,
        matched_content="cash on delivery is the default payment method.",
    )
    frequency_only = _target(
        2, hits=50, in_title=False, first_chunk_index=6,
        matched_content="... " * 40 + "cash on delivery appears late here.",
    )
    ranked = _rank([frequency_only, canonical])
    assert [s.target_url for s in ranked] == ["https://x/1", "https://x/2"]
    assert ranked[0].rank == 1 and ranked[1].rank == 2


def test_canonical_guide_beats_comparison_post_on_frequency():
    # The real gokwik case: both posts have the keyword in the title, but the
    # comparison post repeats it more often (higher hits) while the canonical
    # guide leads with it (chunk 0). The guide must still win — repetition is not
    # authority.
    guide = _target(
        1, hits=12, in_title=True, first_chunk_index=0,
        matched_content="cash on delivery (COD) remains the preferred option.",
    )
    comparison = _target(
        2, hits=15, in_title=True, first_chunk_index=1,
        matched_content="Later on, cash on delivery vs prepaid is compared.",
    )
    ranked = _rank([comparison, guide])
    assert ranked[0].target_url == "https://x/1"  # the canonical guide, not the comparison


def test_keyword_in_heading_boosts_rank():
    # Between two otherwise-equal body-only mentions, a keyword in a heading wins.
    with_heading = _target(
        1, hits=2, in_title=False, first_chunk_index=2,
        matched_content="What is cash on delivery?\nA short explainer follows here.",
    )
    without_heading = _target(
        2, hits=2, in_title=False, first_chunk_index=2,
        matched_content="Somewhere in this long prose paragraph we mention cash on delivery in passing today.",
    )
    ranked = _rank([without_heading, with_heading])
    assert ranked[0].target_url == "https://x/1"
    assert matcher.SIGNAL_HEADING in ranked[0].signals_present
    assert matcher.SIGNAL_HEADING not in ranked[1].signals_present


def test_signals_present_reports_fired_signals():
    target = _target(
        1, hits=4, in_title=True, first_chunk_index=0,
        matched_content="What is cash on delivery?\nCash on delivery explained.",
    )
    (row,) = _rank([target])
    assert set(row.signals_present) == {
        matcher.SIGNAL_TITLE,
        matcher.SIGNAL_HEADING,
        matcher.SIGNAL_EARLY,
    }


def test_score_is_integer_between_0_and_10():
    candidates = [
        _target(1, hits=12, in_title=True, first_chunk_index=0, matched_content="cash on delivery guide"),
        _target(2, hits=1, in_title=False, first_chunk_index=9, matched_content="x " * 50 + "cash on delivery"),
        _target(3, hits=50, in_title=True, first_chunk_index=0, matched_content="cash on delivery cash on delivery"),
    ]
    for row in _rank(candidates):
        assert isinstance(row.score_out_of_10, int)
        assert 0 <= row.score_out_of_10 <= 10


def test_top_row_scores_at_or_near_ten():
    # A page that fires every signal should land at the top of the 0-10 scale.
    perfect = _target(
        1, hits=20, in_title=True, first_chunk_index=0,
        matched_content="What is cash on delivery?\ncash on delivery is defined here.",
    )
    (row,) = _rank([perfect])
    assert row.score_out_of_10 >= 9


def test_max_results_caps_the_shortlist_and_numbers_ranks():
    candidates = [
        _target(i, hits=i, in_title=(i % 2 == 0), matched_content="cash on delivery")
        for i in range(1, 21)  # 20 candidates
    ]
    ranked = _rank(candidates, max_results=8)
    assert len(ranked) == 8
    assert [s.rank for s in ranked] == list(range(1, 9))


def test_empty_candidates_yields_empty_shortlist():
    assert _rank([]) == []


# --------------------------------------------------------------------------- #
# suggest() orchestration over a fake DB layer (no live Postgres)
# --------------------------------------------------------------------------- #
class _FakeDB:
    """Stand-in for linker.db, monkeypatched onto the matcher module."""

    def __init__(self, client_id, candidates):
        self._client_id = client_id
        self._candidates = candidates
        self.calls = []

    def get_client_id(self, conn, name):
        return self._client_id

    def find_exact_target_candidates(self, conn, client_id, keyword, current_url, limit):
        # Record the client_id filter + exclusion + pool size to assert isolation.
        self.calls.append((client_id, keyword, current_url, limit))
        return self._candidates


@pytest.fixture
def patch_db(monkeypatch):
    def _install(client_id, candidates):
        fake = _FakeDB(client_id, candidates)
        monkeypatch.setattr(matcher, "db", fake)
        return fake

    return _install


_POST = "Intro paragraph.\nReducing cart abandonment lifts revenue.\nOutro."


def test_suggest_happy_path_returns_ranked_shortlist(patch_db):
    fake = patch_db(
        client_id=7,
        candidates=[
            _target(42, hits=5, in_title=True, first_chunk_index=0,
                    matched_content="cart abandonment is a big problem."),
            _target(99, hits=2, in_title=False, first_chunk_index=4,
                    matched_content="x " * 30 + "cart abandonment"),
        ],
    )
    result = matcher.suggest(
        conn=None,
        client="gokwik",
        keyword="cart abandonment",
        post_text=_POST,
        current_url="https://x/new",
        max_results=8,
    )
    assert result.status == matcher.STATUS_OK
    assert len(result.suggestions) == 2
    top = result.suggestions[0]
    assert top.rank == 1
    assert top.target_url == "https://x/42"  # title + early beats the body-only page
    assert top.anchor_text == "cart abandonment"
    assert top.doc_paragraph_index == 1
    assert top.match_type == "exact"
    assert 0 <= top.score_out_of_10 <= 10
    # The query was scoped to the resolved client_id and excluded the post's URL;
    # the pool size (>= max_results) was passed through.
    (client_id, keyword, current_url, limit) = fake.calls[0]
    assert (client_id, keyword, current_url) == (7, "cart abandonment", "https://x/new")
    assert limit >= 8


def test_suggest_excludes_current_url_via_db_filter(patch_db):
    fake = patch_db(client_id=7, candidates=[_target(1, hits=1, in_title=True, matched_content="cart abandonment")])
    matcher.suggest(
        conn=None,
        client="gokwik",
        keyword="cart abandonment",
        post_text=_POST,
        current_url="https://x/self",
    )
    # current_url is handed to the DB layer, which applies IS DISTINCT FROM.
    assert fake.calls[0][2] == "https://x/self"


def test_suggest_keyword_absent_from_post_still_returns_shortlist(patch_db):
    patch_db(client_id=7, candidates=[_target(42, hits=5, in_title=True, matched_content="cart abandonment")])
    result = matcher.suggest(
        conn=None,
        client="gokwik",
        keyword="cart abandonment",
        post_text="This post never mentions the phrase.",
        current_url=None,
    )
    # Shortlist is returned, but flagged: no anchor location in the new post.
    assert result.status == matcher.STATUS_NO_ANCHOR
    assert len(result.suggestions) == 1
    assert result.suggestions[0].anchor_text is None
    assert result.suggestions[0].doc_paragraph_index is None


def test_suggest_empty_keyword_is_clean(patch_db):
    patch_db(client_id=7, candidates=[_target(42, hits=5, in_title=True)])
    result = matcher.suggest(
        conn=None, client="gokwik", keyword="   ", post_text=_POST, current_url=None
    )
    assert result.status == matcher.STATUS_EMPTY_KEYWORD
    assert result.suggestions == []


def test_suggest_empty_post_is_clean(patch_db):
    patch_db(client_id=7, candidates=[_target(42, hits=5, in_title=True)])
    result = matcher.suggest(
        conn=None, client="gokwik", keyword="cart abandonment", post_text="   ", current_url=None
    )
    assert result.status == matcher.STATUS_EMPTY_POST
    assert result.suggestions == []


def test_suggest_unknown_client_is_clean(patch_db):
    patch_db(client_id=None, candidates=[])
    result = matcher.suggest(
        conn=None, client="ghost", keyword="cart abandonment", post_text=_POST, current_url=None
    )
    assert result.status == matcher.STATUS_UNKNOWN_CLIENT
    assert result.suggestions == []


def test_suggest_no_target_is_clean(patch_db):
    patch_db(client_id=7, candidates=[])  # no page contains the keyword
    result = matcher.suggest(
        conn=None, client="gokwik", keyword="cart abandonment", post_text=_POST, current_url=None
    )
    assert result.status == matcher.STATUS_NO_TARGET
    assert result.suggestions == []
