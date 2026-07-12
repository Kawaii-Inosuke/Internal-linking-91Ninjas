"""Unit tests for the exact-keyword pass (TRD §6 Step A / §12).

Pure-logic tests: no database and no network. They cover the two pieces of M2
logic that live outside SQL — first-occurrence anchor location and target
ranking — plus the ``suggest`` orchestration over a fake DB layer, so the clean
"no match" outcomes are exercised without a live Postgres.
"""
from __future__ import annotations

import pytest

from linker import matcher
from linker.db import ExactTarget
from linker.matcher import (
    Paragraph,
    find_first_keyword_paragraph,
    select_top_target,
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
# Exact-pass target ranking (mirrors SQL ORDER BY in_title DESC, hits DESC)
# --------------------------------------------------------------------------- #
def _target(page_id: int, hits: int, in_title: bool) -> ExactTarget:
    return ExactTarget(
        page_id=page_id, url=f"https://x/{page_id}", title="t", hits=hits, in_title=in_title
    )


def test_title_match_beats_higher_hit_count():
    # An in-title page wins even though the other has far more body hits.
    candidates = [_target(1, hits=50, in_title=False), _target(2, hits=1, in_title=True)]
    top = select_top_target(candidates)
    assert top is not None and top.page_id == 2


def test_among_equal_title_status_more_hits_wins():
    candidates = [_target(1, hits=3, in_title=False), _target(2, hits=9, in_title=False)]
    top = select_top_target(candidates)
    assert top is not None and top.page_id == 2


def test_empty_candidates_returns_none():
    assert select_top_target([]) is None


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

    def find_exact_target_candidates(self, conn, client_id, keyword, current_url):
        # Record the client_id filter to assert client isolation is honoured.
        self.calls.append((client_id, keyword, current_url))
        return self._candidates


@pytest.fixture
def patch_db(monkeypatch):
    def _install(client_id, candidates):
        fake = _FakeDB(client_id, candidates)
        monkeypatch.setattr(matcher, "db", fake)
        return fake

    return _install


_POST = "Intro paragraph.\nReducing cart abandonment lifts revenue.\nOutro."


def test_suggest_happy_path(patch_db):
    fake = patch_db(client_id=7, candidates=[_target(42, hits=5, in_title=True)])
    result = matcher.suggest(
        conn=None,
        client="gokwik",
        keyword="cart abandonment",
        post_text=_POST,
        current_url="https://x/new",
    )
    assert result.status == matcher.STATUS_OK
    assert len(result.suggestions) == 1
    s = result.suggestions[0]
    assert s.doc_paragraph_index == 1
    assert s.anchor_text == "cart abandonment"
    assert s.target_url == "https://x/42"
    assert s.match_type == "exact"
    assert s.confidence == 1.0
    # The query was scoped to the resolved client_id and excluded the post's URL.
    assert fake.calls == [(7, "cart abandonment", "https://x/new")]


def test_suggest_empty_keyword_is_clean(patch_db):
    patch_db(client_id=7, candidates=[_target(42, 5, True)])
    result = matcher.suggest(
        conn=None, client="gokwik", keyword="   ", post_text=_POST, current_url=None
    )
    assert result.status == matcher.STATUS_EMPTY_KEYWORD
    assert result.suggestions == []


def test_suggest_empty_post_is_clean(patch_db):
    patch_db(client_id=7, candidates=[_target(42, 5, True)])
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


def test_suggest_keyword_not_in_post_is_clean(patch_db):
    patch_db(client_id=7, candidates=[_target(42, 5, True)])
    result = matcher.suggest(
        conn=None,
        client="gokwik",
        keyword="cart abandonment",
        post_text="This post never mentions the phrase.",
        current_url=None,
    )
    assert result.status == matcher.STATUS_KEYWORD_NOT_IN_POST
    assert result.suggestions == []
