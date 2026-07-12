"""Unit tests for ingest planning (TRD §4/§5).

Pure-logic tests over ``_plan_pages``: no database and no embedding model. The
focus is the M1 blank-row filter — rows whose ``content`` is empty or
whitespace-only must be dropped before any chunking/embedding happens.
"""
from __future__ import annotations

from linker.ingest import IngestStats, _plan_pages, _Row


def _plan(rows: list[_Row]) -> tuple[list, IngestStats]:
    stats = IngestStats()
    pending = _plan_pages(rows, stats)
    return pending, stats


def test_blank_and_whitespace_content_rows_are_dropped():
    rows = [
        _Row(url="https://x.com/a", title="A", content="Real content about carts."),
        _Row(url="", title="", content=""),  # fully blank trailing row
        _Row(url="https://x.com/b", title="B", content="   \n\t  "),  # whitespace-only
        _Row(url="https://x.com/c", title="C", content="More real content here."),
    ]
    pending, stats = _plan(rows)

    # Only the two content-bearing rows survive to become pages.
    assert [p.url for p in pending] == ["https://x.com/a", "https://x.com/c"]
    # Both content-less rows are counted as blank and reach neither chunk nor embed.
    assert stats.blank_rows == 2
    assert stats.skipped_rows == 0


def test_content_bearing_row_produces_chunks():
    pending, stats = _plan(
        [_Row(url="https://x.com/a", title="Title", content="A paragraph of text.")]
    )
    assert len(pending) == 1
    assert pending[0].chunks  # chunked, not empty
    assert stats.blank_rows == 0


def test_content_without_url_is_an_anomaly_not_blank():
    # Content present but no link: a real anomaly (skipped), NOT a blank row.
    pending, stats = _plan(
        [_Row(url="", title="Orphan", content="Has content but no link.")]
    )
    assert pending == []
    assert stats.skipped_rows == 1
    assert stats.blank_rows == 0
