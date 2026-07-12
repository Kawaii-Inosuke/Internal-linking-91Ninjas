"""Command-line interface for the Internal Linking Assistant.

M1 implements ``ingest``; M2 adds ``suggest`` (exact-keyword pass, TRD §6 Step A)
and ``serve`` (the FastAPI web UI). ``suggest`` shares one matching code path with
the web UI — see :mod:`linker.matcher`.

    python cli.py ingest  --client gokwik --file "Gokwik content.xlsx"
    python cli.py suggest --client gokwik --keyword "cart abandonment" \
                          --file new_post.txt --url https://example.com/new-post
    python cli.py serve
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from linker import matcher
from linker.config import Config, ConfigError
from linker.ingest import ingest_file

log = logging.getLogger("linker.cli")

SUGGESTIONS_PATH = Path("suggestions.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py", description="Internal Linking Assistant"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest a blog corpus (xlsx/CSV) for a client")
    ingest.add_argument("--client", required=True, help="Client name, e.g. gokwik")
    ingest.add_argument("--file", required=True, help="Path to the corpus .xlsx/.csv")

    suggest = sub.add_parser(
        "suggest", help="Suggest internal links for a new post (exact-keyword pass)"
    )
    suggest.add_argument("--client", required=True, help="Client name, e.g. gokwik")
    suggest.add_argument("--keyword", required=True, help='Anchor keyword, e.g. "cart abandonment"')
    src = suggest.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="Path to the new post as .txt/.md")
    src.add_argument("--text", help="The new post text passed inline")
    suggest.add_argument(
        "--url", default="", help="The post's final live URL (excluded from its own targets)"
    )

    serve = sub.add_parser("serve", help="Run the FastAPI web UI (http://localhost:8000)")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000, help="Bind port (default 8000)")
    return parser


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = Config.from_env()
    stats = ingest_file(config, args.file, args.client)
    print(f"Pages ingested  : {stats.pages}")
    print(f"Chunks created  : {stats.chunks}")
    print(f"Skipped rows    : {stats.skipped_rows}")
    if stats.blank_rows:
        print(f"Blank rows      : {stats.blank_rows} (empty spreadsheet rows, ignored)")
    if stats.zero_chunk_pages:
        print(f"Zero-chunk pages: {len(stats.zero_chunk_pages)}")
        for url in stats.zero_chunk_pages:
            print(f"  - {url}")
    return 0


def _read_post_text(args: argparse.Namespace) -> str:
    """Return the new post's text from ``--file`` or ``--text`` (M2 pasted-text input)."""
    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise FileNotFoundError(f"Post file not found: {path}")
        return path.read_text(encoding="utf-8")
    return args.text or ""


def _cmd_suggest(args: argparse.Namespace) -> int:
    config = Config.from_env()
    post_text = _read_post_text(args)
    result = matcher.suggest_with_config(
        config,
        client=args.client,
        keyword=args.keyword,
        post_text=post_text,
        current_url=args.url or None,
    )

    if not result.suggestions:
        print(f"No suggestions ({result.status}): {result.message}")
    else:
        print(result.message)
        print(f"{'PARA':>4}  {'CONF':>4}  {'ANCHOR':<24}  TARGET")
        for s in result.suggestions:
            print(
                f"{s.doc_paragraph_index:>4}  {s.confidence:>4.2f}  "
                f"{s.anchor_text[:24]:<24}  {s.target_url}"
            )

    SUGGESTIONS_PATH.write_text(
        json.dumps(
            [dataclasses.asdict(s) for s in result.suggestions], indent=2
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {SUGGESTIONS_PATH}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Serving the Internal Linking Assistant on http://{args.host}:{args.port}")
    uvicorn.run("app:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    commands = {
        "ingest": _cmd_ingest,
        "suggest": _cmd_suggest,
        "serve": _cmd_serve,
    }
    try:
        handler = commands.get(args.command)
        if handler is None:
            raise ValueError(f"Unknown command: {args.command}")
        return handler(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level guard for a CLI entry point
        log.exception("Command failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
