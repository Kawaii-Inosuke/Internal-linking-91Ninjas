"""Command-line interface for the Internal Linking Assistant.

M1 implements the ``ingest`` subcommand; ``suggest`` arrives in M2 (TRD §9/§11).

    python cli.py ingest --client gokwik --file "Gokwik content.xlsx"
"""
from __future__ import annotations

import argparse
import logging
import sys

from linker.config import Config, ConfigError
from linker.ingest import ingest_file

log = logging.getLogger("linker.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py", description="Internal Linking Assistant"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest a blog corpus (xlsx/CSV) for a client")
    ingest.add_argument("--client", required=True, help="Client name, e.g. gokwik")
    ingest.add_argument("--file", required=True, help="Path to the corpus .xlsx/.csv")
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


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "ingest":
            return _cmd_ingest(args)
        raise ValueError(f"Unknown command: {args.command}")
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
