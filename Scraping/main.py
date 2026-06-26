"""Command-line entry point for the AniList full-catalog scraper."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import PAGE_SIZE, RAW_DATA_DIR
from src.scraper import AnimeScraper
from src.utils import file_size_mb, setup_logging


DEFAULT_OUTPUT = f"{RAW_DATA_DIR}/anime_full.csv"


def main() -> None:
    args = parse_args()
    setup_logging()

    output_path = Path(args.output)
    scraper = AnimeScraper()
    summary = scraper.scrape(
        output_path=output_path,
        page_limit=args.limit,
        per_page=args.per_page,
        start_page=args.start_page,
    )

    size_mb = file_size_mb(output_path)
    print(
        "Scrape summary:\n"
        f"  Total anime in CSV: {summary['total_rows']}\n"
        f"  New anime saved this run: {summary['new_rows']}\n"
        f"  Pages requested this run: {summary['pages_requested']}\n"
        f"  Output path: {output_path}\n"
        f"  File size: {size_mb:.2f} MB"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape popularity-sorted anime data from the AniList GraphQL API."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of pages to fetch. Defaults to no cap.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"CSV output path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=PAGE_SIZE,
        help=f"AniList Page query perPage value. Defaults to {PAGE_SIZE}.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="AniList popularity page to start from. Defaults to 1.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
