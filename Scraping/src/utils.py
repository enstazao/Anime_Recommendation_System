"""Shared utility functions for logging, CSV output, and table previews."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Iterable

from config import LOG_FILE


FIELDNAMES = [
    "id",
    "idMal",
    "title_romaji",
    "title_english",
    "title_native",
    "type",
    "format",
    "source",
    "episodes",
    "duration",
    "status",
    "season",
    "seasonYear",
    "start_date",
    "end_date",
    "averageScore",
    "meanScore",
    "popularity",
    "favourites",
    "genres",
    "tags",
    "studios",
    "description",
    "cover_image",
    "isAdult",
]


def setup_logging() -> None:
    """Configure logging to both console and a log file."""
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def load_existing_ids(output_path: str | Path) -> set[int]:
    """Return anime IDs already present in an existing CSV output file."""
    path = Path(output_path)
    if not path.exists():
        return set()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            int(row["id"])
            for row in reader
            if row.get("id") and str(row["id"]).isdigit()
        }


def save_to_csv(rows: list[dict], output_path: str | Path) -> None:
    """Incrementally append new rows to CSV output."""
    if not rows:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict]:
    """Load CSV rows for the console preview."""
    csv_path = Path(path)
    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def count_csv_rows(path: str | Path) -> int:
    """Count data rows in a CSV file without loading the full dataset."""
    csv_path = Path(path)
    if not csv_path.exists():
        return 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        row_count = sum(1 for _ in handle)
    return max(row_count - 1, 0)


def file_size_mb(path: str | Path) -> float:
    """Return a file size in megabytes."""
    csv_path = Path(path)
    if not csv_path.exists():
        return 0.0
    return os.path.getsize(csv_path) / (1024 * 1024)


def print_table(rows: list[dict], max_width: int = 28) -> None:
    """Print a compact table preview for sample inspection."""
    if not rows:
        print("No rows to display.")
        return

    columns = [
        "id",
        "idMal",
        "title_romaji",
        "title_english",
        "format",
        "episodes",
        "status",
        "seasonYear",
        "averageScore",
        "popularity",
        "genres",
        "studios",
        "isAdult",
    ]
    columns = [column for column in columns if any(column in row for row in rows)]
    rendered_rows = [
        [_truncate(str(row.get(column, "")), max_width) for column in columns]
        for row in rows
    ]
    widths = [
        min(max(len(column), *(len(row[index]) for row in rendered_rows)), max_width)
        for index, column in enumerate(columns)
    ]

    header = " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns))
    separator = "-+-".join("-" * width for width in widths)
    print(header)
    print(separator)
    for rendered in rendered_rows:
        print(
            " | ".join(
                rendered[index].ljust(widths[index])
                for index in range(len(columns))
            )
        )


def _truncate(value: str, max_width: int) -> str:
    if len(value) <= max_width:
        return value
    return value[: max_width - 3] + "..."
