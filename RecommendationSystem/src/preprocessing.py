"""Text preprocessing for content-based recommendation."""

from __future__ import annotations

import html
import re

import pandas as pd


CONTENT_TEXT_VERSION = "english_romaji_v2"


def pipe_to_text(value: object) -> str:
    """Convert pipe-separated list fields into normal text."""
    return " ".join(part.strip() for part in str(value).split("|") if part.strip())


def clean_text(value: object) -> str:
    """Normalize text lightly while keeping useful anime terms."""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s'-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def repeat_text(text: str, times: int) -> str:
    return " ".join([text] * times)


def build_content_text(row: pd.Series) -> str:
    """Build weighted content text for one anime.

    Genres and tags are repeated because they are strong signals for anime
    similarity. Description is useful, but noisier, so it receives lower weight.
    """
    title = clean_text(
        " ".join(
            [
                str(row.get("title_romaji", "")),
                str(row.get("title_english", "")),
            ]
        )
    )
    genres = clean_text(pipe_to_text(row.get("genres", "")))
    tags = clean_text(pipe_to_text(row.get("tags", "")))
    studios = clean_text(pipe_to_text(row.get("studios", "")))
    description = clean_text(row.get("description", ""))
    fmt = clean_text(row.get("format", ""))
    source = clean_text(row.get("source", ""))

    return " ".join(
        [
            repeat_text(genres, 5),
            repeat_text(tags, 4),
            repeat_text(title, 2),
            repeat_text(studios, 1),
            repeat_text(fmt, 1),
            repeat_text(source, 1),
            description,
        ]
    )
