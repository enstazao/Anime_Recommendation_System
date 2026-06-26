"""Flatten AniList GraphQL JSON media objects into CSV-ready dictionaries."""

from __future__ import annotations

import html
import re
from typing import Any


def parse_anime_page(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return flattened anime rows and pageInfo from an AniList response."""
    page = data.get("data", {}).get("Page", {})
    media_items = page.get("media", []) or []
    page_info = page.get("pageInfo", {}) or {}
    rows = [flatten_anime(item) for item in media_items]
    return rows, page_info


def flatten_anime(item: dict[str, Any]) -> dict[str, Any]:
    """Flatten one AniList media item into the requested CSV schema."""
    title = item.get("title") or {}
    start_date = item.get("startDate") or {}
    end_date = item.get("endDate") or {}
    cover_image = item.get("coverImage") or {}
    studios = item.get("studios") or {}

    return {
        "id": item.get("id"),
        "idMal": item.get("idMal"),
        "title_romaji": title.get("romaji"),
        "title_english": title.get("english"),
        "title_native": title.get("native"),
        "type": item.get("type"),
        "format": item.get("format"),
        "source": item.get("source"),
        "episodes": item.get("episodes"),
        "duration": item.get("duration"),
        "status": item.get("status"),
        "season": item.get("season"),
        "seasonYear": item.get("seasonYear"),
        "start_date": _format_date(start_date),
        "end_date": _format_date(end_date),
        "averageScore": item.get("averageScore"),
        "meanScore": item.get("meanScore"),
        "popularity": item.get("popularity"),
        "favourites": item.get("favourites"),
        "genres": _join_list(item.get("genres") or []),
        "tags": _join_list(tag.get("name") for tag in item.get("tags") or []),
        "studios": _join_list(
            studio.get("name") for studio in studios.get("nodes") or []
        ),
        "description": _clean_description(item.get("description")),
        "cover_image": cover_image.get("large"),
        "isAdult": item.get("isAdult"),
    }


def _format_date(date_parts: dict[str, Any]) -> str:
    year = date_parts.get("year")
    month = date_parts.get("month")
    day = date_parts.get("day")

    if not year:
        return ""
    if month and day:
        return f"{year:04d}-{month:02d}-{day:02d}"
    if month:
        return f"{year:04d}-{month:02d}"
    return str(year)


def _join_list(values: Any) -> str:
    return "|".join(str(value) for value in values if value)


def _clean_description(description: str | None) -> str:
    if not description:
        return ""

    text = html.unescape(description)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

