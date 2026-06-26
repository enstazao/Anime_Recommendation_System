"""High-level AniList scraping orchestration."""

from __future__ import annotations

import logging
from pathlib import Path

from config import PAGE_SIZE
from src.api_client import AniListApiClient
from src.parser import parse_anime_page
from src.queries import ANIME_PAGE_BY_POPULARITY_QUERY, ANIME_PAGE_QUERY
from src.utils import count_csv_rows, load_existing_ids, save_to_csv


class AnimeScraper:
    """Fetch anime popularity pages and save progress after every page."""

    MAX_OFFSET_PAGES = 100

    def __init__(self, client: AniListApiClient | None = None) -> None:
        self.client = client or AniListApiClient()
        self.logger = logging.getLogger(__name__)

    def scrape(
        self,
        output_path: str | Path,
        page_limit: int | None = None,
        per_page: int = PAGE_SIZE,
        start_page: int = 1,
    ) -> dict[str, int | Path]:
        """Fetch popularity-sorted anime until page cap or pageInfo exhaustion."""
        output = Path(output_path)
        existing_ids = load_existing_ids(output)
        page = start_page
        popularity_ceiling: int | None = None
        pages_requested = 0
        new_rows_total = 0

        self.logger.info(
            "Starting AniList scrape: page_limit=%s, per_page=%s, existing_ids=%s",
            page_limit,
            per_page,
            len(existing_ids),
        )

        while page_limit is None or pages_requested < page_limit:
            variables = {
                "page": page,
                "perPage": per_page,
            }
            query = ANIME_PAGE_QUERY
            if popularity_ceiling is not None:
                variables["popularityLesser"] = popularity_ceiling
                query = ANIME_PAGE_BY_POPULARITY_QUERY

            data = self.client.execute(query, variables)
            rows, page_info = parse_anime_page(data)
            pages_requested += 1
            new_rows = [
                row
                for row in rows
                if row.get("id") and int(row["id"]) not in existing_ids
            ]

            if new_rows:
                save_to_csv(new_rows, output)
                existing_ids.update(int(row["id"]) for row in new_rows)
                new_rows_total += len(new_rows)
            else:
                self.logger.info("Page %s had no new anime IDs to save.", page)

            self.logger.info(
                "Progress: page=%s, page_rows=%s, new_page_rows=%s, "
                "total_saved_rows=%s",
                page,
                len(rows),
                len(new_rows),
                len(existing_ids),
            )

            if not page_info.get("hasNextPage"):
                self.logger.info("AniList pageInfo.hasNextPage is false; stopping.")
                break

            if page >= self.MAX_OFFSET_PAGES:
                next_ceiling = _next_popularity_ceiling(rows)
                if next_ceiling is None:
                    self.logger.warning(
                        "Reached page-depth cap but could not determine a "
                        "popularity cursor; stopping."
                    )
                    break
                if next_ceiling == popularity_ceiling:
                    raise RuntimeError(
                        "AniList pagination made no progress at popularity "
                        f"ceiling {popularity_ceiling}; cannot continue safely."
                    )
                self.logger.info(
                    "Reached AniList 5000-entry page-depth window; continuing "
                    "with popularity_lesser=%s",
                    next_ceiling,
                )
                popularity_ceiling = next_ceiling
                page = 1
                continue

            page += 1

        return {
            "new_rows": new_rows_total,
            "total_rows": count_csv_rows(output),
            "pages_requested": pages_requested,
            "last_page": page,
            "output_path": output,
        }


def _next_popularity_ceiling(rows: list[dict]) -> int | None:
    popularity_values = [
        int(row["popularity"])
        for row in rows
        if row.get("popularity") not in (None, "")
    ]
    if not popularity_values:
        return None
    return min(popularity_values) + 1
