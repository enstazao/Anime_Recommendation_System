"""Low-level AniList GraphQL API client."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from config import (
    API_URL,
    DEFAULT_RETRY_AFTER_SECONDS,
    HEADERS,
    MAX_429_RETRIES,
    RATE_LIMIT_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
)


class AniListApiClient:
    """POST GraphQL queries while respecting AniList rate limits."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._last_request_at = 0.0
        self.logger = logging.getLogger(__name__)

    def execute(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """Execute a GraphQL request and return the decoded JSON response."""
        payload = {"query": query, "variables": variables}

        for attempt in range(1, MAX_429_RETRIES + 1):
            self._rate_limit()
            self.logger.info("Requesting AniList page with variables: %s", variables)

            response = self.session.post(
                API_URL,
                data=json.dumps(payload),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            self._last_request_at = time.monotonic()

            if response.status_code == 429:
                retry_after = _retry_after_seconds(response)
                self.logger.warning(
                    "AniList returned HTTP 429; retrying in %ss "
                    "(attempt %s/%s)",
                    retry_after,
                    attempt,
                    MAX_429_RETRIES,
                )
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            data = response.json()
            if data.get("errors"):
                raise RuntimeError(f"AniList GraphQL errors: {data['errors']}")
            return data

        raise TimeoutError(
            f"AniList kept returning HTTP 429 after {MAX_429_RETRIES} retries."
        )

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = RATE_LIMIT_SECONDS - elapsed
        if remaining > 0:
            time.sleep(remaining)


def _retry_after_seconds(response: requests.Response) -> int:
    retry_after = response.headers.get("Retry-After")
    if retry_after and retry_after.isdigit():
        return int(retry_after)
    return DEFAULT_RETRY_AFTER_SECONDS
