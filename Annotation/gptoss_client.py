"""Shared GPT-OSS client helpers for annotation scripts."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests


DEFAULT_BASE_URL = os.getenv("GPTOSS_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
DEFAULT_MODEL = os.getenv("GPTOSS_MODEL", "openai/gpt-oss-120b")


@dataclass
class GPTOSSClient:
    """Small OpenAI-compatible chat client used by the annotation pipeline."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout: int = 180
    retries: int = 5
    retry_delay: int = 8

    def call(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code == 429:
                    wait_seconds = int(response.headers.get("Retry-After", self.retry_delay))
                    time.sleep(wait_seconds)
                    continue
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - retry wrapper should catch API/network errors.
                last_error = exc
                if attempt == self.retries:
                    break
                time.sleep(self.retry_delay * attempt)

        raise RuntimeError(f"GPT-OSS request failed after {self.retries} attempts: {last_error}")


def resolve_api_key(api_key: str | None = None) -> str:
    """Resolve the GPT-OSS API key from CLI argument or environment."""
    key = api_key or os.getenv("GPTOSS_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing GPT-OSS API key. Pass --api-key or set GPTOSS_API_KEY/GROQ_API_KEY."
        )
    return key


def extract_json(raw: str) -> dict[str, Any]:
    """Parse a JSON object from a model response."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def pipe_to_commas(value: Any) -> str:
    """Convert pipe-separated metadata into readable comma-separated text."""
    parts = [item.strip() for item in str(value or "").split("|") if item.strip()]
    return ", ".join(parts)


def top_items(value: Any, limit: int = 16) -> str:
    """Return the first metadata items from a pipe-separated list."""
    parts = [item.strip() for item in str(value or "").split("|") if item.strip()]
    return ", ".join(parts[:limit])


def shorten(value: Any, limit: int = 500) -> str:
    """Keep prompts compact while preserving enough description context."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."
