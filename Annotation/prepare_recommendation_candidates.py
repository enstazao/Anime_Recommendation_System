"""Prepare candidate recommendation lists for GPT-OSS ranking."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "Scraping" / "data" / "raw" / "anime_full.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "recommendation_candidates.csv"
RANDOM_SEED = 42


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    anime = load_anime(args.input)
    candidates = build_recommendation_candidates(
        anime=anime,
        seed_count=args.seed_count,
        candidates_per_seed=args.candidates_per_seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output, index=False)

    print(f"Saved {len(candidates)} candidate rows to {output}")
    print(f"Seeds: {candidates['seed_id'].nunique()}")
    print(candidates["candidate_source"].value_counts().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ranked recommendation candidates.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--candidates-per-seed", type=int, default=40)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def load_anime(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"id": str, "idMal": str}).fillna("")
    df["popularity_num"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    df = df[
        df["genres"].str.strip().ne("")
        & df["description"].str.strip().ne("")
        & df["title_romaji"].str.strip().ne("")
    ].copy()
    df["title_key"] = df.apply(title_key, axis=1)
    df["genre_set"] = df["genres"].apply(split_pipe)
    df["tag_set"] = df["tags"].apply(split_pipe)
    return df.sort_values("popularity_num", ascending=False).reset_index(drop=True)


def build_recommendation_candidates(
    anime: pd.DataFrame,
    seed_count: int,
    candidates_per_seed: int,
) -> pd.DataFrame:
    seeds = anime.head(seed_count)
    search_pool = anime.head(10000).copy()
    rows: list[dict] = []

    for _, seed in seeds.iterrows():
        selected: list[tuple[pd.Series, str]] = []
        seen_ids = {str(seed["id"])}

        for partner in thematic_matches(seed, search_pool, limit=28):
            add_candidate(selected, seen_ids, partner, "thematic_overlap")

        for partner in direct_franchise(seed, anime, limit=5):
            add_candidate(selected, seen_ids, partner, "same_franchise_candidate")

        for partner in partial_matches(seed, search_pool, limit=12):
            add_candidate(selected, seen_ids, partner, "partial_overlap")

        for partner in popular_fillers(seed, search_pool, limit=12):
            add_candidate(selected, seen_ids, partner, "popular_fallback")

        for partner, source in selected[:candidates_per_seed]:
            rows.append(format_candidate(seed, partner, source))

    return pd.DataFrame(rows)


def direct_franchise(seed: pd.Series, anime: pd.DataFrame, limit: int) -> list[pd.Series]:
    if not useful_franchise_key(seed["title_key"]):
        return []
    matches = anime[
        (anime["id"].ne(seed["id"]))
        & (anime["title_key"].eq(seed["title_key"]))
    ].sort_values("popularity_num", ascending=False)
    return [row for _, row in matches.head(limit).iterrows()]


def thematic_matches(seed: pd.Series, pool: pd.DataFrame, limit: int) -> list[pd.Series]:
    ranked: list[tuple[int, int, pd.Series]] = []
    for _, partner in pool.iterrows():
        if seed["id"] == partner["id"] or seed["title_key"] == partner["title_key"]:
            continue
        shared_genres = len(seed["genre_set"] & partner["genre_set"])
        shared_tags = len(seed["tag_set"] & partner["tag_set"])
        if shared_genres >= 2 and shared_tags >= 4:
            score = (shared_genres * 12) + shared_tags
            ranked.append((score, int(partner["popularity_num"]), partner))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [row for _, _, row in ranked[:limit]]


def partial_matches(seed: pd.Series, pool: pd.DataFrame, limit: int) -> list[pd.Series]:
    ranked: list[tuple[int, int, pd.Series]] = []
    for _, partner in pool.iterrows():
        if seed["id"] == partner["id"] or seed["title_key"] == partner["title_key"]:
            continue
        shared_genres = len(seed["genre_set"] & partner["genre_set"])
        shared_tags = len(seed["tag_set"] & partner["tag_set"])
        if shared_genres >= 1 and shared_tags >= 1:
            score = (shared_genres * 8) + shared_tags
            ranked.append((score, int(partner["popularity_num"]), partner))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [row for _, _, row in ranked[:limit]]


def popular_fillers(seed: pd.Series, pool: pd.DataFrame, limit: int) -> list[pd.Series]:
    rows = []
    for _, partner in pool.iterrows():
        if seed["id"] != partner["id"] and seed["title_key"] != partner["title_key"]:
            rows.append(partner)
        if len(rows) >= limit:
            break
    return rows


def add_candidate(
    selected: list[tuple[pd.Series, str]],
    seen_ids: set[str],
    partner: pd.Series,
    source: str,
) -> None:
    partner_id = str(partner["id"])
    if partner_id in seen_ids:
        return
    seen_ids.add(partner_id)
    selected.append((partner, source))


def format_candidate(seed: pd.Series, partner: pd.Series, source: str) -> dict:
    return {
        "seed_id": seed["id"],
        "seed_title": preferred_title(seed),
        "seed_genres": seed["genres"],
        "seed_tags": seed["tags"],
        "seed_description": trim_description(seed["description"], limit=450),
        "candidate_id": partner["id"],
        "candidate_title": preferred_title(partner),
        "candidate_genres": partner["genres"],
        "candidate_tags": partner["tags"],
        "candidate_description": trim_description(partner["description"], limit=450),
        "candidate_source": source,
    }


def split_pipe(value: str) -> set[str]:
    return {part.strip().lower() for part in str(value).split("|") if part.strip()}


def preferred_title(row: pd.Series) -> str:
    return row.get("title_english") or row.get("title_romaji") or row.get("title_native") or ""


def trim_description(value: str, limit: int = 300) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def title_key(row: pd.Series) -> str:
    title = preferred_title(row).lower()
    title = re.sub(r"\([^)]*\)", " ", title)
    title = re.sub(r"\b(season|part|movie|ova|ona|special|final|the|episode)\b", " ", title)
    title = re.sub(r"\b[0-9ivx]+\b", " ", title)
    title = re.sub(r"[^a-z0-9]+", " ", title)
    words = [word for word in title.split() if len(word) > 2]
    return " ".join(words[:4])


def useful_franchise_key(key: str) -> bool:
    words = key.split()
    generic = {
        "love",
        "ghost",
        "hero",
        "king",
        "queen",
        "girl",
        "boys",
        "dream",
        "story",
        "special",
    }
    if key in generic:
        return False
    return len(words) >= 2 or key in {"naruto", "bleach", "overlord", "kingdom", "gintama"}


if __name__ == "__main__":
    main()
