"""TF-IDF content-based anime recommender."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .preprocessing import build_content_text


@dataclass
class Recommendation:
    id: str
    title: str
    title_english: str
    score: float
    genres: str
    tags: str
    cover_image: str


class TfidfAnimeRecommender:
    """Fit TF-IDF vectors and return nearest anime by cosine similarity."""

    def __init__(
        self,
        anime: pd.DataFrame,
        max_features: int = 50000,
        min_df: int = 2,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.anime = anime.copy().reset_index(drop=True)
        self.id_to_index = {str(row_id): idx for idx, row_id in enumerate(self.anime["id"].astype(str))}
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            min_df=min_df,
            ngram_range=ngram_range,
            sublinear_tf=True,
            norm="l2",
        )
        content = self.anime.apply(build_content_text, axis=1)
        self.matrix = self.vectorizer.fit_transform(content)

    def recommend_by_id(self, anime_id: str, top_k: int = 10) -> list[Recommendation]:
        """Return top-k similar anime for a given AniList id."""
        anime_id = str(anime_id)
        if anime_id not in self.id_to_index:
            raise KeyError(f"Anime id not found: {anime_id}")

        source_index = self.id_to_index[anime_id]
        scores = linear_kernel(self.matrix[source_index], self.matrix).ravel()
        ranked_indices = scores.argsort()[::-1]

        recommendations: list[Recommendation] = []
        for idx in ranked_indices:
            if idx == source_index:
                continue
            row = self.anime.iloc[idx]
            recommendations.append(
                Recommendation(
                    id=str(row["id"]),
                    title=str(row["title_romaji"]),
                    title_english=str(row.get("title_english", "")),
                    score=float(scores[idx]),
                    genres=str(row.get("genres", "")),
                    tags=str(row.get("tags", "")),
                    cover_image=str(row.get("cover_image", "")),
                )
            )
            if len(recommendations) >= top_k:
                break
        return recommendations

    def search_titles(self, query: str, limit: int = 10) -> list[dict]:
        """Simple case-insensitive title search for the web app."""
        query = query.strip().lower()
        if not query:
            return []

        title_blob = (
            self.anime["title_romaji"].astype(str)
            + " "
            + self.anime["title_english"].astype(str)
        ).str.lower()
        matches = self.anime[title_blob.str.contains(query, regex=False)].head(limit)
        return [
            {
                "id": str(row["id"]),
                "title": str(row["title_romaji"]),
                "title_english": str(row.get("title_english", "")),
                "cover_image": str(row.get("cover_image", "")),
            }
            for _, row in matches.iterrows()
        ]

    def best_title_match(self, query: str) -> str:
        """Return the id of the first title search match."""
        matches = self.search_titles(query, limit=1)
        if not matches:
            raise KeyError(f"No anime title matched: {query}")
        return matches[0]["id"]
