"""Hybrid anime recommender combining retrieval and metadata signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

from .bm25f_recommender import BM25FAnimeRecommender
from .embedding_recommender import EmbeddingAnimeRecommender
from .tfidf_recommender import TfidfAnimeRecommender


@dataclass
class HybridRecommendation:
    id: str
    title: str
    title_english: str
    score: float
    genres: str
    tags: str
    cover_image: str


class HybridAnimeRecommender:
    """Blend BM25F, TF-IDF, embeddings, and explicit metadata overlap."""

    def __init__(
        self,
        anime: pd.DataFrame,
        tfidf: TfidfAnimeRecommender,
        bm25f: BM25FAnimeRecommender,
        embeddings: EmbeddingAnimeRecommender,
        weights: dict[str, float],
    ) -> None:
        self.anime = anime.copy().reset_index(drop=True)
        self.tfidf = tfidf
        self.bm25f = bm25f
        self.embeddings = embeddings
        self.weights = weights
        self.id_to_index = {str(row_id): idx for idx, row_id in enumerate(self.anime["id"].astype(str))}
        self.genre_sets = self.anime["genres"].apply(split_pipe).tolist()
        self.tag_sets = self.anime["tags"].apply(split_pipe).tolist()

    def recommend_by_id(self, anime_id: str, top_k: int = 10) -> list[HybridRecommendation]:
        anime_id = str(anime_id)
        if anime_id not in self.id_to_index:
            raise KeyError(f"Anime id not found: {anime_id}")

        source_index = self.id_to_index[anime_id]
        bm25f_scores = self._bm25f_scores(source_index)
        tfidf_scores = linear_kernel(self.tfidf.matrix[source_index], self.tfidf.matrix).ravel()
        embedding_scores = self.embeddings.embeddings @ self.embeddings.embeddings[source_index]
        genre_scores = overlap_scores(self.genre_sets[source_index], self.genre_sets)
        tag_scores = overlap_scores(self.tag_sets[source_index], self.tag_sets)

        final_scores = (
            self.weights["bm25f"] * normalize(bm25f_scores)
            + self.weights["tfidf"] * normalize(tfidf_scores)
            + self.weights["embedding"] * normalize(embedding_scores)
            + self.weights["tag_overlap"] * tag_scores
            + self.weights["genre_overlap"] * genre_scores
        )
        final_scores[source_index] = -np.inf
        ranked_indices = np.argsort(final_scores)[::-1]

        recommendations: list[HybridRecommendation] = []
        for idx in ranked_indices:
            if not np.isfinite(final_scores[idx]):
                continue
            row = self.anime.iloc[idx]
            recommendations.append(
                HybridRecommendation(
                    id=str(row["id"]),
                    title=str(row["title_romaji"]),
                    title_english=str(row.get("title_english", "")),
                    score=float(final_scores[idx]),
                    genres=str(row.get("genres", "")),
                    tags=str(row.get("tags", "")),
                    cover_image=str(row.get("cover_image", "")),
                )
            )
            if len(recommendations) >= top_k:
                break
        return recommendations

    def _bm25f_scores(self, source_index: int) -> np.ndarray:
        query_terms = self.bm25f._query_terms(self.anime.iloc[source_index])
        return self.bm25f._score_terms(query_terms)


def split_pipe(value: object) -> set[str]:
    return {part.strip().lower() for part in str(value).split("|") if part.strip()}


def overlap_scores(source: set[str], candidates: list[set[str]]) -> np.ndarray:
    scores = np.zeros(len(candidates), dtype=np.float32)
    if not source:
        return scores
    for idx, candidate in enumerate(candidates):
        union = source | candidate
        if union:
            scores[idx] = len(source & candidate) / len(union)
    return scores


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    min_value = float(finite.min())
    max_value = float(finite.max())
    if max_value <= min_value:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)
