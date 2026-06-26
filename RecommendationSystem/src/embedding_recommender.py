"""Sentence-embedding content-based anime recommender."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import CONTENT_TEXT_VERSION, build_content_text


@dataclass
class EmbeddingRecommendation:
    id: str
    title: str
    title_english: str
    score: float
    genres: str
    tags: str
    cover_image: str


class EmbeddingAnimeRecommender:
    """Recommend anime by cosine similarity over sentence embeddings."""

    def __init__(
        self,
        anime: pd.DataFrame,
        model_name: str,
        cache_path: str | Path,
        ids_path: str | Path,
        batch_size: int = 64,
    ) -> None:
        self.anime = anime.copy().reset_index(drop=True)
        self.model_name = model_name
        self.cache_path = Path(cache_path)
        self.ids_path = Path(ids_path)
        self.batch_size = batch_size
        self.id_to_index = {str(row_id): idx for idx, row_id in enumerate(self.anime["id"].astype(str))}
        self.embeddings = self._load_or_build_embeddings()

    def recommend_by_id(self, anime_id: str, top_k: int = 10) -> list[EmbeddingRecommendation]:
        anime_id = str(anime_id)
        if anime_id not in self.id_to_index:
            raise KeyError(f"Anime id not found: {anime_id}")

        source_index = self.id_to_index[anime_id]
        scores = self.embeddings @ self.embeddings[source_index]
        ranked_indices = np.argsort(scores)[::-1]

        recommendations: list[EmbeddingRecommendation] = []
        for idx in ranked_indices:
            if idx == source_index:
                continue
            row = self.anime.iloc[idx]
            recommendations.append(
                EmbeddingRecommendation(
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

    def _load_or_build_embeddings(self) -> np.ndarray:
        if self.cache_path.exists() and self.ids_path.exists():
            cache_metadata = pd.read_csv(self.ids_path, dtype=str)
            cached_version = (
                cache_metadata["content_text_version"].iloc[0]
                if "content_text_version" in cache_metadata.columns and not cache_metadata.empty
                else ""
            )
            cached_model_name = (
                cache_metadata["embedding_model_name"].iloc[0]
                if "embedding_model_name" in cache_metadata.columns and not cache_metadata.empty
                else ""
            )
            cached_ids = cache_metadata["id"].tolist()
            current_ids = self.anime["id"].astype(str).tolist()
            if (
                cached_ids == current_ids
                and cached_version == CONTENT_TEXT_VERSION
                and cached_model_name == self.model_name
            ):
                return np.load(self.cache_path)

        embeddings = self._build_embeddings()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_path, embeddings)
        pd.DataFrame(
            {
                "id": self.anime["id"].astype(str),
                "content_text_version": CONTENT_TEXT_VERSION,
                "embedding_model_name": self.model_name,
            }
        ).to_csv(self.ids_path, index=False)
        return embeddings

    def _build_embeddings(self) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the embedding recommender. "
                "Install it with: pip install -r RecommendationSystem/requirements.txt"
            ) from exc

        model = SentenceTransformer(self.model_name)
        content = self.anime.apply(build_content_text, axis=1).tolist()
        embeddings = model.encode(
            content,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
