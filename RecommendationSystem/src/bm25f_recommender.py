"""BM25F field-weighted content recommender for anime metadata."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .preprocessing import clean_text, pipe_to_text


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*")


@dataclass
class BM25FRecommendation:
    id: str
    title: str
    title_english: str
    score: float
    genres: str
    tags: str
    cover_image: str


class BM25FAnimeRecommender:
    """Rank anime using BM25F over structured metadata fields."""

    def __init__(
        self,
        anime: pd.DataFrame,
        field_weights: dict[str, float],
        field_b: dict[str, float],
        k1: float = 1.5,
    ) -> None:
        self.anime = anime.copy().reset_index(drop=True)
        self.field_weights = field_weights
        self.field_b = field_b
        self.k1 = k1
        self.id_to_index = {str(row_id): idx for idx, row_id in enumerate(self.anime["id"].astype(str))}
        self.n_docs = len(self.anime)
        self.fields = list(field_weights)
        self.avg_lengths: dict[str, float] = {}
        self.field_lengths: dict[str, np.ndarray] = {}
        self.postings: dict[str, dict[str, list[tuple[int, int]]]] = {}
        self.idf: dict[str, float] = {}
        self._build_index()

    def recommend_by_id(self, anime_id: str, top_k: int = 10) -> list[BM25FRecommendation]:
        anime_id = str(anime_id)
        if anime_id not in self.id_to_index:
            raise KeyError(f"Anime id not found: {anime_id}")

        source_index = self.id_to_index[anime_id]
        query_terms = self._query_terms(self.anime.iloc[source_index])
        scores = self._score_terms(query_terms)
        scores[source_index] = -np.inf
        ranked_indices = np.argsort(scores)[::-1]

        recommendations: list[BM25FRecommendation] = []
        for idx in ranked_indices:
            if not np.isfinite(scores[idx]) or scores[idx] <= 0:
                continue
            row = self.anime.iloc[idx]
            recommendations.append(
                BM25FRecommendation(
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

    def _build_index(self) -> None:
        doc_frequency: Counter[str] = Counter()
        field_postings: dict[str, dict[str, list[tuple[int, int]]]] = {
            field: defaultdict(list) for field in self.fields
        }

        for field in self.fields:
            lengths = np.zeros(self.n_docs, dtype=np.float32)
            for idx, row in self.anime.iterrows():
                tokens = self._field_tokens(row, field)
                lengths[idx] = len(tokens)
                counts = Counter(tokens)
                for term, tf in counts.items():
                    field_postings[field][term].append((idx, tf))
            self.field_lengths[field] = lengths
            self.avg_lengths[field] = float(lengths.mean()) if float(lengths.mean()) > 0 else 1.0

        seen_by_term: dict[str, set[int]] = defaultdict(set)
        for field in self.fields:
            for term, postings in field_postings[field].items():
                for doc_idx, _ in postings:
                    seen_by_term[term].add(doc_idx)
        for term, doc_ids in seen_by_term.items():
            doc_frequency[term] = len(doc_ids)

        self.postings = {field: dict(values) for field, values in field_postings.items()}
        self.idf = {
            term: math.log(1 + ((self.n_docs - df + 0.5) / (df + 0.5)))
            for term, df in doc_frequency.items()
        }

    def _score_terms(self, query_terms: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            weighted_tf = np.zeros(self.n_docs, dtype=np.float32)
            for field in self.fields:
                postings = self.postings[field].get(term)
                if not postings:
                    continue
                weight = self.field_weights[field]
                b_value = self.field_b[field]
                avg_len = self.avg_lengths[field]
                lengths = self.field_lengths[field]
                for doc_idx, tf in postings:
                    norm = 1 - b_value + b_value * (lengths[doc_idx] / avg_len)
                    weighted_tf[doc_idx] += weight * (tf / norm)
            term_scores = idf * ((self.k1 + 1) * weighted_tf) / (self.k1 + weighted_tf + 1e-9)
            scores += term_scores
        return scores

    def _query_terms(self, row: pd.Series) -> list[str]:
        weighted_parts = [
            ("title", 2),
            ("genres", 5),
            ("tags", 4),
            ("description", 1),
            ("studios", 1),
        ]
        terms: list[str] = []
        for field, repeat in weighted_parts:
            field_terms = self._field_tokens(row, field)
            if field == "description":
                field_terms = field_terms[:80]
            for _ in range(repeat):
                terms.extend(field_terms)
        counts = Counter(terms)
        return [term for term, _ in counts.most_common(140)]

    def _field_tokens(self, row: pd.Series, field: str) -> list[str]:
        if field == "title":
            value = " ".join(
                [
                    str(row.get("title_romaji", "")),
                    str(row.get("title_english", "")),
                ]
            )
        elif field in {"genres", "tags", "studios"}:
            value = pipe_to_text(row.get(field, ""))
        else:
            value = row.get(field, "")
        return tokenize(value)


def tokenize(value: object) -> list[str]:
    return TOKEN_RE.findall(clean_text(value))
