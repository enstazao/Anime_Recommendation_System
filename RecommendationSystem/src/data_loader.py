"""Data loading helpers for the recommender."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_anime(path: str | Path) -> pd.DataFrame:
    """Load anime metadata and keep rows usable for content recommendation."""
    df = pd.read_csv(path, dtype={"id": str, "idMal": str}).fillna("")
    df = df[df["id"].str.strip().ne("") & df["title_romaji"].str.strip().ne("")].copy()
    df["popularity_num"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    df["average_score_num"] = pd.to_numeric(df["averageScore"], errors="coerce").fillna(0)
    return df.reset_index(drop=True)


def load_test_set(path: str | Path) -> pd.DataFrame:
    """Load GPT-OSS annotated recommendation test set."""
    df = pd.read_csv(path, dtype={"seed_id": str, "recommended_id": str}).fillna("")
    df["similarity_label"] = pd.to_numeric(df["similarity_label"], errors="coerce").fillna(0).astype(int)
    return df
