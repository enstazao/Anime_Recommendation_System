"""Evaluate recommenders with graded relevance using NDCG@K."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RecommendationSystem import config
from RecommendationSystem.src.bm25f_recommender import BM25FAnimeRecommender
from RecommendationSystem.src.data_loader import load_anime, load_test_set
from RecommendationSystem.src.embedding_recommender import EmbeddingAnimeRecommender
from RecommendationSystem.src.hybrid_recommender import HybridAnimeRecommender
from RecommendationSystem.src.tfidf_recommender import TfidfAnimeRecommender


KS = (3, 5, 10, 15)
REPORT_PATH = PROJECT_ROOT / "reports" / "ndcg_comparison.md"


def gain(label: int) -> float:
    """Convert graded relevance label to gain."""
    return (2**label) - 1


def dcg(labels: list[int], k: int) -> float:
    """Discounted cumulative gain for labels ordered by model rank."""
    total = 0.0
    for rank, label in enumerate(labels[:k], start=1):
        total += gain(int(label)) / math.log2(rank + 1)
    return total


def evaluate_ndcg_at_k(recommender, test_df: pd.DataFrame, k: int) -> float:
    """Compute macro NDCG@K over seed anime."""
    available_ids = set(recommender.anime["id"].astype(str))
    scores: list[float] = []

    for seed_id, group in test_df.groupby("seed_id"):
        seed_id = str(seed_id)
        if seed_id not in available_ids:
            continue

        relevance_by_id = {
            str(row["recommended_id"]): int(row["similarity_label"])
            for _, row in group.iterrows()
        }
        ideal_labels = sorted(relevance_by_id.values(), reverse=True)
        ideal_dcg = dcg(ideal_labels, k)
        if ideal_dcg <= 0:
            continue

        predictions = recommender.recommend_by_id(seed_id, top_k=k)
        predicted_labels = [relevance_by_id.get(item.id, 0) for item in predictions]
        scores.append(dcg(predicted_labels, k) / ideal_dcg)

    return sum(scores) / len(scores) if scores else 0.0


def build_recommenders(anime: pd.DataFrame):
    tfidf = TfidfAnimeRecommender(
        anime=anime,
        max_features=config.TFIDF_MAX_FEATURES,
        min_df=config.TFIDF_MIN_DF,
        ngram_range=config.TFIDF_NGRAM_RANGE,
    )
    bm25f = BM25FAnimeRecommender(
        anime=anime,
        field_weights=config.BM25F_FIELD_WEIGHTS,
        field_b=config.BM25F_FIELD_B,
        k1=config.BM25F_K1,
    )
    minilm_config = config.EMBEDDING_MODELS["minilm"]
    bge_config = config.EMBEDDING_MODELS["bge"]
    embeddings_minilm = EmbeddingAnimeRecommender(
        anime=anime,
        model_name=minilm_config["model_name"],
        cache_path=minilm_config["cache_path"],
        ids_path=minilm_config["ids_path"],
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )
    embeddings_bge = EmbeddingAnimeRecommender(
        anime=anime,
        model_name=bge_config["model_name"],
        cache_path=bge_config["cache_path"],
        ids_path=bge_config["ids_path"],
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )
    hybrid = HybridAnimeRecommender(
        anime=anime,
        tfidf=tfidf,
        bm25f=bm25f,
        embeddings=embeddings_bge,
        weights=config.HYBRID_WEIGHTS,
    )
    return {
        "TF-IDF": tfidf,
        "BM25F": bm25f,
        "Embeddings MiniLM": embeddings_minilm,
        "Embeddings BGE": embeddings_bge,
        "Hybrid BGE": hybrid,
    }


def write_report(results: dict[str, dict[int, float]], rows: int, seeds: int) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NDCG Model Comparison",
        "",
        "Evaluation test set: `Annotation/data/final_cleaned_annotated_set.csv`",
        "",
        f"Test rows: {rows}",
        f"Seed anime/searches: {seeds}",
        "",
        "NDCG uses graded relevance directly: label 2 > label 1 > label 0.",
        "",
        "| Model | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@15 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_name, scores in results.items():
        lines.append(
            f"| {model_name} | "
            f"{scores[3]:.4f} | {scores[5]:.4f} | "
            f"{scores[10]:.4f} | {scores[15]:.4f} |"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    anime = load_anime(config.ANIME_DATA_PATH)
    test_df = load_test_set(config.ANNOTATED_TEST_PATH)
    recommenders = build_recommenders(anime)

    results: dict[str, dict[int, float]] = {}
    for model_name, recommender in recommenders.items():
        results[model_name] = {
            k: evaluate_ndcg_at_k(recommender, test_df, k) for k in KS
        }

    write_report(results, rows=len(test_df), seeds=test_df["seed_id"].nunique())
    print("Model,NDCG@3,NDCG@5,NDCG@10,NDCG@15")
    for model_name, scores in results.items():
        print(
            f"{model_name},"
            f"{scores[3]:.4f},{scores[5]:.4f},"
            f"{scores[10]:.4f},{scores[15]:.4f}"
        )
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
