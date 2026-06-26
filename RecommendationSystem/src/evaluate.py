"""Evaluate TF-IDF recommender using Precision@10 and Recall@10."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RecommendationSystem import config
from RecommendationSystem.src.data_loader import load_anime, load_test_set
from RecommendationSystem.src.tfidf_recommender import TfidfAnimeRecommender


def evaluate_precision_recall(
    recommender: TfidfAnimeRecommender,
    test_df: pd.DataFrame,
    top_k: int,
    relevant_label_threshold: int,
) -> dict[str, float]:
    """Compute macro Precision@K and Recall@K over seed anime."""
    available_ids = set(recommender.anime["id"].astype(str))
    scores = []

    for seed_id, group in test_df.groupby("seed_id"):
        seed_id = str(seed_id)
        if seed_id not in available_ids:
            continue

        relevant_ids = set(
            group[group["similarity_label"] >= relevant_label_threshold]["recommended_id"].astype(str)
        )
        if not relevant_ids:
            continue

        predictions = recommender.recommend_by_id(seed_id, top_k=top_k)
        predicted_ids = {item.id for item in predictions}
        hits = len(predicted_ids & relevant_ids)

        scores.append(
            {
                "precision": hits / top_k,
                "recall": hits / len(relevant_ids),
            }
        )

    if not scores:
        return {"precision_at_10": 0.0, "recall_at_10": 0.0, "evaluated_seeds": 0}

    precision = sum(item["precision"] for item in scores) / len(scores)
    recall = sum(item["recall"] for item in scores) / len(scores)
    return {
        "precision_at_10": precision,
        "recall_at_10": recall,
        "evaluated_seeds": len(scores),
    }


def write_report(metrics: dict[str, float]) -> None:
    config.REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# TF-IDF Test Report

Evaluation test set:

`Annotation/data/final_cleaned_annotated_set.csv`

Relevant recommendations are rows with `similarity_label >= {config.RELEVANT_LABEL_THRESHOLD}`.

| Metric | Value |
|---|---:|
| Evaluated seeds | {int(metrics["evaluated_seeds"])} |
| Precision@10 | {metrics["precision_at_10"]:.4f} |
| Recall@10 | {metrics["recall_at_10"]:.4f} |
"""
    config.REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    anime = load_anime(config.ANIME_DATA_PATH)
    test_df = load_test_set(config.ANNOTATED_TEST_PATH)
    recommender = TfidfAnimeRecommender(
        anime=anime,
        max_features=config.TFIDF_MAX_FEATURES,
        min_df=config.TFIDF_MIN_DF,
        ngram_range=config.TFIDF_NGRAM_RANGE,
    )
    metrics = evaluate_precision_recall(
        recommender=recommender,
        test_df=test_df,
        top_k=config.TOP_K,
        relevant_label_threshold=config.RELEVANT_LABEL_THRESHOLD,
    )
    write_report(metrics)
    print(f"Precision@10: {metrics['precision_at_10']:.4f}")
    print(f"Recall@10: {metrics['recall_at_10']:.4f}")
    print(f"Report: {config.REPORT_PATH}")


if __name__ == "__main__":
    main()
