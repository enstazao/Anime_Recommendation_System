"""Evaluate BM25F recommender using Precision@10 and Recall@10."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RecommendationSystem import config
from RecommendationSystem.src.bm25f_recommender import BM25FAnimeRecommender
from RecommendationSystem.src.data_loader import load_anime, load_test_set
from RecommendationSystem.src.evaluate import evaluate_precision_recall


def write_report(metrics: dict[str, float]) -> None:
    config.BM25F_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# BM25F Test Report

Evaluation test set:

`Annotation/data/final_cleaned_annotated_set.csv`

Relevant recommendations are rows with `similarity_label >= {config.RELEVANT_LABEL_THRESHOLD}`.

| Metric | Value |
|---|---:|
| Evaluated seeds | {int(metrics["evaluated_seeds"])} |
| Precision@10 | {metrics["precision_at_10"]:.4f} |
| Recall@10 | {metrics["recall_at_10"]:.4f} |
"""
    config.BM25F_REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    anime = load_anime(config.ANIME_DATA_PATH)
    test_df = load_test_set(config.ANNOTATED_TEST_PATH)
    recommender = BM25FAnimeRecommender(
        anime=anime,
        field_weights=config.BM25F_FIELD_WEIGHTS,
        field_b=config.BM25F_FIELD_B,
        k1=config.BM25F_K1,
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
    print(f"Report: {config.BM25F_REPORT_PATH}")


if __name__ == "__main__":
    main()
