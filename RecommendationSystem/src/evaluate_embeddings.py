"""Evaluate a sentence-embedding recommender using Precision@10 and Recall@10."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RecommendationSystem import config
from RecommendationSystem.src.data_loader import load_anime, load_test_set
from RecommendationSystem.src.embedding_recommender import EmbeddingAnimeRecommender
from RecommendationSystem.src.evaluate import evaluate_precision_recall


def write_report(metrics: dict[str, float], embedding_key: str, model_name: str) -> None:
    config.EMBEDDING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_path = config.PROJECT_ROOT / "reports" / f"embedding_{embedding_key}_test_report.md"
    report = f"""# Embedding Recommender Test Report

Model:

`{model_name}`

Evaluation test set:

`Annotation/data/final_cleaned_annotated_set.csv`

Relevant recommendations are rows with `similarity_label >= {config.RELEVANT_LABEL_THRESHOLD}`.

| Metric | Value |
|---|---:|
| Evaluated seeds | {int(metrics["evaluated_seeds"])} |
| Precision@10 | {metrics["precision_at_10"]:.4f} |
| Recall@10 | {metrics["recall_at_10"]:.4f} |
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate embedding recommender.")
    parser.add_argument(
        "--embedding-key",
        choices=sorted(config.EMBEDDING_MODELS),
        default=config.DEFAULT_EMBEDDING_KEY,
        help="Embedding model configuration to evaluate.",
    )
    args = parser.parse_args()
    model_config = config.EMBEDDING_MODELS[args.embedding_key]

    anime = load_anime(config.ANIME_DATA_PATH)
    test_df = load_test_set(config.ANNOTATED_TEST_PATH)
    recommender = EmbeddingAnimeRecommender(
        anime=anime,
        model_name=model_config["model_name"],
        cache_path=model_config["cache_path"],
        ids_path=model_config["ids_path"],
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )
    metrics = evaluate_precision_recall(
        recommender=recommender,
        test_df=test_df,
        top_k=config.TOP_K,
        relevant_label_threshold=config.RELEVANT_LABEL_THRESHOLD,
    )
    write_report(metrics, args.embedding_key, model_config["model_name"])
    print(f"Precision@10: {metrics['precision_at_10']:.4f}")
    print(f"Recall@10: {metrics['recall_at_10']:.4f}")
    print(f"Embedding key: {args.embedding_key}")


if __name__ == "__main__":
    main()
