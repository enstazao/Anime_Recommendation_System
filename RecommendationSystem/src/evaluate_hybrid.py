"""Evaluate hybrid recommender using Precision@10 and Recall@10."""

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
from RecommendationSystem.src.embedding_recommender import EmbeddingAnimeRecommender
from RecommendationSystem.src.evaluate import evaluate_precision_recall
from RecommendationSystem.src.hybrid_recommender import HybridAnimeRecommender
from RecommendationSystem.src.tfidf_recommender import TfidfAnimeRecommender


def write_report(metrics: dict[str, float]) -> None:
    config.HYBRID_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# Hybrid Recommender Test Report

Evaluation test set:

`Annotation/data/final_cleaned_annotated_set.csv`

Relevant recommendations are rows with `similarity_label >= {config.RELEVANT_LABEL_THRESHOLD}`.

Hybrid weights:

| Signal | Weight |
|---|---:|
| BM25F | {config.HYBRID_WEIGHTS["bm25f"]:.2f} |
| TF-IDF | {config.HYBRID_WEIGHTS["tfidf"]:.2f} |
| Embeddings | {config.HYBRID_WEIGHTS["embedding"]:.2f} |
| Tag overlap | {config.HYBRID_WEIGHTS["tag_overlap"]:.2f} |
| Genre overlap | {config.HYBRID_WEIGHTS["genre_overlap"]:.2f} |

| Metric | Value |
|---|---:|
| Evaluated seeds | {int(metrics["evaluated_seeds"])} |
| Precision@10 | {metrics["precision_at_10"]:.4f} |
| Recall@10 | {metrics["recall_at_10"]:.4f} |
"""
    config.HYBRID_REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    anime = load_anime(config.ANIME_DATA_PATH)
    test_df = load_test_set(config.ANNOTATED_TEST_PATH)
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
    embedding_config = config.EMBEDDING_MODELS[config.HYBRID_EMBEDDING_KEY]
    embeddings = EmbeddingAnimeRecommender(
        anime=anime,
        model_name=embedding_config["model_name"],
        cache_path=embedding_config["cache_path"],
        ids_path=embedding_config["ids_path"],
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )
    recommender = HybridAnimeRecommender(
        anime=anime,
        tfidf=tfidf,
        bm25f=bm25f,
        embeddings=embeddings,
        weights=config.HYBRID_WEIGHTS,
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
    print(f"Report: {config.HYBRID_REPORT_PATH}")


if __name__ == "__main__":
    main()
