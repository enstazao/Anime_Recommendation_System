"""Flask API for the anime recommender."""

from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from RecommendationSystem import config
from RecommendationSystem.src.bm25f_recommender import BM25FAnimeRecommender
from RecommendationSystem.src.data_loader import load_anime
from RecommendationSystem.src.embedding_recommender import EmbeddingAnimeRecommender
from RecommendationSystem.src.hybrid_recommender import HybridAnimeRecommender
from RecommendationSystem.src.tfidf_recommender import TfidfAnimeRecommender


app = Flask(__name__, static_folder=None)
_anime = None
_tfidf_recommender: TfidfAnimeRecommender | None = None
_embedding_recommenders: dict[str, EmbeddingAnimeRecommender] = {}
_bm25f_recommender: BM25FAnimeRecommender | None = None
_hybrid_recommender: HybridAnimeRecommender | None = None


def get_anime():
    global _anime
    if _anime is None:
        _anime = load_anime(config.ANIME_DATA_PATH)
    return _anime


def get_tfidf_recommender() -> TfidfAnimeRecommender:
    global _tfidf_recommender
    if _tfidf_recommender is None:
        _tfidf_recommender = TfidfAnimeRecommender(
            anime=get_anime(),
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
        )
    return _tfidf_recommender


def get_embedding_recommender(key: str = config.DEFAULT_EMBEDDING_KEY) -> EmbeddingAnimeRecommender:
    if key not in config.EMBEDDING_MODELS:
        raise KeyError(f"Unknown embedding model: {key}")
    if key not in _embedding_recommenders:
        model_config = config.EMBEDDING_MODELS[key]
        _embedding_recommenders[key] = EmbeddingAnimeRecommender(
            anime=get_anime(),
            model_name=model_config["model_name"],
            cache_path=model_config["cache_path"],
            ids_path=model_config["ids_path"],
            batch_size=config.EMBEDDING_BATCH_SIZE,
        )
    return _embedding_recommenders[key]


def get_bm25f_recommender() -> BM25FAnimeRecommender:
    global _bm25f_recommender
    if _bm25f_recommender is None:
        _bm25f_recommender = BM25FAnimeRecommender(
            anime=get_anime(),
            field_weights=config.BM25F_FIELD_WEIGHTS,
            field_b=config.BM25F_FIELD_B,
            k1=config.BM25F_K1,
        )
    return _bm25f_recommender


def get_hybrid_recommender() -> HybridAnimeRecommender:
    global _hybrid_recommender
    if _hybrid_recommender is None:
        _hybrid_recommender = HybridAnimeRecommender(
            anime=get_anime(),
            tfidf=get_tfidf_recommender(),
            bm25f=get_bm25f_recommender(),
            embeddings=get_embedding_recommender(config.HYBRID_EMBEDDING_KEY),
            weights=config.HYBRID_WEIGHTS,
        )
    return _hybrid_recommender


def get_recommender(model_name: str):
    if model_name in {"embedding", "embedding_minilm"}:
        return get_embedding_recommender()
    if model_name == "embedding_bge":
        return get_embedding_recommender("bge")
    if model_name == "bm25f":
        return get_bm25f_recommender()
    if model_name == "hybrid":
        return get_hybrid_recommender()
    return get_tfidf_recommender()


@app.get("/")
def index():
    return send_from_directory(Path(__file__).resolve().parent, "index.html")


@app.get("/api/search")
def search():
    query = request.args.get("q", "")
    recommender = get_tfidf_recommender()
    return jsonify({"results": recommender.search_titles(query, limit=10)})


@app.get("/api/recommend")
def recommend():
    model_name = request.args.get("model", "tfidf").strip().lower()
    valid_models = {"tfidf", "embedding", "embedding_minilm", "embedding_bge", "bm25f", "hybrid"}
    if model_name not in valid_models:
        return jsonify({"error": "model must be one of: " + ", ".join(sorted(valid_models))}), 400

    search_recommender = get_tfidf_recommender()
    recommender = get_recommender(model_name)
    anime_id = request.args.get("id", "")
    query = request.args.get("q", "")
    top_k = int(request.args.get("top_k", config.TOP_K))

    try:
        if not anime_id and query:
            anime_id = search_recommender.best_title_match(query)
        recommendations = recommender.recommend_by_id(anime_id, top_k=top_k)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 404

    source = recommender.anime.iloc[recommender.id_to_index[str(anime_id)]]
    return jsonify(
        {
            "source": {
                "id": str(source["id"]),
                "title": str(source["title_romaji"]),
                "title_english": str(source.get("title_english", "")),
                "cover_image": str(source.get("cover_image", "")),
            },
            "model": model_name,
            "model_label": model_label(model_name),
            "recommendations": [
                {
                    "id": item.id,
                    "title": item.title,
                    "title_english": item.title_english,
                    "score": round(item.score, 4),
                    "genres": item.genres,
                    "tags": item.tags,
                    "cover_image": item.cover_image,
                }
                for item in recommendations
            ],
        }
    )


def model_label(model_name: str) -> str:
    labels = {
        "tfidf": "TF-IDF",
        "embedding": "Embeddings MiniLM",
        "embedding_minilm": "Embeddings MiniLM",
        "embedding_bge": "Embeddings BGE",
        "bm25f": "BM25F",
        "hybrid": "Hybrid",
    }
    return labels.get(model_name, "TF-IDF")


if __name__ == "__main__":
    app.run(debug=True)
