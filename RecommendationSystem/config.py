"""Configuration for the recommendation system."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

ANIME_DATA_PATH = REPO_ROOT / "DataQuality" / "data" / "anime_clean.csv"
ANNOTATED_TEST_PATH = REPO_ROOT / "Annotation" / "data" / "final_cleaned_annotated_set.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "tfidf_test_report.md"
EMBEDDING_REPORT_PATH = PROJECT_ROOT / "reports" / "embedding_test_report.md"

TOP_K = 10
RELEVANT_LABEL_THRESHOLD = 2

TFIDF_MAX_FEATURES = 50000
TFIDF_MIN_DF = 2
TFIDF_NGRAM_RANGE = (1, 2)

EMBEDDING_BATCH_SIZE = 64
EMBEDDING_MODELS = {
    "minilm": {
        "label": "Embeddings MiniLM",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "cache_path": PROJECT_ROOT / "models" / "anime_embeddings_minilm.npy",
        "ids_path": PROJECT_ROOT / "models" / "anime_embedding_ids_minilm.csv",
    },
    "bge": {
        "label": "Embeddings BGE",
        "model_name": "BAAI/bge-base-en-v1.5",
        "cache_path": PROJECT_ROOT / "models" / "anime_embeddings_bge_base.npy",
        "ids_path": PROJECT_ROOT / "models" / "anime_embedding_ids_bge_base.csv",
    },
}
DEFAULT_EMBEDDING_KEY = "minilm"
HYBRID_EMBEDDING_KEY = "bge"
EMBEDDING_MODEL_NAME = EMBEDDING_MODELS[DEFAULT_EMBEDDING_KEY]["model_name"]
EMBEDDING_CACHE_PATH = EMBEDDING_MODELS[DEFAULT_EMBEDDING_KEY]["cache_path"]
EMBEDDING_IDS_PATH = EMBEDDING_MODELS[DEFAULT_EMBEDDING_KEY]["ids_path"]

BM25F_REPORT_PATH = PROJECT_ROOT / "reports" / "bm25f_test_report.md"
BM25F_K1 = 1.5
BM25F_FIELD_WEIGHTS = {
    "title": 2.0,
    "genres": 5.0,
    "tags": 4.0,
    "description": 1.0,
    "studios": 0.5,
    "format": 0.5,
    "source": 0.5,
}
BM25F_FIELD_B = {
    "title": 0.2,
    "genres": 0.1,
    "tags": 0.3,
    "description": 0.75,
    "studios": 0.2,
    "format": 0.1,
    "source": 0.1,
}

HYBRID_REPORT_PATH = PROJECT_ROOT / "reports" / "hybrid_test_report.md"
HYBRID_WEIGHTS = {
    "bm25f": 0.35,
    "tfidf": 0.15,
    "embedding": 0.30,
    "tag_overlap": 0.15,
    "genre_overlap": 0.05,
}
