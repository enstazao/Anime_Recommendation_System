# Model Comparison Report

Evaluation test set: `Annotation/data/final_cleaned_annotated_set.csv`

Catalog used by recommender: `DataQuality/data/anime_clean.csv`

Test rows: 1500
Seed anime: 100
Recommendations per seed: 15

Strict relevance: only rows with `similarity_label = 2` count as relevant.

| Model | Precision@5 | Recall@5 | Precision@10 | Recall@10 | Precision@15 | Recall@15 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.4820 | 0.3435 | 0.3410 | 0.4810 | 0.2587 | 0.5450 |
| BM25F | 0.5420 | 0.3905 | 0.3670 | 0.5170 | 0.2807 | 0.5907 |
| Embeddings MiniLM | 0.4220 | 0.3022 | 0.2730 | 0.3858 | 0.2007 | 0.4279 |
| Embeddings BGE | 0.5740 | 0.4074 | 0.3800 | 0.5347 | 0.2853 | 0.6034 |
| Hybrid BGE | 0.5960 | 0.4232 | 0.4190 | 0.5906 | 0.3187 | 0.6726 |
