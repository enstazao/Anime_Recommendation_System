# Hybrid BGE Test Report

Catalog used by recommender: `DataQuality/data/anime_clean.csv`
Evaluation test set: `Annotation/data/final_cleaned_annotated_set.csv`

Strict relevance: `similarity_label = 2`.
Hybrid weights: BM25F 0.35, TF-IDF 0.15, BGE 0.30, tag overlap 0.15, genre overlap 0.05.


| Metric | Value |
|---|---:|
| Precision@10 | 0.4190 |
| Recall@10 | 0.5906 |
