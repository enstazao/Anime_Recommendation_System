# NDCG Model Comparison

Evaluation test set: `Annotation/data/final_cleaned_annotated_set.csv`

Test rows: 1500
Seed anime/searches: 100

NDCG uses graded relevance directly: label 2 > label 1 > label 0.

| Model | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@15 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.6240 | 0.5552 | 0.5005 | 0.4864 |
| BM25F | 0.6929 | 0.6165 | 0.5472 | 0.5372 |
| Embeddings MiniLM | 0.5769 | 0.4924 | 0.4260 | 0.4085 |
| Embeddings BGE | 0.7527 | 0.6659 | 0.5854 | 0.5705 |
| Hybrid BGE | 0.7545 | 0.6769 | 0.6225 | 0.6094 |
