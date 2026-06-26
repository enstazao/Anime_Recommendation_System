# Embedding Recommender Test Report

Model:

`sentence-transformers/all-MiniLM-L6-v2`

Evaluation test set:

`Annotation/data/final_cleaned_annotated_set.csv`

Relevant recommendations are rows with `similarity_label >= 2`.

| Metric | Value |
|---|---:|
| Evaluated seeds | 100 |
| Precision@10 | 0.2740 |
| Recall@10 | 0.3878 |
