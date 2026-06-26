# Strict and Lenient Model Comparison

Catalog used by recommender: `DataQuality/data/anime_clean.csv`
Evaluation test set: `Annotation/data/final_cleaned_annotated_set.csv`

Test rows: 1500
Seed anime/searches: 100

Strict means only `similarity_label = 2` counts as relevant.
Lenient means `similarity_label >= 1` counts as relevant.

## Strict Evaluation

| Model | P@3 | R@3 | P@5 | R@5 | P@10 | R@10 | P@15 | R@15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.5733 | 0.2473 | 0.4820 | 0.3435 | 0.3410 | 0.4810 | 0.2587 | 0.5450 |
| BM25F | 0.6500 | 0.2810 | 0.5420 | 0.3905 | 0.3670 | 0.5170 | 0.2807 | 0.5907 |
| Embeddings MiniLM | 0.5400 | 0.2350 | 0.4220 | 0.3022 | 0.2730 | 0.3858 | 0.2007 | 0.4279 |
| Embeddings BGE | 0.6967 | 0.3017 | 0.5740 | 0.4074 | 0.3800 | 0.5347 | 0.2853 | 0.6034 |
| Hybrid BGE | 0.7033 | 0.3048 | 0.5960 | 0.4232 | 0.4190 | 0.5906 | 0.3187 | 0.6726 |

## Lenient Evaluation

| Model | P@3 | R@3 | P@5 | R@5 | P@10 | R@10 | P@15 | R@15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.6233 | 0.1278 | 0.5420 | 0.1863 | 0.3960 | 0.2721 | 0.3133 | 0.3224 |
| BM25F | 0.6967 | 0.1443 | 0.6060 | 0.2091 | 0.4400 | 0.3030 | 0.3600 | 0.3711 |
| Embeddings MiniLM | 0.5867 | 0.1220 | 0.4800 | 0.1663 | 0.3390 | 0.2335 | 0.2613 | 0.2696 |
| Embeddings BGE | 0.7667 | 0.1584 | 0.6640 | 0.2283 | 0.4830 | 0.3316 | 0.3913 | 0.4018 |
| Hybrid BGE | 0.7733 | 0.1599 | 0.6780 | 0.2339 | 0.5390 | 0.3702 | 0.4333 | 0.4456 |
