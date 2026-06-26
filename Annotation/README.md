# Anime Recommendation Annotation

This folder contains the final GPT-OSS annotation pipeline used to create the
ground-truth test set for the anime recommender.

The final dataset is:

`Annotation/data/final_cleaned_annotated_set.csv`

It contains 1,500 rows: 100 seed anime with 15 ranked recommendations per seed.
The labels are created by GPT-OSS using anime genres, tags, studios,
descriptions, and recommendation fit.

## Kept Files

- `prepare_recommendation_candidates.py`: builds a larger candidate pool for
  each popular seed anime.
- `gptoss_client.py`: shared GPT-OSS API client and prompt-formatting helpers.
- `rank_recommendations_gptoss.py`: sends candidates to GPT-OSS, ranks the best
  recommendations, assigns labels and 0-10 scores, then runs verifier passes.
- `make_recommendation_wide.py`: converts the final long CSV into a wide
  spreadsheet-friendly review file.
- `data/recommendation_candidates.csv`: candidate pool used by GPT-OSS.
- `data/final_cleaned_annotated_set.csv`: final long-form annotated test set.
- `data/final_cleaned_annotated_set_wide.csv`: final wide review version.

## Run Order

```bash
python3 Annotation/prepare_recommendation_candidates.py
python3 Annotation/rank_recommendations_gptoss.py
python3 Annotation/make_recommendation_wide.py
```

## Final Dataset Columns

- `seed_id`, `seed_title`: anime used as the query.
- `rank`: recommendation position, where `1` is strongest.
- `recommended_id`, `recommended_title`: recommended anime.
- `similarity_label`: `2`, `1`, or `0`.
- `similarity_score_10`: graded similarity strength from `0` to `10`.
- `reason`: one-line GPT-OSS explanation.
- `seed_genres`, `recommended_genres`: genre metadata.
- `seed_tags`, `recommended_tags`: tag metadata.

## Label Meaning

- `2`: strong recommendation similarity.
- `1`: somewhat similar; useful but weaker.
- `0`: not similar or poor recommendation fit.

## Score Meaning

- `9-10`: extremely strong recommendation match.
- `7-8`: strong recommendation.
- `5-6`: moderate/somewhat similar.
- `0-4`: weak or not useful as a recommendation.

## GPT-OSS Pipeline

For each seed anime, the candidate builder selects a diverse candidate pool:
thematic matches, possible franchise-related entries, partial matches, and some
weak negatives. GPT-OSS then chooses the final ranked recommendations.

The ranking prompt uses three internal passes:

1. Selector pass: identify candidates that are genuinely useful recommendations.
2. Critic pass: remove weak, repetitive, or superficial matches.
3. Calibration pass: make ranks, labels, scores, and reasons consistent.

After that, verifier passes check whether the reasoning is accurate and whether
the labels/scores match the actual recommendation relationship.

Same-franchise anime can be labeled as similar, but the label should still be
based on recommendation value: story, tone, audience fit, genres, tags, and
description, not only the franchise name.
