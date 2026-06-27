# Anime Recommendation System - Data Pipeline

This repository contains the data workflow for an anime recommendation system.
It scrapes anime metadata from AniList, prepares candidate recommendation pairs,
uses GPT-OSS to annotate ranked recommendations, and provides final cleaned
evaluation CSV files for recommender testing.

## Repository Structure

```text
Project/
├── Scraping/          # AniList GraphQL scraper
├── Annotation/        # GPT-OSS recommendation annotation pipeline
├── DataQuality/       # Data quality checks and cleaned catalog pipeline
└── README.md
```

## Stage 1 - Scraping

Source: AniList public GraphQL API, no authentication required  
Script: `Scraping/main.py`  
Main output: `Scraping/data/raw/anime_full.csv`

The scraper collects the AniList anime catalog with fields such as title,
format, status, season, popularity, scores, genres, tags, studios,
description, cover image, and adult-content flag. List fields such as `genres`,
`tags`, and `studios` are stored as `|`-joined strings.

Current scraped dataset:

| Metric | Value |
|---|---:|
| Rows | 22,234 |
| Columns | 25 |
| Adult entries kept | 1,718 |

## Stage 2 - GPT-OSS Annotation

The final testing dataset is a ranked recommendation dataset, not just isolated
pairs. Each seed anime has a top-10 list of recommended anime with an annotation
label, a score out of 10, and a short reason.

Final long-format output:

```text
Annotation/data/final_cleaned_annotated_set.csv
```

Final wide-format review output:

```text
Annotation/data/final_cleaned_annotated_set_wide.csv
```

The long file is best for model evaluation. The wide file is easier to open in
a spreadsheet because each seed anime appears on one row with `rec_1`, `rec_2`,
..., `rec_10` columns.

### Annotation Pipeline

1. `Annotation/prepare_recommendation_candidates.py`
   - Selects 100 popular seed anime.
   - Builds 24 candidate recommendations per seed.
   - Uses mostly thematic/content overlap from genres, tags, descriptions, and
     popularity.
   - Keeps a small number of same-franchise candidates, but they are not given
     automatic priority in the final scoring.

2. `Annotation/rank_recommendations_gptoss.py`
   - Sends one seed anime and its 24 candidates to GPT-OSS.
   - Asks GPT-OSS to choose the top 10 recommendations.
   - GPT-OSS judges recommendation quality from story, tone, premise, audience,
     genres, tags, descriptions, and viewer appeal.
   - Same-franchise anime can score highly, but only when they are genuinely
     good recommendations.

3. `Annotation/make_recommendation_wide.py`
   - Converts the long final CSV into a wide one-row-per-seed review file.

### Final Annotation Dataset

| Metric | Value |
|---|---:|
| Seed anime | 100 |
| Recommendations per seed | 10 |
| Total recommendation rows | 1,000 |
| Duplicate recommendations per seed | 0 |
| Self-recommendations | 0 |

Long-format columns:

```text
seed_id
seed_title
rank
recommended_id
recommended_title
similarity_label
similarity_score_10
reason
seed_genres
recommended_genres
seed_tags
recommended_tags
```

Label meaning:

| Label | Meaning |
|---:|---|
| 2 | Strong recommendation similarity |
| 1 | Somewhat similar |
| 0 | Not similar |

Score meaning:

| Score | Meaning |
|---:|---|
| 10 | Almost perfect recommendation match |
| 8-9 | Very strong recommendation |
| 6-7 | Useful but weaker recommendation |
| 4-5 | Loose similarity |
| 0-3 | Poor recommendation |

Current distribution:

| Field | Value | Count |
|---|---:|---:|
| `similarity_label` | 0 | 2 |
| `similarity_label` | 1 | 413 |
| `similarity_label` | 2 | 585 |

Score distribution:

| `similarity_score_10` | Count |
|---:|---:|
| 3 | 3 |
| 4 | 22 |
| 5 | 155 |
| 6 | 199 |
| 7 | 180 |
| 8 | 194 |
| 9 | 156 |
| 10 | 91 |

Average score decreases by rank, which is expected for a ranked test set:

| Rank | Avg. score |
|---:|---:|
| 1 | 9.75 |
| 2 | 8.94 |
| 3 | 8.41 |
| 4 | 7.86 |
| 5 | 7.38 |
| 6 | 6.79 |
| 7 | 6.43 |
| 8 | 5.88 |
| 9 | 5.45 |
| 10 | 5.03 |

## Stage 3 - Data Quality and Cleaning

Scripts:

```text
DataQuality/data_quality.py
DataQuality/clean_data.py
```

Main cleaned catalog output:

```text
DataQuality/data/anime_clean.csv
```

The cleaning stage removes rows with too little usable content, decodes HTML
entities, removes near-duplicate titles, flags episode-count outliers, and adds
helper columns such as `has_score`.

## How to Run

```bash
# 1. Scrape AniList catalog
python3 Scraping/main.py

# 2. Build candidate recommendation rows
python3 Annotation/prepare_recommendation_candidates.py

# 3. Annotate ranked recommendations with GPT-OSS
python3 Annotation/rank_recommendations_gptoss.py

# 4. Create wide spreadsheet-style version
python3 Annotation/make_recommendation_wide.py

# 5. Optional data quality pipeline
python3 DataQuality/data_quality.py
python3 DataQuality/clean_data.py
```
