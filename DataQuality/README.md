# DataQuality

Data-quality analysis and cleaning pipeline for `anime_full.csv` (~22,000 anime scraped from AniList).

## How to run

```bash
# From the project root:
python3 DataQuality/data_quality.py   # generates reports/quality_report.md
python3 DataQuality/clean_data.py     # generates data/anime_clean.csv
```

Both scripts are fully reproducible and require only **pandas** (`pip install pandas`).

---

## What data_quality.py measures

| Category | What is checked |
|---|---|
| **Completeness** | % non-null for every column. `title_english` and `averageScore` are reported separately as "expected missing" since many anime have no English title and unrated anime have no score. |
| **Validity** | Range checks (`averageScore`/`meanScore` 0–100, `episodes` ≥ 0, `duration` > 0, `seasonYear` 1940–2027) and enum checks (`format`, `status`, `season` against known AniList values). |
| **Consistency** | Cross-field contradictions: `FINISHED` entries with no `end_date`; episodes=0 on a finished show; `season` set on a movie. Also checks if `averageScore` and `meanScore` presence agree. |
| **Uniqueness** | Duplicate `id` values; near-duplicate `title_romaji` (normalized: lowercase, remove `' " - ! . ,`, collapse spaces). |
| **Outliers** | Episodes > 1500 (long-running children's shows); duration > 180 min/episode; zero `popularity` or `favourites` (potential stub entries). |
| **Text quality** | Empty descriptions; descriptions < 20 characters; HTML entities (`&amp;` etc.); placeholder text; missing `cover_image`. |

The report is written to `reports/quality_report.md`.

---

## What clean_data.py fixes

| Step | Decision |
|---|---|
| **Drop empty core rows** | Remove rows where both `genres` and `description` are empty — these entries have no usable content for a recommender. |
| **Decode HTML entities** | `html.unescape()` on `description` to convert `&amp;` → `&`, `&lt;` → `<`, etc. |
| **Trim whitespace** | Strip leading/trailing whitespace from `title_romaji`, `title_english`, `title_native`, `description`. |
| **Standardize list fields** | Rebuild `genres`, `tags`, `studios` as clean `|`-joined strings with no empty segments or extra spaces. |
| **Add has_score** | Boolean column inserted after `averageScore`. Lets downstream code exclude unrated entries without losing the rows. `averageScore` itself is left as-is (NaN stays NaN). |
| **Remove duplicate IDs** | Keep the row with highest `popularity` when the same `id` appears more than once. |
| **Remove near-duplicate titles** | Same normalization as the quality check. Keep the most popular entry when multiple rows share a normalized `title_romaji`. Different seasons with distinct names (e.g. "Gintama°" vs "Gintama'") are treated as separate entries. |
| **Flag episode outliers** | Add boolean `episodes_outlier` for entries with > 1500 episodes. These are real shows (e.g. Sazae-san) — they are flagged but **not dropped**. |

Cleaned output: `data/anime_clean.csv`.

---

## Output files

| File | Description |
|---|---|
| `reports/quality_report.md` | Full quality metrics report with counts and percentages |
| `data/anime_clean.csv` | Cleaned dataset, ready for the recommender pipeline |
