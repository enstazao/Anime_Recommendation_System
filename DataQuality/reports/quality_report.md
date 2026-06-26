# Anime Dataset — Quality Report

**Source:** `Scraping/data/raw/anime_full.csv`  
**Total rows:** 22,234  
**Total columns:** 25

---

## 1. Completeness

### 1a. Expected missing (not counted as errors)

These columns have structurally missing values that are normal for the domain.

| Column | Missing | % |
|---|---|---|
| title_english | 11,795 | 53.0% |
| averageScore | 5,424 | 24.4% |

### 1b. Actual missing values

| Column | Missing | % |
|---|---|---|
| idMal | 1,602 | 7.2% |
| title_native | 175 | 0.8% |
| format | 81 | 0.4% |
| source | 2,423 | 10.9% |
| episodes | 771 | 3.5% |
| duration | 938 | 4.2% |
| season | 7,847 | 35.3% |
| seasonYear | 7,847 | 35.3% |
| start_date | 454 | 2.0% |
| end_date | 1,033 | 4.6% |
| meanScore | 708 | 3.2% |
| genres | 2,981 | 13.4% |
| tags | 4,595 | 20.7% |
| studios | 4,363 | 19.6% |
| description | 1,436 | 6.5% |

### 1c. Complete columns (no missing values)

`id`, `title_romaji`, `type`, `status`, `popularity`, `favourites`, `cover_image`, `isAdult`

## 2. Validity

| check | count | pct |
|---|---|---|
| averageScore out of 0–100 | 0 | 0.0% |
| meanScore out of 0–100 | 0 | 0.0% |
| episodes < 0 | 0 | 0.0% |
| duration <= 0 | 0 | 0.0% |
| seasonYear outside 1940–2027 | 0 | 0.0% |
| format not in known AniList values | 0 | 0.0% |
| status not in known AniList values | 0 | 0.0% |
| season not in known values | 0 | 0.0% |

## 3. Consistency

| check | count | pct |
|---|---|---|
| status=FINISHED but end_date empty | 73 | 0.3% |
| episodes=0 but status=FINISHED | 0 | 0.0% |
| season set but format=MOVIE | 2185 | 9.8% |
| averageScore/meanScore presence mismatch | 4716 | 21.2% |

## 4. Uniqueness

| check | count | pct |
|---|---|---|
| duplicate id | 0 | 0.0% |
| near-duplicate title_romaji (normalized) | 178 | 0.8% |

> Near-duplicate detection normalizes title_romaji: lowercase, remove `'"-!.,`, collapse spaces.

## 5. Outliers

| check | count | pct |
|---|---|---|
| episodes > 1500 | 3 | 0.0% |
| duration > 180 min/episode | 0 | 0.0% |
| popularity == 0 | 0 | 0.0% |
| favourites == 0 | 4377 | 19.7% |

> High episode counts may be legitimate (e.g. long-running children's shows). Flagged, not dropped.

## 6. Text Quality

| check | count | pct |
|---|---|---|
| description empty | 1436 | 6.5% |
| description < 20 characters (non-empty) | 58 | 0.3% |
| description contains HTML entities (&amp; etc.) | 2 | 0.0% |
| description is placeholder text | 0 | 0.0% |
| cover_image missing | 0 | 0.0% |
