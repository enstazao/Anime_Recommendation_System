# AniList Anime Scraper

Modular Python scraper for public anime data from the AniList GraphQL API.
AniList does not require authentication for this public query.

The default command performs a full catalog run: it fetches all AniList anime
pages sorted by popularity and saves the result to `data/raw/anime_full.csv`.

## Project Structure

```text
Scraping/
├── README.md
├── requirements.txt
├── config.py
├── src/
│   ├── __init__.py
│   ├── api_client.py
│   ├── queries.py
│   ├── parser.py
│   ├── scraper.py
│   └── utils.py
├── data/
│   └── raw/
└── main.py
```

## Modules

- `config.py`: AniList endpoint URL, request headers, rate limit, retry settings,
  log path, and default page size.
- `src/api_client.py`: low-level GraphQL POST client using `requests`, including
  one-second request spacing and HTTP `429` retry/backoff from `Retry-After`.
- `src/queries.py`: GraphQL query strings.
- `src/parser.py`: flattens AniList JSON into CSV-ready dictionaries, joins list
  fields with `|`, and strips leftover HTML from descriptions.
- `src/scraper.py`: orchestrates page fetching, incremental saving, and resume
  behavior by skipping IDs already present in the output CSV.
- `src/utils.py`: logging setup, CSV helpers, and a compact console table preview.
- `main.py`: CLI entry point.

## Setup

```bash
cd Scraping
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Full Run

```bash
python main.py
```

This writes:

```text
data/raw/anime_full.csv
```

The command prints a final summary with total rows, pages requested, output path,
and file size.

Useful options:

```bash
python main.py --limit 5
python main.py --output data/raw/my_anime_full.csv
python main.py --start-page 2 --limit 10 --output data/raw/anime_pages_2_to_11.csv
```

`--limit` is a page cap for testing. By default there is no cap, so the scraper
continues until AniList returns `pageInfo.hasNextPage = false`.

## Fields Captured

The scraper flattens these AniList fields into CSV columns:

- `id`, `idMal`
- `title_romaji`, `title_english`, `title_native`
- `type`, `format`, `source`
- `episodes`, `duration`
- `status`
- `season`, `seasonYear`
- `start_date`, `end_date`
- `averageScore`, `meanScore`, `popularity`, `favourites`
- `genres`
- `tags`
- `studios`
- `description`
- `cover_image`
- `isAdult`

List fields such as `genres`, `tags`, and `studios` are stored as `|`-joined
strings.

## Resume Behavior

Before saving, the scraper reads existing IDs from the output CSV and skips items
already present. New rows are appended after each page so a crash or interruption
does not lose completed progress.

## Scaling Later

Scaling is controlled by CLI values rather than changing parser/client code:

```bash
python main.py --per-page 50 --output data/raw/anime_popular.csv
```

The scraper follows `pageInfo.hasNextPage` and increments pages until AniList
reports that there are no more pages. AniList limits offset pagination to 5,000
entries, so the scraper automatically continues in popularity windows using the
`popularity_lesser` filter when it reaches that page-depth boundary.
