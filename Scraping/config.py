"""Configuration constants for the AniList anime scraper."""

API_URL = "https://graphql.anilist.co"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "BHT-Berlin-DataScience-Workflows-AniListScraper/0.1",
}

RATE_LIMIT_SECONDS = 1.0
MAX_429_RETRIES = 5
DEFAULT_RETRY_AFTER_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 30
PAGE_SIZE = 50

RAW_DATA_DIR = "data/raw"
LOG_DIR = "logs"
LOG_FILE = f"{LOG_DIR}/scraper.log"
