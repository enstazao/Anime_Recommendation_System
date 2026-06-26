"""Clean anime_full.csv and save to DataQuality/data/anime_clean.csv."""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "Scraping" / "data" / "raw" / "anime_full.csv"
OUTPUT = Path(__file__).resolve().parent / "data" / "anime_clean.csv"

HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]+|#\d+);")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


# ---------------------------------------------------------------------------
# Snapshot for before/after comparison
# ---------------------------------------------------------------------------

def snapshot(df: pd.DataFrame) -> dict:
    total = len(df)

    def pct(n: int) -> str:
        return f"{n / total * 100:.1f}%" if total else "0.0%"

    def filled(col: str) -> int:
        return int((df[col].str.strip() != "").sum())

    return {
        "rows": total,
        "genres_%": pct(filled("genres")),
        "tags_%": pct(filled("tags")),
        "description_%": pct(filled("description")),
        "averageScore_%": pct(filled("averageScore")),
        "cover_image_%": pct(filled("cover_image")),
    }


# ---------------------------------------------------------------------------
# Step 1: Drop rows where BOTH genres AND description are empty
# ---------------------------------------------------------------------------

def drop_empty_core(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    mask = (df["genres"].str.strip() == "") & (df["description"].str.strip() == "")
    n = int(mask.sum())
    return df[~mask].reset_index(drop=True), n


# ---------------------------------------------------------------------------
# Step 2: Decode HTML entities in descriptions
# ---------------------------------------------------------------------------

def decode_html_entities(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    has_entity = df["description"].apply(lambda x: bool(HTML_ENTITY_RE.search(x)))
    n = int(has_entity.sum())
    df = df.copy()
    df["description"] = df["description"].apply(html.unescape)
    return df, n


# ---------------------------------------------------------------------------
# Step 3: Trim whitespace in text fields
# ---------------------------------------------------------------------------

def trim_text(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = ["title_romaji", "title_english", "title_native", "description"]
    df = df.copy()
    for col in text_cols:
        df[col] = df[col].str.strip()
    return df


# ---------------------------------------------------------------------------
# Step 4: Standardize pipe-separated list fields
# ---------------------------------------------------------------------------

def standardize_list_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    list_cols = ["genres", "tags", "studios"]
    df = df.copy()
    fixed = 0
    for col in list_cols:
        original = df[col].copy()
        df[col] = df[col].apply(
            lambda v: "|".join(p.strip() for p in v.split("|") if p.strip())
        )
        fixed += int((df[col] != original).sum())
    return df, fixed


# ---------------------------------------------------------------------------
# Step 5: Add has_score boolean column
# ---------------------------------------------------------------------------

def add_has_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.insert(
        df.columns.get_loc("averageScore") + 1,
        "has_score",
        df["averageScore"].str.strip() != "",
    )
    return df


# ---------------------------------------------------------------------------
# Step 6: Remove exact duplicate IDs (keep highest popularity)
# ---------------------------------------------------------------------------

def remove_duplicate_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    df["_pop"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    df = df.sort_values("_pop", ascending=False).drop_duplicates(subset=["id"], keep="first")
    df = df.drop(columns=["_pop"]).reset_index(drop=True)
    return df, 0  # count returned by caller via len diff


# ---------------------------------------------------------------------------
# Step 7: Remove near-duplicate titles (keep highest popularity)
# ---------------------------------------------------------------------------

def _normalize_title(t: str) -> str:
    t = t.lower()
    t = re.sub(r"['\"\-!.,]", "", t)
    return " ".join(t.split())


def remove_near_duplicate_titles(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    df["_pop"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    df["_norm"] = df["title_romaji"].apply(_normalize_title)
    before = len(df)
    df = df.sort_values("_pop", ascending=False).drop_duplicates(subset=["_norm"], keep="first")
    removed = before - len(df)
    df = df.drop(columns=["_pop", "_norm"]).reset_index(drop=True)
    return df, removed


# ---------------------------------------------------------------------------
# Step 8: Flag episode outliers (don't drop — long-running shows are real)
# ---------------------------------------------------------------------------

def flag_episode_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    eps = pd.to_numeric(df["episodes"], errors="coerce")
    df["episodes_outlier"] = eps > 1500
    n = int(df["episodes_outlier"].sum())
    return df, n


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(
    before: dict,
    after: dict,
    steps: list[tuple[str, int | str]],
) -> None:
    print()
    print("=" * 60)
    print("  BEFORE vs AFTER SUMMARY")
    print("=" * 60)
    metrics = [
        ("Total rows", "rows", "", ""),
        ("genres complete", "genres_%", "", ""),
        ("tags complete", "tags_%", "", ""),
        ("description complete", "description_%", "", ""),
        ("averageScore complete", "averageScore_%", "", ""),
        ("cover_image complete", "cover_image_%", "", ""),
    ]
    print(f"  {'Metric':<30} {'Before':>10} {'After':>10}")
    print("  " + "-" * 52)
    for label, key, *_ in metrics:
        bv = before[key]
        av = after[key]
        if isinstance(bv, int):
            print(f"  {label:<30} {bv:>10,} {av:>10,}")
        else:
            print(f"  {label:<30} {bv:>10} {av:>10}")
    print()
    print("  Fixes applied:")
    for step, count in steps:
        print(f"    • {step}: {count}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading {INPUT} …")
    df = load_data(INPUT)
    print(f"  {len(df):,} rows loaded\n")
    before = snapshot(df)
    steps: list[tuple[str, int | str]] = []

    # Step 1
    df, n = drop_empty_core(df)
    steps.append(("Rows dropped (no genres AND no description)", n))

    # Step 2
    df, n = decode_html_entities(df)
    steps.append(("Descriptions with HTML entities decoded", n))

    # Step 3
    df = trim_text(df)
    steps.append(("Text fields whitespace-trimmed", "all rows"))

    # Step 4
    df, n = standardize_list_fields(df)
    steps.append(("List fields standardized (stray pipes cleaned)", n))

    # Step 5
    df = add_has_score(df)
    steps.append(("has_score column added", len(df)))

    # Step 6
    before_6 = len(df)
    df, _ = remove_duplicate_ids(df)
    n6 = before_6 - len(df)
    steps.append(("Exact duplicate IDs removed", n6))

    # Step 7
    df, n7 = remove_near_duplicate_titles(df)
    steps.append(("Near-duplicate titles removed (kept most popular)", n7))

    # Step 8
    df, n8 = flag_episode_outliers(df)
    steps.append((f"Rows flagged episodes_outlier (> 1500 eps)", n8))

    after = snapshot(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df):,} rows → {OUTPUT}")

    print_summary(before, after, steps)


if __name__ == "__main__":
    main()
