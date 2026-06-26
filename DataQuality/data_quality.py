"""Compute data-quality metrics for anime_full.csv and write quality_report.md."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "Scraping" / "data" / "raw" / "anime_full.csv"
REPORT_DIR = Path(__file__).resolve().parent / "reports"
REPORT_PATH = REPORT_DIR / "quality_report.md"

KNOWN_FORMATS = {"TV", "TV_SHORT", "MOVIE", "SPECIAL", "OVA", "ONA", "MUSIC"}
KNOWN_STATUSES = {"FINISHED", "RELEASING", "NOT_YET_RELEASED", "CANCELLED"}
KNOWN_SEASONS = {"SPRING", "SUMMER", "FALL", "WINTER"}
HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]+|#\d+);")
PLACEHOLDER_TEXTS = {"no description", "n/a", "none", "tba", "to be announced"}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct(count: int, total: int) -> str:
    return f"{count / total * 100:.1f}%"


def num(val: pd.Series) -> pd.Series:
    return pd.to_numeric(val, errors="coerce")


# ---------------------------------------------------------------------------
# 1. Completeness
# ---------------------------------------------------------------------------

EXPECTED_MISSING = {"title_english", "averageScore"}


def check_completeness(df: pd.DataFrame) -> dict[str, dict]:
    total = len(df)
    results: dict[str, dict] = {}
    for col in df.columns:
        missing = int((df[col].str.strip() == "").sum())
        results[col] = {
            "missing": missing,
            "pct": pct(missing, total),
            "expected": col in EXPECTED_MISSING,
        }
    return results


# ---------------------------------------------------------------------------
# 2. Validity
# ---------------------------------------------------------------------------

def check_validity(df: pd.DataFrame) -> list[dict]:
    total = len(df)
    issues: list[dict] = []

    def issue(label: str, count: int) -> dict:
        return {"check": label, "count": int(count), "pct": pct(int(count), total)}

    avg = num(df["averageScore"])
    issues.append(issue("averageScore out of 0–100", ((avg < 0) | (avg > 100)).sum()))

    mean = num(df["meanScore"])
    issues.append(issue("meanScore out of 0–100", ((mean < 0) | (mean > 100)).sum()))

    eps = num(df["episodes"])
    issues.append(issue("episodes < 0", (eps < 0).sum()))

    dur = num(df["duration"])
    issues.append(issue("duration <= 0", (dur <= 0).sum()))

    sy = num(df["seasonYear"])
    issues.append(issue("seasonYear outside 1940–2027", ((sy < 1940) | (sy > 2027)).sum()))

    non_empty_fmt = df[df["format"].str.strip() != ""]["format"]
    issues.append(issue(
        "format not in known AniList values",
        (~non_empty_fmt.isin(KNOWN_FORMATS)).sum(),
    ))

    non_empty_status = df[df["status"].str.strip() != ""]["status"]
    issues.append(issue(
        "status not in known AniList values",
        (~non_empty_status.isin(KNOWN_STATUSES)).sum(),
    ))

    non_empty_season = df[df["season"].str.strip() != ""]["season"]
    issues.append(issue(
        "season not in known values",
        (~non_empty_season.isin(KNOWN_SEASONS)).sum(),
    ))

    return issues


# ---------------------------------------------------------------------------
# 3. Consistency
# ---------------------------------------------------------------------------

def check_consistency(df: pd.DataFrame) -> list[dict]:
    total = len(df)
    issues: list[dict] = []

    def issue(label: str, count: int) -> dict:
        return {"check": label, "count": int(count), "pct": pct(int(count), total)}

    finished = df["status"] == "FINISHED"
    issues.append(issue(
        "status=FINISHED but end_date empty",
        (finished & (df["end_date"].str.strip() == "")).sum(),
    ))

    eps_zero = num(df["episodes"]) == 0
    issues.append(issue(
        "episodes=0 but status=FINISHED",
        (eps_zero & finished).sum(),
    ))

    season_set = df["season"].str.strip() != ""
    issues.append(issue(
        "season set but format=MOVIE",
        (season_set & (df["format"].str.strip() == "MOVIE")).sum(),
    ))

    has_avg = df["averageScore"].str.strip() != ""
    has_mean = df["meanScore"].str.strip() != ""
    issues.append(issue(
        "averageScore/meanScore presence mismatch",
        ((has_avg & ~has_mean) | (~has_avg & has_mean)).sum(),
    ))

    return issues


# ---------------------------------------------------------------------------
# 4. Uniqueness
# ---------------------------------------------------------------------------

def _normalize_title(t: str) -> str:
    t = t.lower()
    t = re.sub(r"['\"\-!.,]", "", t)
    return " ".join(t.split())


def check_uniqueness(df: pd.DataFrame) -> list[dict]:
    total = len(df)
    issues: list[dict] = []

    def issue(label: str, count: int) -> dict:
        return {"check": label, "count": int(count), "pct": pct(int(count), total)}

    issues.append(issue("duplicate id", int(df["id"].duplicated().sum())))

    norm = df["title_romaji"].apply(_normalize_title)
    dup_rows = int(norm.duplicated().sum())
    issues.append(issue("near-duplicate title_romaji (normalized)", dup_rows))

    return issues


# ---------------------------------------------------------------------------
# 5. Outliers
# ---------------------------------------------------------------------------

def check_outliers(df: pd.DataFrame) -> list[dict]:
    total = len(df)
    issues: list[dict] = []

    def issue(label: str, count: int) -> dict:
        return {"check": label, "count": int(count), "pct": pct(int(count), total)}

    eps = num(df["episodes"])
    issues.append(issue("episodes > 1500", (eps > 1500).sum()))

    dur = num(df["duration"])
    issues.append(issue("duration > 180 min/episode", (dur > 180).sum()))

    pop = num(df["popularity"])
    issues.append(issue("popularity == 0", (pop == 0).sum()))

    fav = num(df["favourites"])
    issues.append(issue("favourites == 0", (fav == 0).sum()))

    return issues


# ---------------------------------------------------------------------------
# 6. Text quality
# ---------------------------------------------------------------------------

def check_text_quality(df: pd.DataFrame) -> list[dict]:
    total = len(df)
    issues: list[dict] = []

    def issue(label: str, count: int) -> dict:
        return {"check": label, "count": int(count), "pct": pct(int(count), total)}

    desc = df["description"].str.strip()

    issues.append(issue("description empty", int((desc == "").sum())))
    issues.append(issue(
        "description < 20 characters (non-empty)",
        int(((desc.str.len() < 20) & (desc != "")).sum()),
    ))
    issues.append(issue(
        "description contains HTML entities (&amp; etc.)",
        int(desc.apply(lambda x: bool(HTML_ENTITY_RE.search(x))).sum()),
    ))
    issues.append(issue(
        "description is placeholder text",
        int(desc.str.lower().apply(lambda x: x.strip() in PLACEHOLDER_TEXTS).sum()),
    ))
    issues.append(issue("cover_image missing", int((df["cover_image"].str.strip() == "").sum())))

    return issues


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _table(rows: list[dict], headers: list[str]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return lines


def write_report(
    df: pd.DataFrame,
    completeness: dict[str, dict],
    validity: list[dict],
    consistency: list[dict],
    uniqueness: list[dict],
    outliers: list[dict],
    text_quality: list[dict],
    path: Path,
) -> None:
    total = len(df)
    lines: list[str] = []

    lines += [
        "# Anime Dataset — Quality Report",
        "",
        f"**Source:** `Scraping/data/raw/anime_full.csv`  ",
        f"**Total rows:** {total:,}  ",
        f"**Total columns:** {len(df.columns)}",
        "",
        "---",
        "",
    ]

    # ---------- 1. Completeness ----------
    lines += ["## 1. Completeness", ""]

    lines += ["### 1a. Expected missing (not counted as errors)", ""]
    lines += ["These columns have structurally missing values that are normal for the domain.", ""]
    exp_rows = [
        {"Column": col, "Missing": f"{v['missing']:,}", "%": v["pct"]}
        for col, v in completeness.items()
        if v["expected"]
    ]
    lines += _table(exp_rows, ["Column", "Missing", "%"])
    lines += [""]

    lines += ["### 1b. Actual missing values", ""]
    act_rows = [
        {"Column": col, "Missing": f"{v['missing']:,}", "%": v["pct"]}
        for col, v in completeness.items()
        if not v["expected"] and v["missing"] > 0
    ]
    if act_rows:
        lines += _table(act_rows, ["Column", "Missing", "%"])
    else:
        lines += ["> No unexpected missing values found."]
    lines += [""]

    lines += ["### 1c. Complete columns (no missing values)", ""]
    ok_cols = [col for col, v in completeness.items() if not v["expected"] and v["missing"] == 0]
    lines += [", ".join(f"`{c}`" for c in ok_cols), ""]

    # ---------- 2. Validity ----------
    lines += ["## 2. Validity", ""]
    lines += _table(validity, ["check", "count", "pct"])
    lines += [""]

    # ---------- 3. Consistency ----------
    lines += ["## 3. Consistency", ""]
    lines += _table(consistency, ["check", "count", "pct"])
    lines += [""]

    # ---------- 4. Uniqueness ----------
    lines += ["## 4. Uniqueness", ""]
    lines += _table(uniqueness, ["check", "count", "pct"])
    lines += ["", "> Near-duplicate detection normalizes title_romaji: lowercase, remove `'\"-!.,`, collapse spaces.", ""]

    # ---------- 5. Outliers ----------
    lines += ["## 5. Outliers", ""]
    lines += _table(outliers, ["check", "count", "pct"])
    lines += ["", "> High episode counts may be legitimate (e.g. long-running children's shows). Flagged, not dropped.", ""]

    # ---------- 6. Text quality ----------
    lines += ["## 6. Text Quality", ""]
    lines += _table(text_quality, ["check", "count", "pct"])
    lines += [""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading {INPUT} …")
    df = load_data(INPUT)
    print(f"  {len(df):,} rows, {len(df.columns)} columns\n")

    print("Checking completeness …")
    completeness = check_completeness(df)

    print("Checking validity …")
    validity = check_validity(df)

    print("Checking consistency …")
    consistency = check_consistency(df)

    print("Checking uniqueness …")
    uniqueness = check_uniqueness(df)

    print("Checking outliers …")
    outliers = check_outliers(df)

    print("Checking text quality …")
    text_quality = check_text_quality(df)

    print()
    write_report(df, completeness, validity, consistency, uniqueness, outliers, text_quality, REPORT_PATH)


if __name__ == "__main__":
    main()
