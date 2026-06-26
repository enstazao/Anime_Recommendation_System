"""Convert ranked recommendation rows into one-row-per-seed wide format."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().parent / "data" / "final_cleaned_annotated_set.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "final_cleaned_annotated_set_wide.csv"


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    output = Path(args.output)

    df = pd.read_csv(source).sort_values(["seed_title", "rank"])
    rows = []
    for (seed_id, seed_title), group in df.groupby(["seed_id", "seed_title"], sort=False):
        row = {"seed_id": seed_id, "seed_title": seed_title}
        for _, rec in group.sort_values("rank").iterrows():
            rank = int(rec["rank"])
            row[f"rec_{rank}_title"] = rec["recommended_title"]
            row[f"rec_{rank}_label"] = int(rec["similarity_label"])
            row[f"rec_{rank}_score_10"] = int(rec["similarity_score_10"])
            row[f"rec_{rank}_reason"] = rec["reason"]
        rows.append(row)

    wide = pd.DataFrame(rows)
    wide.to_csv(output, index=False)
    print(f"Created {output} with {len(wide)} seed rows.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create wide recommendation-list CSV.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
