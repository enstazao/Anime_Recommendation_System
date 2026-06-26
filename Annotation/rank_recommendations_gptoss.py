"""Create GPT-OSS ranked anime recommendation lists."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from gptoss_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    GPTOSSClient,
    extract_json,
    pipe_to_commas,
    resolve_api_key,
    shorten,
    top_items,
)


DEFAULT_INPUT = Path(__file__).resolve().parent / "data" / "recommendation_candidates.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "final_cleaned_annotated_set.csv"
DEFAULT_AUDIT = Path(__file__).resolve().parent / "data" / "final_cleaned_annotated_set_audit.jsonl"
DEFAULT_TOP_K = 15


SYSTEM_PROMPT = """You are an expert anime recommendation annotator creating a high-quality offline test set.

Your job is to choose and rank recommendations for one seed anime from a candidate list.
Think like a careful human anime fan and recommender-system evaluator, not like a keyword counter.

Use all available evidence:
- genres and tags
- description/premise
- tone and emotional appeal
- target audience and viewing experience
- story structure, character dynamics, setting, and themes
- franchise continuity only when it genuinely improves recommendation value

Important rules:
- Do not rank something highly only because it has a similar title or belongs to the same franchise.
- Same-franchise anime, sequels, films, OVAs, remakes, and spin-offs may score highly, but only if they are genuinely useful recommendations for a fan of the seed.
- Prefer strong thematic/content fit over shallow title matching.
- Avoid duplicate-feeling recommendations when a stronger, more informative recommendation exists.
- Rank the list so rank 1 is the strongest recommendation and later ranks gradually become weaker.
- Use the full 0-10 scale; do not give everything 8-10.
- Do not recommend the seed anime itself.

Before returning JSON, internally perform these three passes:
1. Selector pass: identify the strongest direct-continuation and strongest different-franchise thematic candidates.
2. Critic pass: remove weak, redundant, or only superficially related candidates.
3. Calibration pass: make sure scores decrease sensibly by rank and labels match scores.

Labels and scores:
- similarity_label is one of 0, 1, or 2.
- 2 = strong recommendation similarity; a fan of the seed would very likely enjoy it.
- 1 = somewhat similar; useful connection, but weaker or clearly different in tone/premise/audience.
- 0 = not similar; poor recommendation. This should be rare in a top list.
- similarity_score_10 is an integer from 0 to 10 showing strength:
  10 = nearly perfect recommendation, 8-9 = very strong, 6-7 = useful but weaker,
  4-5 = loose similarity, 0-3 = poor recommendation.

Score-label consistency:
- label 2 should usually have score 7-10.
- label 1 should usually have score 4-6.
- label 0 should usually have score 0-3.

Reasons:
- one concise sentence
- explain the actual recommendation relationship
- mention story/tone/theme/franchise only when relevant

Return only valid JSON with this exact shape:
{"recommendations":[{"candidate_id":"123","rank":1,"similarity_label":2,"similarity_score_10":9,"reason":"one concise sentence"}, ...]}
"""


VERIFIER_PROMPT = """You are a strict verifier agent for an anime recommendation test set.

You will receive:
1. one seed anime,
2. the full candidate pool,
3. the current ranked recommendation list.

Your job is to audit the current list and fix it if needed.

Check carefully:
- Is each recommendation genuinely useful for a fan of the seed?
- Is the reasoning accurate and specific?
- Are same-franchise items ranked highly only when they are genuinely useful?
- Are strong different-franchise thematic matches included when they are better than weak franchise/sidebar entries?
- Are labels consistent with scores?
- Does the list gradually move from strongest to weaker recommendations?
- Are there duplicates, self-recommendations, or superficial matches?

You may keep, reorder, rescore, rewrite reasons, or replace recommendations using items from the candidate pool.
Return exactly the requested number of recommendations.

Labels and scores:
- label 2: score 7-10, strong recommendation.
- label 1: score 4-6, somewhat similar.
- label 0: score 0-3, poor recommendation.

Return only valid JSON with this exact shape:
{"recommendations":[{"candidate_id":"123","rank":1,"similarity_label":2,"similarity_score_10":9,"reason":"one concise sentence"}, ...]}
"""


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.input, dtype=str).fillna("")

    client = GPTOSSClient(
        api_key=resolve_api_key(args.api_key),
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        retries=args.retries,
        retry_delay=args.retry_delay,
    )

    output_path = Path(args.output)
    audit_path = Path(args.audit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    completed_seed_ids = completed_seeds(output_path) if args.resume else set()
    rows: list[dict[str, Any]] = load_existing(output_path) if args.resume else []

    groups = list(candidates.groupby(["seed_id", "seed_title"], sort=False))
    if args.max_seeds:
        groups = groups[: args.max_seeds]

    for index, ((seed_id, seed_title), group) in enumerate(groups, start=1):
        if str(seed_id) in completed_seed_ids:
            continue

        print(
            f"[seed {index}/{len(groups)}] ranking {seed_title} "
            f"from {len(group)} candidates",
            flush=True,
        )

        recommendations = rank_seed(client, group, top_k=args.top_k)
        for verify_pass in range(1, args.verify_passes + 1):
            recommendations = verify_seed(
                client=client,
                group=group,
                recommendations=recommendations,
                top_k=args.top_k,
                verify_pass=verify_pass,
            )
        seed_rows = build_output_rows(group, recommendations, top_k=args.top_k)
        rows.extend(seed_rows)
        write_output(output_path, rows)
        append_audit(audit_path, seed_id=str(seed_id), seed_title=str(seed_title), recommendations=recommendations)

        print(f"  wrote {len(seed_rows)} recommendations; total rows={len(rows)}", flush=True)
        time.sleep(args.batch_pause)

    write_output(output_path, rows)
    print_summary(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank anime recommendations with GPT-OSS.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", default=DEFAULT_AUDIT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=int, default=8)
    parser.add_argument("--batch-pause", type=float, default=0.5)
    parser.add_argument("--verify-passes", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def rank_seed(client: GPTOSSClient, group: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    raw = client.call(SYSTEM_PROMPT, build_user_prompt(group, top_k=top_k))
    data = extract_json(raw)
    return parse_recommendations(data=data, group=group, top_k=top_k, raw=raw)


def verify_seed(
    client: GPTOSSClient,
    group: pd.DataFrame,
    recommendations: list[dict[str, Any]],
    top_k: int,
    verify_pass: int,
) -> list[dict[str, Any]]:
    raw = client.call(VERIFIER_PROMPT, build_verifier_prompt(group, recommendations, top_k, verify_pass))
    data = extract_json(raw)
    return parse_recommendations(data=data, group=group, top_k=top_k, raw=raw)


def parse_recommendations(
    data: dict[str, Any],
    group: pd.DataFrame,
    top_k: int,
    raw: str,
) -> list[dict[str, Any]]:
    recommendations = data.get("recommendations")
    if not isinstance(recommendations, list):
        raise ValueError(f"Expected recommendations list, got: {raw[:500]}")

    candidate_ids = set(group["candidate_id"].astype(str))
    cleaned = []
    seen_ids: set[str] = set()
    for item in recommendations:
        candidate_id = str(item["candidate_id"])
        if candidate_id not in candidate_ids or candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        cleaned.append(
            {
                "candidate_id": candidate_id,
                "rank": int(item["rank"]),
                "similarity_label": normalize_label_from_score(
                    clamp_label(int(item["similarity_label"])),
                    clamp_score_10(int(item["similarity_score_10"])),
                ),
                "similarity_score_10": clamp_score_10(int(item["similarity_score_10"])),
                "reason": str(item["reason"]).strip(),
            }
        )

    if len(cleaned) < top_k:
        fill_missing_recommendations(cleaned, group, seen_ids, top_k)

    cleaned = sorted(cleaned, key=lambda row: (-row["similarity_score_10"], -row["similarity_label"], row["rank"]))[:top_k]
    for rank, item in enumerate(cleaned, start=1):
        item["rank"] = rank
    return cleaned


def fill_missing_recommendations(
    cleaned: list[dict[str, Any]],
    group: pd.DataFrame,
    seen_ids: set[str],
    top_k: int,
) -> None:
    """Keep long runs alive if the model returns one or two fewer rows."""
    next_rank = len(cleaned) + 1
    for _, row in group.iterrows():
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        cleaned.append(
            {
                "candidate_id": candidate_id,
                "rank": next_rank,
                "similarity_label": 1,
                "similarity_score_10": 4,
                "reason": "Backup candidate added to keep the verified recommendation list complete.",
            }
        )
        next_rank += 1
        if len(cleaned) >= top_k:
            return


def build_user_prompt(group: pd.DataFrame, top_k: int) -> str:
    seed = group.iloc[0]
    candidates = []
    for _, row in group.iterrows():
        candidates.append(
            "\n".join(
                [
                    f"Candidate ID: {row['candidate_id']}",
                    f"Title: {row['candidate_title']}",
                    f"Genres: {pipe_to_commas(row['candidate_genres'])}",
                    f"Tags: {top_items(row['candidate_tags'])}",
                    f"Description: {shorten(row['candidate_description'])}",
                ]
            )
        )

    return (
        f"Seed anime:\n"
        f"ID: {seed['seed_id']}\n"
        f"Title: {seed['seed_title']}\n"
        f"Genres: {pipe_to_commas(seed['seed_genres'])}\n"
        f"Tags: {top_items(seed['seed_tags'])}\n"
        f"Description: {shorten(seed['seed_description'])}\n\n"
        f"Choose and rank the top {top_k} recommendations from these {len(group)} candidates. "
        f"Return exactly {top_k} recommendation objects.\n\n"
        + "\n\n".join(candidates)
    )


def build_verifier_prompt(
    group: pd.DataFrame,
    recommendations: list[dict[str, Any]],
    top_k: int,
    verify_pass: int,
) -> str:
    seed = group.iloc[0]
    by_candidate = {str(row["candidate_id"]): row for _, row in group.iterrows()}
    current = []
    for item in recommendations:
        candidate = by_candidate[item["candidate_id"]]
        current.append(
            "\n".join(
                [
                    f"Rank: {item['rank']}",
                    f"Candidate ID: {item['candidate_id']}",
                    f"Title: {candidate['candidate_title']}",
                    f"Current label: {item['similarity_label']}",
                    f"Current score: {item['similarity_score_10']}",
                    f"Current reason: {item['reason']}",
                    f"Genres: {pipe_to_commas(candidate['candidate_genres'])}",
                    f"Tags: {top_items(candidate['candidate_tags'])}",
                    f"Description: {shorten(candidate['candidate_description'])}",
                ]
            )
        )

    candidates = []
    for _, row in group.iterrows():
        candidates.append(
            "\n".join(
                [
                    f"Candidate ID: {row['candidate_id']}",
                    f"Title: {row['candidate_title']}",
                    f"Genres: {pipe_to_commas(row['candidate_genres'])}",
                    f"Tags: {top_items(row['candidate_tags'])}",
                    f"Description: {shorten(row['candidate_description'])}",
                ]
            )
        )

    return (
        f"Verifier pass: {verify_pass}\n\n"
        f"Seed anime:\n"
        f"ID: {seed['seed_id']}\n"
        f"Title: {seed['seed_title']}\n"
        f"Genres: {pipe_to_commas(seed['seed_genres'])}\n"
        f"Tags: {top_items(seed['seed_tags'])}\n"
        f"Description: {shorten(seed['seed_description'])}\n\n"
        f"Current ranked list to audit:\n\n"
        + "\n\n".join(current)
        + f"\n\nFull candidate pool you may use for replacements:\n\n"
        + "\n\n".join(candidates)
        + f"\n\nReturn exactly {top_k} verified recommendations."
    )


def build_output_rows(group: pd.DataFrame, recommendations: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    seed = group.iloc[0]
    by_candidate = {str(row["candidate_id"]): row for _, row in group.iterrows()}
    rows = []
    for item in recommendations[:top_k]:
        candidate = by_candidate[item["candidate_id"]]
        rows.append(
            {
                "seed_id": seed["seed_id"],
                "seed_title": seed["seed_title"],
                "rank": item["rank"],
                "recommended_id": candidate["candidate_id"],
                "recommended_title": candidate["candidate_title"],
                "similarity_label": item["similarity_label"],
                "similarity_score_10": item["similarity_score_10"],
                "reason": item["reason"],
                "seed_genres": seed["seed_genres"],
                "recommended_genres": candidate["candidate_genres"],
                "seed_tags": seed["seed_tags"],
                "recommended_tags": candidate["candidate_tags"],
            }
        )
    return rows


def clamp_label(score: int) -> int:
    return min(max(score, 0), 2)


def clamp_score_10(score: int) -> int:
    return min(max(score, 0), 10)


def normalize_label_from_score(label: int, score: int) -> int:
    if score >= 7:
        return 2
    if score >= 4:
        return 1
    return 0


def completed_seeds(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str).fillna("")
    return set(df["seed_id"].astype(str))


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).fillna("").to_dict("records")


def write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "seed_id",
        "seed_title",
        "rank",
        "recommended_id",
        "recommended_title",
        "similarity_label",
        "similarity_score_10",
        "reason",
        "seed_genres",
        "recommended_genres",
        "seed_tags",
        "recommended_tags",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def append_audit(path: Path, seed_id: str, seed_title: str, recommendations: list[dict[str, Any]]) -> None:
    record = {
        "seed_id": seed_id,
        "seed_title": seed_title,
        "recommendations": recommendations,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(path: Path) -> None:
    df = pd.read_csv(path)
    print(f"\nFinished. Wrote {len(df)} ranked recommendations to {path}")
    print(f"Seeds: {df['seed_id'].nunique()}")
    print("Similarity label counts:")
    print(df["similarity_label"].value_counts().sort_index().to_string())
    print("Similarity score /10 summary:")
    print(df["similarity_score_10"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
