from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[1]

HUMAN_FILE = ROOT / "data" / "golden" / "human_response_review.csv"
JUDGE_FILE = ROOT / "reports" / "llm_judge_results.json"


DIMENSIONS = [
    "grounding",
    "resolution_correctness",
    "relevance",
    "safety",
    "style",
    "overall",
]


def main():
    human = pd.read_csv(HUMAN_FILE)

    with open(JUDGE_FILE, "r", encoding="utf-8") as f:
        judge_data = json.load(f)

    judge_rows = []

    for item in judge_data["results"]:
        row = {"tweet_id": item["tweet_id"]}

        for dimension in DIMENSIONS:
            row[dimension] = item["judgment"][dimension]

        judge_rows.append(row)

    judge = pd.DataFrame(judge_rows)

    merged = human.merge(
        judge,
        on="tweet_id",
        suffixes=("_human", "_judge"),
    )

    print("=" * 70)
    print("HUMAN vs LLM JUDGE AGREEMENT")
    print("=" * 70)
    print(f"Examples compared: {len(merged)}")
    print()

    agreement_rows = []

    for dimension in DIMENSIONS:
        human_scores = merged[f"{dimension}_human"]
        judge_scores = merged[f"{dimension}_judge"]

        exact_agreement = (
            human_scores == judge_scores
        ).mean()

        mean_absolute_difference = (
            human_scores - judge_scores
        ).abs().mean()

        kappa = cohen_kappa_score(
            human_scores,
            judge_scores,
            weights="quadratic",
        )

        agreement_rows.append(
            {
                "dimension": dimension,
                "exact_agreement": exact_agreement,
                "mean_absolute_difference": mean_absolute_difference,
                "quadratic_weighted_kappa": kappa,
            }
        )

        print(f"{dimension}")
        print(f"  Exact agreement:       {exact_agreement:.3f}")
        print(f"  Mean absolute diff:    {mean_absolute_difference:.3f}")
        print(f"  Quadratic weighted κ:  {kappa:.3f}")
        print()

    output = pd.DataFrame(agreement_rows)

    output_path = ROOT / "reports" / "human_agreement.json"

    output.to_json(
        output_path,
        orient="records",
        indent=2,
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()