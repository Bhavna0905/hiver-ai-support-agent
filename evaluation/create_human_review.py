from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "reports" / "end_to_end_results.json"
OUTPUT_PATH = ROOT / "data" / "golden" / "human_response_review.csv"


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for example in data["examples"]:
        rows.append(
            {
                "tweet_id": example["tweet_id"],
                "customer_message": example["customer_message"],
                "response": example["response"],
                "grounding": "",
                "resolution_correctness": "",
                "relevance": "",
                "safety": "",
                "style": "",
                "overall": "",
                "human_note": "",
            }
        )

    df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("=" * 70)
    print("HUMAN RESPONSE REVIEW SET")
    print("=" * 70)
    print(f"Examples: {len(df)}")
    print(f"Saved to: {OUTPUT_PATH}")
    print()
    print("Rate each dimension from 1 to 5:")
    print("  1 = very poor")
    print("  3 = acceptable")
    print("  5 = excellent")


if __name__ == "__main__":
    main()