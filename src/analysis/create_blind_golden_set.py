from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "golden" / "golden_dataset.csv"
OUTPUT_PATH = ROOT / "data" / "golden" / "golden_200_blind.csv"


COLUMNS = [
    "tweet_id",
    "customer_message",
    "turn_count",
]


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input golden candidate file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    missing = set(COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Keep exactly the information a human annotator
    # should see. Do not expose candidate/model labels.
    blind = df[COLUMNS].copy()

    if len(blind) != 200:
        raise ValueError(
            f"Expected exactly 200 examples, found {len(blind)}"
        )

    # Add empty annotation fields.
    blind["gold_intent"] = ""
    blind["expected_escalation"] = ""
    blind["annotation_note"] = ""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    blind.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 60)
    print("BLIND GOLDEN SET CREATED")
    print("=" * 60)

    print(f"\nExamples: {len(blind)}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nColumns visible to annotator:")
    print("  tweet_id")
    print("  customer_message")
    print("  turn_count")
    print("  gold_intent")
    print("  expected_escalation")
    print("  annotation_note")

    print("\nModel/candidate labels are NOT included.")


if __name__ == "__main__":
    main()