import pandas as pd
from pathlib import Path

from src.config import PROCESSED_DATA_DIR


INPUT_FILE = PROCESSED_DATA_DIR / "golden_candidates.csv"
OUTPUT_FILE = Path("data/golden/golden_dataset.csv")


TARGET_COUNTS = {
    "delivery_problem": 30,
    "order_management": 20,
    "returns_refunds": 20,
    "product_issue": 15,
    "payment_billing": 15,
    "amazon_pay": 10,
    "account_access": 15,
    "prime_membership": 15,
    "digital_content": 15,
    "product_information": 10,
    "technical_issue": 15,
    "support_followup": 10,
    "other": 10,
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing candidate file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    selected = []

    for intent, count in TARGET_COUNTS.items():

        bucket = df[
            df["candidate_intent"] == intent
        ].copy()

        if len(bucket) < count:
            raise ValueError(
                f"Not enough candidates for {intent}: "
                f"need {count}, found {len(bucket)}"
            )

        # Deterministic sampling.
        sample = bucket.sample(
            n=count,
            random_state=42,
        )

        selected.append(sample)

    golden = pd.concat(
        selected,
        ignore_index=True,
    )

    # Shuffle final dataset.
    golden = golden.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    # Create human annotation columns.
    golden["gold_intent"] = ""
    golden["expected_escalation"] = ""
    golden["annotation_notes"] = ""

    # Keep only fields we actually need.
    golden = golden[
        [
            "conversation_id",
            "tweet_id",
            "customer_message",
            "turn_count",
            "candidate_intent",
            "gold_intent",
            "expected_escalation",
            "annotation_notes",
        ]
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    golden.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Created golden annotation file: "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Examples: {len(golden)}"
    )

    print("\nCandidate distribution:")
    print(
        golden["candidate_intent"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()