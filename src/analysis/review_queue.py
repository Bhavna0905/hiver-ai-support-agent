from pathlib import Path

import pandas as pd


GOLDEN_FILE = Path("data/golden/golden_dataset.csv")
REVIEW_FILE = Path("data/golden/review_queue.csv")

LOW_CONFIDENCE_THRESHOLD = 70


def is_blank(value):
    return str(value).strip() == ""


def normalize_bool(value):
    """Convert common CSV boolean representations to a real bool."""
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def build_review_reason(row):
    reasons = []

    # Check disagreement between the existing gold label
    # and the AI suggestion.
    if (
        not is_blank(row["ai_intent"])
        and not is_blank(row["gold_intent"])
        and str(row["ai_intent"]).strip()
        != str(row["gold_intent"]).strip()
    ):
        reasons.append("ai_gold_intent_disagreement")

    # Check disagreement between expected escalation
    # and the AI escalation prediction.
    if (
        not is_blank(row["ai_escalation"])
        and not is_blank(row["expected_escalation"])
        and normalize_bool(row["ai_escalation"])
        != normalize_bool(row["expected_escalation"])
    ):
        reasons.append("ai_gold_escalation_disagreement")

    # Check low-confidence AI predictions.
    confidence = pd.to_numeric(
        row["ai_confidence"],
        errors="coerce",
    )

    if pd.notna(confidence) and confidence < LOW_CONFIDENCE_THRESHOLD:
        reasons.append("low_ai_confidence")

    # Check whether the AI disagreed with the original
    # candidate label selected during sampling.
    if normalize_bool(row["ai_disagrees_with_candidate"]):
        reasons.append("candidate_disagreement")

    # These examples were previously accepted from AI
    # without individual human review.
    if row["annotation_status"] == "ai_bulk_accepted":
        reasons.append("bulk_accepted_not_human_reviewed")

    return ";".join(reasons)


def main():
    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {GOLDEN_FILE}"
        )

    df = pd.read_csv(GOLDEN_FILE).fillna("")

    required_columns = [
        "tweet_id",
        "customer_message",
        "candidate_intent",
        "gold_intent",
        "expected_escalation",
        "ai_intent",
        "ai_escalation",
        "ai_confidence",
        "ai_reason",
        "annotation_status",
        "ai_disagrees_with_candidate",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df["review_reason"] = df.apply(
        build_review_reason,
        axis=1,
    )

    review_df = df[
        df["review_reason"].str.strip().ne("")
    ].copy()

    review_columns = [
        "tweet_id",
        "customer_message",
        "candidate_intent",
        "gold_intent",
        "expected_escalation",
        "ai_intent",
        "ai_escalation",
        "ai_confidence",
        "ai_reason",
        "annotation_status",
        "review_reason",
    ]

    review_df = review_df[review_columns]

    review_df.to_csv(
        REVIEW_FILE,
        index=False,
    )

    print("=" * 70)
    print("Golden Set Review Queue")
    print("=" * 70)

    print(f"Total golden examples: {len(df)}")
    print(f"Examples needing review: {len(review_df)}")
    print(
        f"Examples not currently flagged: "
        f"{len(df) - len(review_df)}"
    )

    print("\nReview reasons:")

    reason_counts = {}

    for reasons in review_df["review_reason"]:
        for reason in reasons.split(";"):
            if reason:
                reason_counts[reason] = (
                    reason_counts.get(reason, 0) + 1
                )

    if reason_counts:
        for reason, count in sorted(
            reason_counts.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(f"  {reason}: {count}")
    else:
        print("  No review reasons found.")

    print("\nReview queue:")
    print(f"  {REVIEW_FILE}")


if __name__ == "__main__":
    main()