import argparse
from pathlib import Path

import pandas as pd


GOLDEN_FILE = Path("data/golden/golden_dataset.csv")

INTENTS = [
    "delivery_problem",
    "order_management",
    "returns_refunds",
    "product_issue",
    "payment_billing",
    "amazon_pay",
    "account_access",
    "prime_membership",
    "digital_content",
    "product_information",
    "technical_issue",
    "support_followup",
    "other",
]


def is_blank(value):
    return str(value).strip() == "" or str(value).lower() == "nan"


def display(value):
    if is_blank(value):
        return "(none)"
    return str(value)


def get_intent():
    while True:
        print("\nChoose final HUMAN intent:\n")

        for i, intent in enumerate(INTENTS, start=1):
            print(f"{i:2}. {intent}")

        choice = input("\nYour choice: ").strip()

        try:
            number = int(choice)

            if 1 <= number <= len(INTENTS):
                return INTENTS[number - 1]

        except ValueError:
            pass

        print("Invalid choice. Enter a number from 1 to 13.")


def get_escalation():
    while True:
        print("\nShould this conversation be escalated?")
        print("  y = Yes")
        print("  n = No")

        choice = input("\nYour choice: ").strip().lower()

        if choice in {"y", "yes"}:
            return True

        if choice in {"n", "no"}:
            return False

        print("Invalid choice. Enter y or n.")


def get_note():
    print("\nOptional annotation note.")
    print("Press Enter to leave blank.")

    return input("Note: ").strip()


def annotate_row(df, index):
    row = df.iloc[index]

    print("\n")
    print("=" * 70)
    print("HUMAN GOLDEN SET ANNOTATION")
    print("=" * 70)

    print(f"\nExample {index + 1} / {len(df)}")

    print("\nTweet ID:")
    print(display(row["tweet_id"]))

    print("\nCustomer message:")
    print("-" * 70)
    print(display(row["customer_message"]))
    print("-" * 70)

    print("\nExisting information")
    print(f"Candidate intent: {display(row['candidate_intent'])}")

    print("\nAI suggestion")
    print(f"Intent:     {display(row['ai_intent'])}")
    print(f"Confidence: {display(row['ai_confidence'])}")
    print(f"Escalation: {display(row['ai_escalation'])}")
    print(f"Reason:     {display(row['ai_reason'])}")

    print("\nIMPORTANT:")
    print("The AI label is ONLY a suggestion.")
    print("Your decision becomes the ground truth.")

    gold_intent = get_intent()
    escalation = get_escalation()
    note = get_note()

    df.at[index, "gold_intent"] = gold_intent
    df.at[index, "expected_escalation"] = (
        "true" if escalation else "false"
    )
    df.at[index, "annotation_notes"] = note
    df.at[index, "annotation_status"] = "human_reviewed"

    return df


def find_next_unreviewed(df):
    for index, row in df.iterrows():
        status = str(row["annotation_status"]).strip().lower()

        if status != "human_reviewed":
            return index

    return None


def summarize(df):
    reviewed_mask = (
        df["annotation_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("human_reviewed")
    )

    reviewed = reviewed_mask.sum()

    print("\n")
    print("=" * 70)
    print("ANNOTATION PROGRESS")
    print("=" * 70)

    print(f"Human reviewed: {reviewed}/{len(df)}")
    print(f"Remaining: {len(df) - reviewed}")

    print("\nStatus distribution:")
    print(
        df["annotation_status"]
        .fillna("")
        .replace("", "unset")
        .value_counts()
        .to_string()
    )

    if reviewed > 0:
        reviewed_df = df[reviewed_mask]

        print("\nHuman-reviewed intent distribution:")
        print(
            reviewed_df["gold_intent"]
            .value_counts()
            .to_string()
        )

        print("\nHuman-reviewed escalation distribution:")
        print(
            reviewed_df["expected_escalation"]
            .value_counts()
            .to_string()
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Human annotation workflow for the Hiver golden set."
    )

    parser.add_argument(
        "--golden-file",
        type=Path,
        default=GOLDEN_FILE,
        help="Path to golden dataset CSV.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to annotate in this run.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.golden_file.exists():
        raise FileNotFoundError(
            f"Could not find golden dataset: {args.golden_file}"
        )

    df = pd.read_csv(args.golden_file)

    required_columns = [
        "ai_intent",
        "ai_escalation",
        "ai_confidence",
        "ai_reason",
        "annotation_status",
        "gold_intent",
        "expected_escalation",
        "annotation_notes",
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = ""

    df[required_columns] = (
        df[required_columns]
        .fillna("")
        .astype(str)
    )

    annotated_this_run = 0

    while True:
        index = find_next_unreviewed(df)

        if index is None:
            print("\nAll examples have been human-reviewed.")
            break

        if (
            args.limit is not None
            and annotated_this_run >= args.limit
        ):
            break

        try:
            df = annotate_row(df, index)

        except KeyboardInterrupt:
            print("\n\nAnnotation interrupted.")
            print("Progress has been saved.")
            break

        # Save after every human annotation.
        df.to_csv(
            args.golden_file,
            index=False,
        )

        annotated_this_run += 1

        print("\nSaved successfully.")

    summarize(df)


if __name__ == "__main__":
    main()