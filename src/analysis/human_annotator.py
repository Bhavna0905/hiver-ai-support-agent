import csv
from pathlib import Path


INPUT_PATH = Path("data/golden/golden_20_candidates.csv")
OUTPUT_PATH = Path("data/golden/golden_20.csv")


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


def ask_intent():
    print("\nSelect intent:")

    for number, intent in enumerate(INTENTS, start=1):
        print(f"{number:2}. {intent}")

    while True:
        choice = input("\nEnter choice: ").strip()

        try:
            number = int(choice)

            if 1 <= number <= len(INTENTS):
                return INTENTS[number - 1]

        except ValueError:
            pass

        print("Invalid choice. Enter a number from 1 to 13.")


def ask_escalation():
    while True:
        answer = input(
            "\nShould this escalate to a human? [Y/N]: "
        ).strip().lower()

        if answer in {"y", "yes"}:
            return True

        if answer in {"n", "no"}:
            return False

        print("Please enter Y or N.")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        candidates = list(csv.DictReader(file))

    if len(candidates) != 20:
        raise ValueError(
            f"Expected 20 candidates, found {len(candidates)}."
        )

    print("=" * 70)
    print("HUMAN GOLDEN SET ANNOTATION")
    print("=" * 70)
    print(f"Examples: {len(candidates)}")
    print(f"Input:    {INPUT_PATH}")
    print(f"Output:   {OUTPUT_PATH}")
    print()
    print("You will manually label every example.")
    print("No LLM predictions are used.")
    print()
    print("Progress is saved after every example.")
    print("Press Ctrl+C to stop safely.")
    print()

    # ---------------------------------------------------------
    # Resume support
    # ---------------------------------------------------------
    completed = {}

    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            for row in csv.DictReader(file):
                tweet_id = row.get("tweet_id")

                if tweet_id:
                    completed[tweet_id] = row

        print(
            f"Found {len(completed)} previously annotated examples."
        )
        print("Those examples will be skipped.")
        print()

    output_fields = [
        "tweet_id",
        "customer_message",
        "candidate_intent",
        "gold_intent",
        "expected_escalation",
        "annotation_note",
    ]

    # ---------------------------------------------------------
    # Annotate each example
    # ---------------------------------------------------------
    for index, candidate in enumerate(candidates, start=1):

        tweet_id = candidate["tweet_id"]

        if tweet_id in completed:
            continue

        print("\n" + "=" * 70)
        print(f"Example {index}/{len(candidates)}")
        print("=" * 70)

        print("\nCustomer message:")
        print("-" * 70)
        print(candidate["customer_message"])
        print("-" * 70)

        print(
            f"\nCandidate intent from sampling: "
            f"{candidate['candidate_intent']}"
        )

        # Human chooses intent.
        gold_intent = ask_intent()

        # Human chooses escalation.
        expected_escalation = ask_escalation()

        annotation_note = input(
            "\nOptional annotation note "
            "(press Enter to skip): "
        ).strip()

        if not annotation_note:
            annotation_note = "Manually reviewed by human annotator."

        result = {
            "tweet_id": tweet_id,
            "customer_message": candidate["customer_message"],
            "candidate_intent": candidate["candidate_intent"],
            "gold_intent": gold_intent,
            "expected_escalation": str(expected_escalation),
            "annotation_note": annotation_note,
        }

        completed[tweet_id] = result

        # -----------------------------------------------------
        # Save immediately after every annotation
        # -----------------------------------------------------
        with OUTPUT_PATH.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=output_fields,
            )

            writer.writeheader()

            for row in completed.values():
                writer.writerow(row)

        print(
            f"\nSaved progress: "
            f"{len(completed)}/{len(candidates)} completed."
        )

    print("\n" + "=" * 70)
    print("ANNOTATION COMPLETE")
    print("=" * 70)
    print(f"Total examples: {len(candidates)}")
    print(f"Human-reviewed: {len(completed)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()