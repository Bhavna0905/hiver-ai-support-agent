from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = ROOT / "reports" / "end_to_end_results.json"
OUTPUT_FILE = ROOT / "reports" / "failure_analysis.csv"


def main():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for item in data["examples"]:
        gold_intent = item["gold_intent"]
        predicted_intent = item["predicted_intent"]

        expected_escalation = item["expected_escalation"]
        predicted_escalation = item["predicted_escalation"]

        intent_failure = gold_intent != predicted_intent
        escalation_failure = expected_escalation != predicted_escalation

        response = item["response"]

        rows.append(
            {
                "tweet_id": item["tweet_id"],
                "customer_message": item["customer_message"],
                "gold_intent": gold_intent,
                "predicted_intent": predicted_intent,
                "intent_failure": intent_failure,
                "confidence": item["confidence"],
                "expected_escalation": expected_escalation,
                "predicted_escalation": predicted_escalation,
                "escalation_failure": escalation_failure,
                "escalation_reason": item["escalation_reason"],
                "response": response,
                "response_length": len(response),
            }
        )

    df = pd.DataFrame(rows)

    print("=" * 70)
    print("FAILURE ANALYSIS")
    print("=" * 70)

    print(f"Examples: {len(df)}")
    print()

    print("Intent failures:")
    print(f"  {df['intent_failure'].sum()} / {len(df)}")

    print("Escalation failures:")
    print(f"  {df['escalation_failure'].sum()} / {len(df)}")
    print()

    print("-" * 70)
    print("INTENT MISCLASSIFICATIONS")
    print("-" * 70)

    intent_failures = df[df["intent_failure"]]

    for _, row in intent_failures.iterrows():
        print()
        print(f"Tweet: {row['tweet_id']}")
        print(f"Gold:  {row['gold_intent']}")
        print(f"Pred:  {row['predicted_intent']}")
        print(f"Confidence: {row['confidence']:.3f}")
        print(f"Message: {row['customer_message']}")

    print()
    print("-" * 70)
    print("ESCALATION MISCLASSIFICATIONS")
    print("-" * 70)

    escalation_failures = df[df["escalation_failure"]]

    if escalation_failures.empty:
        print("No escalation errors.")

    for _, row in escalation_failures.iterrows():
        print()
        print(f"Tweet: {row['tweet_id']}")
        print(f"Expected: {row['expected_escalation']}")
        print(f"Predicted: {row['predicted_escalation']}")
        print(f"Reason: {row['escalation_reason']}")
        print(f"Message: {row['customer_message']}")

    print()
    print("-" * 70)
    print("LOW-CONFIDENCE PREDICTIONS")
    print("-" * 70)

    low_confidence = df[df["confidence"] < 0.60]

    for _, row in low_confidence.iterrows():
        print()
        print(f"Tweet: {row['tweet_id']}")
        print(f"Gold: {row['gold_intent']}")
        print(f"Pred: {row['predicted_intent']}")
        print(f"Confidence: {row['confidence']:.3f}")
        print(f"Message: {row['customer_message']}")

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()