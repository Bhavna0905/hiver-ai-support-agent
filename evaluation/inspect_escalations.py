from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "reports" / "end_to_end_results.json"


def main():
    with open(
        RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    print("ESCALATION CASES")
    print("=" * 100)

    for example in data["examples"]:
        gold = example["expected_escalation"]
        predicted = example["predicted_escalation"]

        if not gold and not predicted:
            continue

        print(f"Tweet: {example['tweet_id']}")
        print(f"Gold escalation: {gold}")
        print(f"Predicted escalation: {predicted}")
        print(f"Predicted intent: {example['predicted_intent']}")
        print(f"Confidence: {example['confidence']:.3f}")
        print(f"Reason: {example['escalation_reason']}")
        print(f"Message: {example['customer_message']}")
        print("-" * 100)


if __name__ == "__main__":
    main()