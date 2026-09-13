from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.intent.classifier import IntentClassifier


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "data" / "golden" / "golden_20.csv"

THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60]


def main():
    df = pd.read_csv(GOLDEN_PATH)

    classifier = IntentClassifier.load()

    predictions = []

    for message in df["customer_message"]:
        result = classifier.predict_with_confidence(message)

        predictions.append(
            {
                "intent": result["intent"],
                "confidence": result["confidence"],
            }
        )

    df["model_intent"] = [x["intent"] for x in predictions]
    df["confidence"] = [x["confidence"] for x in predictions]

    print("=" * 70)
    print("CONFIDENCE → OTHER THRESHOLD EXPERIMENT")
    print("=" * 70)

    print(
        "\nRule: confidence < threshold → other"
    )

    print(
        f"\n{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Macro-F1':<12}"
        f"{'Fallbacks':<12}"
    )

    print("-" * 48)

    for threshold in THRESHOLDS:
        adjusted_predictions = df["model_intent"].copy()

        adjusted_predictions[
            df["confidence"] < threshold
        ] = "other"

        accuracy = accuracy_score(
            df["gold_intent"],
            adjusted_predictions,
        )

        macro_f1 = f1_score(
            df["gold_intent"],
            adjusted_predictions,
            average="macro",
            zero_division=0,
        )

        fallbacks = (
            df["confidence"] < threshold
        ).sum()

        print(
            f"{threshold:<12.2f}"
            f"{accuracy:<12.3f}"
            f"{macro_f1:<12.3f}"
            f"{fallbacks:<12}"
        )

    print("\nLow-confidence examples")
    print("-" * 70)

    low_confidence = df.sort_values(
        "confidence"
    )[
        [
            "tweet_id",
            "gold_intent",
            "model_intent",
            "confidence",
        ]
    ]

    print(
        low_confidence.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()