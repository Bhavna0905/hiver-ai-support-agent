from pathlib import Path

import pandas as pd

from src.intent.classifier import IntentClassifier


TRAINING_DATA_PATH = Path(
    "data/processed/intent_training.csv"
)

MODEL_PATH = Path(
    "data/processed/intent_classifier.joblib"
)


def main():
    print("Loading training data...")

    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAINING_DATA_PATH}"
        )

    df = pd.read_csv(TRAINING_DATA_PATH)

    required_columns = {"text", "intent"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.dropna(
        subset=["text", "intent"]
    )

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df["intent"] = (
        df["intent"]
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["text"] != "")
        & (df["intent"] != "")
    ]

    print(f"Training examples: {len(df):,}")
    print(
        f"Intent classes: {df['intent'].nunique()}"
    )

    print("\nExamples per intent:")

    counts = (
        df["intent"]
        .value_counts()
        .sort_index()
    )

    for intent, count in counts.items():
        print(f"  {intent:<25} {count:,}")

    print("\nTraining classifier...")

    classifier = IntentClassifier()

    classifier.fit(
        df["text"],
        df["intent"],
    )

    classifier.save(MODEL_PATH)

    print("\nTraining complete.")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()