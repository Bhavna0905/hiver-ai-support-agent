from pathlib import Path

import pandas as pd

from src.intent.classifier import IntentClassifier


TRAINING_FILE = Path(
    "data/processed/intent_training.csv"
)

MODEL_FILE = Path(
    "data/processed/intent_classifier.joblib"
)


def main():
    if not TRAINING_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found: {TRAINING_FILE}"
        )

    print("Loading training data...")

    df = pd.read_csv(TRAINING_FILE)

    required_columns = [
        "text",
        "intent",
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

    df = df[
        df["text"].notna()
        & df["intent"].notna()
    ].copy()

    print(
        f"Training examples: {len(df):,}"
    )

    print(
        f"Intent classes: "
        f"{df['intent'].nunique()}"
    )

    print("\nExamples per intent:")

    print(
        df["intent"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTraining classifier...")

    classifier = IntentClassifier()

    classifier.fit(
        messages=df["text"],
        intents=df["intent"],
    )

    classifier.save(MODEL_FILE)

    print("\nTraining complete.")
    print(
        f"Model saved to: {MODEL_FILE}"
    )


if __name__ == "__main__":
    main()