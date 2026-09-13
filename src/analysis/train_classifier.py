from pathlib import Path

import pandas as pd

from src.intent.classifier import IntentClassifier


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "processed" / "intent_training.csv"
MODEL_PATH = ROOT / "data" / "processed" / "intent_classifier.joblib"

MAX_PER_CLASS = 5_000
RANDOM_STATE = 42


def build_balanced_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cap large classes while retaining every example from smaller classes."""
    parts = []

    for intent, group in df.groupby("intent"):
        if len(group) > MAX_PER_CLASS:
            group = group.sample(
                n=MAX_PER_CLASS,
                random_state=RANDOM_STATE,
            )

        parts.append(group)

    balanced = pd.concat(parts, ignore_index=True)

    # Shuffle after combining classes.
    balanced = balanced.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return balanced


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {"tweet_id", "text", "intent"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    print(f"Original training examples: {len(df):,}")

    print("\nOriginal class distribution:")
    print(df["intent"].value_counts())

    balanced_df = build_balanced_training_data(df)

    print(
        f"\nBalanced training examples: "
        f"{len(balanced_df):,}"
    )

    print("\nBalanced class distribution:")
    print(balanced_df["intent"].value_counts())

    classifier = IntentClassifier()

    classifier.fit(
        balanced_df["text"].tolist(),
        balanced_df["intent"].tolist(),
    )

    classifier.save(MODEL_PATH)

    print("\nTraining complete.")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()