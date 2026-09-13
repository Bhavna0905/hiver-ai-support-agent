from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[1]

GOLDEN_PATH = ROOT / "data" / "golden" / "golden_20.csv"
MODEL_PATH = ROOT / "data" / "processed" / "intent_classifier_minilm.joblib"


def main():
    df = pd.read_csv(GOLDEN_PATH)

    artifact = joblib.load(MODEL_PATH)

    encoder_name = artifact["encoder_name"]
    classifier = artifact["classifier"]

    print("=" * 65)
    print("SEMANTIC CLASSIFIER EVALUATION")
    print("=" * 65)

    print(f"\nExamples: {len(df)}")
    print(f"Encoder: {encoder_name}")

    encoder = SentenceTransformer(encoder_name)

    print("\nCreating evaluation embeddings...")

    embeddings = encoder.encode(
        df["customer_message"].astype(str).tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    predictions = classifier.predict(embeddings)

    probabilities = classifier.predict_proba(embeddings)
    confidences = probabilities.max(axis=1)

    df["predicted_intent"] = predictions
    df["confidence"] = confidences

    accuracy = accuracy_score(
        df["gold_intent"],
        predictions,
    )

    macro_f1 = f1_score(
        df["gold_intent"],
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        df["gold_intent"],
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\nResults")
    print("-" * 25)
    print(f"Accuracy:    {accuracy:.3f}")
    print(f"Macro F1:    {macro_f1:.3f}")
    print(f"Weighted F1: {weighted_f1:.3f}")

    print("\nPer-intent results")
    print("-" * 65)

    report = classification_report(
        df["gold_intent"],
        predictions,
        output_dict=True,
        zero_division=0,
    )

    for intent in sorted(
        set(df["gold_intent"]) | set(predictions)
    ):
        metrics = report.get(intent, {})

        print(
            f"{intent:22s} "
            f"P={metrics.get('precision', 0):.3f} "
            f"R={metrics.get('recall', 0):.3f} "
            f"F1={metrics.get('f1-score', 0):.3f}"
        )

    print("\nPredictions")
    print("-" * 65)

    print(
        df[
            [
                "tweet_id",
                "gold_intent",
                "predicted_intent",
                "confidence",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()