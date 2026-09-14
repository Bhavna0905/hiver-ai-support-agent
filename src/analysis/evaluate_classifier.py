from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[2]

GOLDEN_FILE = ROOT / "data" / "golden" / "golden_dataset.csv"
MODEL_FILE = ROOT / "data" / "processed" / "intent_classifier_minilm.joblib"
RESULTS_FILE = ROOT / "reports" / "intent_classifier_results.json"
PREDICTION_FILE = ROOT / "data" / "processed" / "intent_classifier_predictions.csv"

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

BATCH_SIZE = 64


def main():

    print("Loading golden dataset...")

    df = pd.read_csv(GOLDEN_FILE).fillna("")

    df = df[
        df["customer_message"].astype(str).str.strip().ne("")
        & df["gold_intent"].astype(str).str.strip().ne("")
    ].copy()

    print(f"Golden examples: {len(df)}")

    print("\nLoading hybrid classifier...")

    artifact = joblib.load(MODEL_FILE)

    encoder_name = artifact["encoder_name"]
    classifier = artifact["classifier"]
    tfidf = artifact["tfidf_vectorizer"]

    print(f"Embedding model: {encoder_name}")
    print(f"Feature type: {artifact['feature_type']}")

    messages = df["customer_message"].astype(str).tolist()
    y_true = df["gold_intent"].astype(str).tolist()

    print("\nLoading Sentence Transformer...")

    encoder = SentenceTransformer(encoder_name)

    print("\nGenerating MiniLM embeddings...")

    embeddings = encoder.encode(
        messages,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    print(f"MiniLM features: {embeddings.shape}")

    print("\nGenerating TF-IDF features...")

    tfidf_features = tfidf.transform(messages)

    print(f"TF-IDF features: {tfidf_features.shape}")

    print("\nCombining features...")

    combined_features = hstack(
        [
            csr_matrix(embeddings),
            tfidf_features,
        ],
        format="csr",
    )

    print(f"Combined features: {combined_features.shape}")

    print("\nGenerating predictions and probabilities...")

    probabilities = classifier.predict_proba(
        combined_features
    )

    class_names = classifier.classes_

    top_indices = np.argsort(
        probabilities,
        axis=1
    )[:, ::-1]

    y_pred = class_names[top_indices[:, 0]]

    confidence = probabilities[
        np.arange(len(probabilities)),
        top_indices[:, 0],
    ]

    second_confidence = probabilities[
        np.arange(len(probabilities)),
        top_indices[:, 1],
    ]

    second_intent = class_names[
        top_indices[:, 1]
    ]

    margin = confidence - second_confidence

    df["predicted_intent"] = y_pred
    df["prediction_confidence"] = confidence
    df["second_intent"] = second_intent
    df["second_confidence"] = second_confidence
    df["confidence_margin"] = margin

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=INTENTS,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=INTENTS,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=INTENTS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=INTENTS,
    )

    confusion = {
        actual: {
            predicted: int(matrix[i][j])
            for j, predicted in enumerate(INTENTS)
        }
        for i, actual in enumerate(INTENTS)
    }

    # ---------------------------------------------------------
    # Correct vs incorrect confidence
    # ---------------------------------------------------------

    correct = df["gold_intent"] == df["predicted_intent"]

    correct_confidence = float(
        df.loc[correct, "prediction_confidence"].mean()
    )

    incorrect_confidence = float(
        df.loc[~correct, "prediction_confidence"].mean()
    )

    correct_margin = float(
        df.loc[correct, "confidence_margin"].mean()
    )

    incorrect_margin = float(
        df.loc[~correct, "confidence_margin"].mean()
    )

    # ---------------------------------------------------------
    # Confidence buckets
    # ---------------------------------------------------------

    bins = [0.0, 0.50, 0.70, 0.85, 1.01]
    labels = [
        "<0.50",
        "0.50-0.70",
        "0.70-0.85",
        ">=0.85",
    ]

    df["confidence_bucket"] = pd.cut(
        df["prediction_confidence"],
        bins=bins,
        labels=labels,
        right=False,
    )

    bucket_results = {}

    for bucket in labels:

        subset = df[
            df["confidence_bucket"] == bucket
        ]

        if len(subset) == 0:
            continue

        bucket_results[bucket] = {
            "examples": int(len(subset)),
            "accuracy": float(
                (
                    subset["gold_intent"]
                    == subset["predicted_intent"]
                ).mean()
            ),
            "mean_confidence": float(
                subset["prediction_confidence"].mean()
            ),
        }

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    PREDICTION_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        PREDICTION_FILE,
        index=False,
        encoding="utf-8",
    )

    results = {
        "dataset": {
            "golden_examples": len(df),
            "intents": INTENTS,
        },
        "model": {
            "embedding_model": encoder_name,
            "feature_type": artifact["feature_type"],
            "feature_dimensions": int(
                combined_features.shape[1]
            ),
        },
        "overall": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        },
        "confidence_analysis": {
            "correct_mean_confidence": correct_confidence,
            "incorrect_mean_confidence": incorrect_confidence,
            "correct_mean_margin": correct_margin,
            "incorrect_mean_margin": incorrect_margin,
            "buckets": bucket_results,
        },
        "per_intent": report,
        "confusion_matrix": confusion,
    }

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("HYBRID CLASSIFIER + CONFIDENCE ANALYSIS")
    print("=" * 70)

    print(f"\nAccuracy:    {accuracy:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")

    print("\nConfidence comparison:")

    print(
        f"  Correct predictions:   "
        f"{correct_confidence:.3f}"
    )

    print(
        f"  Incorrect predictions: "
        f"{incorrect_confidence:.3f}"
    )

    print("\nMargin comparison:")

    print(
        f"  Correct predictions:   "
        f"{correct_margin:.3f}"
    )

    print(
        f"  Incorrect predictions: "
        f"{incorrect_margin:.3f}"
    )

    print("\nConfidence buckets:")

    for bucket, values in bucket_results.items():

        print(
            f"  {bucket:10s} "
            f"n={values['examples']:3d} "
            f"accuracy={values['accuracy']:.3f} "
            f"mean_conf={values['mean_confidence']:.3f}"
        )

    print("\nSaved predictions:")
    print(PREDICTION_FILE)

    print("\nSaved results:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()
