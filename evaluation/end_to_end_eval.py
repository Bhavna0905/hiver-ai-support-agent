from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from src.escalation.policy import EscalationPolicy
from src.generation.response_generator import ResponseGenerator
from src.retrieval.retrieve import ResolutionRetriever


ROOT = Path(__file__).resolve().parents[1]

GOLDEN_PATH = ROOT / "data" / "golden" / "golden_dataset.csv"
MODEL_PATH = ROOT / "data" / "processed" / "intent_classifier_minilm.joblib"
OUTPUT_PATH = ROOT / "reports" / "end_to_end_results.json"


def load_classifier():
    artifact = joblib.load(MODEL_PATH)

    encoder = SentenceTransformer(
        artifact["encoder_name"]
    )

    return encoder, artifact["classifier"]


def classify_messages(df, encoder, classifier):
    embeddings = encoder.encode(
        df["customer_message"].astype(str).tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    predictions = classifier.predict(embeddings)
    probabilities = classifier.predict_proba(embeddings)
    confidences = probabilities.max(axis=1)

    return predictions, confidences


def main():
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden set not found: {GOLDEN_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Semantic classifier not found: {MODEL_PATH}"
        )

    df = pd.read_csv(GOLDEN_PATH)

    print("=" * 70)
    print("END-TO-END SUPPORT AGENT EVALUATION")
    print("=" * 70)

    print(f"\nExamples: {len(df)}")

    # ---------------------------------------------------------
    # 1. Intent classification
    # ---------------------------------------------------------

    print("\nLoading semantic classifier...")

    encoder, classifier = load_classifier()

    print("\nClassifying messages...")

    predicted_intents, confidences = classify_messages(
        df,
        encoder,
        classifier,
    )

    df["predicted_intent"] = predicted_intents
    df["confidence"] = confidences

    # ---------------------------------------------------------
    # 2. Retrieval
    # ---------------------------------------------------------

    print("\nLoading resolution retriever...")

    retriever = ResolutionRetriever()

    # ---------------------------------------------------------
    # 3. Generation
    # ---------------------------------------------------------

    generator = ResponseGenerator()

    # ---------------------------------------------------------
    # 4. Escalation
    # ---------------------------------------------------------

    escalation_policy = EscalationPolicy()

    results = []

    print("\nRunning full agent pipeline...")

    for _, row in df.iterrows():
        message = row["customer_message"]
        intent = row["predicted_intent"]
        confidence = float(row["confidence"])

        retrieved = retriever.retrieve(
            message,
            top_k=3,
        )

        generation = generator.generate(
            customer_message=message,
            intent=intent,
            historical_examples=retrieved,
        )

        escalation = escalation_policy.decide(
            customer_message=message,
            intent=intent,
            confidence=confidence,
            retrieved_examples=retrieved,
        )

        results.append(
            {
                "tweet_id": int(row["tweet_id"]),
                "customer_message": message,
                "gold_intent": row["gold_intent"],
                "predicted_intent": intent,
                "confidence": confidence,
                "expected_escalation": bool(
                    row["expected_escalation"]
                ),
                "predicted_escalation": bool(
                    escalation["escalate"]
                ),
                "escalation_reason": escalation["reason"],
                "response": generation["response"],
                "evidence": retrieved,
            }
        )

    # ---------------------------------------------------------
    # 5. Metrics
    # ---------------------------------------------------------

    y_true_intent = [
        x["gold_intent"]
        for x in results
    ]

    y_pred_intent = [
        x["predicted_intent"]
        for x in results
    ]

    intent_accuracy = accuracy_score(
        y_true_intent,
        y_pred_intent,
    )

    intent_macro_f1 = f1_score(
        y_true_intent,
        y_pred_intent,
        average="macro",
        zero_division=0,
    )

    intent_weighted_f1 = f1_score(
        y_true_intent,
        y_pred_intent,
        average="weighted",
        zero_division=0,
    )

    y_true_escalation = [
        x["expected_escalation"]
        for x in results
    ]

    y_pred_escalation = [
        x["predicted_escalation"]
        for x in results
    ]

    escalation_accuracy = accuracy_score(
        y_true_escalation,
        y_pred_escalation,
    )

    escalation_precision = (
        sum(
            p and t
            for p, t in zip(
                y_pred_escalation,
                y_true_escalation,
            )
        )
        / max(sum(y_pred_escalation), 1)
    )

    escalation_recall = (
        sum(
            p and t
            for p, t in zip(
                y_pred_escalation,
                y_true_escalation,
            )
        )
        / max(sum(y_true_escalation), 1)
    )

    if (
        escalation_precision + escalation_recall
        > 0
    ):
        escalation_f1 = (
            2
            * escalation_precision
            * escalation_recall
            / (
                escalation_precision
                + escalation_recall
            )
        )
    else:
        escalation_f1 = 0.0

    false_auto_handle = sum(
        t and not p
        for t, p in zip(
            y_true_escalation,
            y_pred_escalation,
        )
    )

    false_escalation = sum(
        not t and p
        for t, p in zip(
            y_true_escalation,
            y_pred_escalation,
        )
    )

    total = len(results)

    output = {
        "dataset": {
            "examples": total,
            "human_reviewed": True,
            "annotators": 1,
        },
        "intent": {
            "accuracy": float(intent_accuracy),
            "macro_f1": float(intent_macro_f1),
            "weighted_f1": float(intent_weighted_f1),
            "classification_report": classification_report(
                y_true_intent,
                y_pred_intent,
                output_dict=True,
                zero_division=0,
            ),
        },
        "escalation": {
            "accuracy": float(escalation_accuracy),
            "precision": float(escalation_precision),
            "recall": float(escalation_recall),
            "f1": float(escalation_f1),
            "false_auto_handle_rate": float(
                false_auto_handle / total
            ),
            "false_escalation_rate": float(
                false_escalation / total
            ),
        },
        "examples": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\nIntent")
    print("-" * 30)
    print(f"Accuracy:    {intent_accuracy:.3f}")
    print(f"Macro F1:    {intent_macro_f1:.3f}")
    print(f"Weighted F1: {intent_weighted_f1:.3f}")

    print("\nEscalation")
    print("-" * 30)
    print(f"Accuracy:              {escalation_accuracy:.3f}")
    print(f"Precision:             {escalation_precision:.3f}")
    print(f"Recall:                {escalation_recall:.3f}")
    print(f"F1:                    {escalation_f1:.3f}")
    print(
        "False auto-handle rate:"
        f" {false_auto_handle / total:.3f}"
    )
    print(
        "False escalation rate:"
        f" {false_escalation / total:.3f}"
    )

    print(
        f"\nSaved results to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()