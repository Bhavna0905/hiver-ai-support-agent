from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.intent.classifier import IntentClassifier
from src.escalation.policy import EscalationPolicy


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "data" / "golden" / "golden_20.csv"
OUTPUT_PATH = ROOT / "reports" / "human_eval_results.json"


def evaluate_intent(df, classifier):
    predictions = []

    for message in df["customer_message"]:
        result = classifier.predict_with_confidence(message)
        predictions.append(result)

    df = df.copy()
    df["predicted_intent"] = [x["intent"] for x in predictions]
    df["confidence"] = [x["confidence"] for x in predictions]

    y_true = df["gold_intent"]
    y_pred = df["predicted_intent"]

    labels = sorted(set(y_true) | set(y_pred))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return df, {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ).tolist(),
        "confusion_labels": labels,
    }


def evaluate_escalation(df):
    policy = EscalationPolicy()

    predicted = []

    for _, row in df.iterrows():
        decision = policy.decide(
            customer_message=row["customer_message"],
            intent=row["predicted_intent"],
            confidence=row["confidence"],
            retrieved_examples=[],
        )

        predicted.append(decision["escalate"])

    y_true = df["expected_escalation"].astype(bool)
    y_pred = pd.Series(predicted, dtype=bool)

    false_auto_handle = ((y_true == True) & (y_pred == False)).sum()
    false_escalation = ((y_true == False) & (y_pred == True)).sum()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "false_auto_handle_rate": float(
            false_auto_handle / len(df)
        ),
        "false_escalation_rate": float(
            false_escalation / len(df)
        ),
        "predicted_escalations": int(y_pred.sum()),
        "actual_escalations": int(y_true.sum()),
    }


def main():
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden set not found: {GOLDEN_PATH}"
        )

    df = pd.read_csv(GOLDEN_PATH)

    required_columns = {
        "tweet_id",
        "customer_message",
        "gold_intent",
        "expected_escalation",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    classifier = IntentClassifier.load()

    evaluated_df, intent_results = evaluate_intent(
        df,
        classifier,
    )

    escalation_results = evaluate_escalation(evaluated_df)

    results = {
        "dataset": {
            "path": str(GOLDEN_PATH),
            "examples": len(df),
            "human_reviewed": True,
            "annotators": 1,
        },
        "intent": intent_results,
        "escalation": escalation_results,
        "examples": evaluated_df[
            [
                "tweet_id",
                "gold_intent",
                "predicted_intent",
                "confidence",
            ]
        ].to_dict(orient="records"),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 55)
    print("HUMAN-REVIEWED EVALUATION")
    print("=" * 55)

    print(f"\nExamples: {len(df)}")

    print("\nIntent Classification")
    print("-" * 25)
    print(f"Accuracy:   {intent_results['accuracy']:.3f}")
    print(f"Macro F1:   {intent_results['macro_f1']:.3f}")
    print(f"Weighted F1:{intent_results['weighted_f1']:.3f}")

    print("\nPer-intent results")
    print("-" * 25)

    for label in intent_results["confusion_labels"]:
        metrics = intent_results["classification_report"].get(
            label,
            {},
        )

        print(
            f"{label:22s} "
            f"P={metrics.get('precision', 0):.3f} "
            f"R={metrics.get('recall', 0):.3f} "
            f"F1={metrics.get('f1-score', 0):.3f}"
        )

    print("\nEscalation")
    print("-" * 25)
    print(f"Accuracy:              {escalation_results['accuracy']:.3f}")
    print(f"Precision:             {escalation_results['precision']:.3f}")
    print(f"Recall:                {escalation_results['recall']:.3f}")
    print(f"F1:                    {escalation_results['f1']:.3f}")
    print(
        "False auto-handle rate:"
        f" {escalation_results['false_auto_handle_rate']:.3f}"
    )
    print(
        "False escalation rate: "
        f"{escalation_results['false_escalation_rate']:.3f}"
    )

    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()