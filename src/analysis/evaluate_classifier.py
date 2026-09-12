from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.intent.classifier import IntentClassifier


GOLDEN_FILE = Path(
    "data/golden/golden_dataset.csv"
)

MODEL_FILE = Path(
    "data/processed/intent_classifier.joblib"
)

RESULTS_FILE = Path(
    "reports/intent_classifier_results.json"
)

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


def main():
    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {GOLDEN_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Classifier model not found: {MODEL_FILE}"
        )

    print("Loading golden dataset...")

    df = pd.read_csv(
        GOLDEN_FILE
    ).fillna("")

    required_columns = [
        "customer_message",
        "gold_intent",
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
        df["customer_message"]
        .astype(str)
        .str.strip()
        .ne("")
        & df["gold_intent"]
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    print(
        f"Golden examples: {len(df)}"
    )

    print("\nLoading trained classifier...")

    classifier = IntentClassifier.load(
        MODEL_FILE
    )

    messages = (
        df["customer_message"]
        .astype(str)
        .tolist()
    )

    y_true = (
        df["gold_intent"]
        .astype(str)
        .tolist()
    )

    print("Generating predictions...")

    y_pred = classifier.predict(
        messages
    )

    y_pred = [
        str(prediction)
        for prediction in y_pred
    ]

    df["predicted_intent"] = y_pred

    # ---------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------

    overall_accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    overall_macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=INTENTS,
        average="macro",
        zero_division=0,
    )

    # ---------------------------------------------------------
    # Supported-class metrics
    #
    # "other" was not available in the weakly-labelled
    # training data, so the classifier cannot intentionally
    # learn that class.
    # ---------------------------------------------------------

    supported_intents = [
        intent
        for intent in INTENTS
        if intent != "other"
    ]

    supported_mask = [
        gold in supported_intents
        for gold in y_true
    ]

    supported_df = df[
        supported_mask
    ].copy()

    supported_y_true = (
        supported_df["gold_intent"]
        .tolist()
    )

    supported_y_pred = (
        supported_df["predicted_intent"]
        .tolist()
    )

    supported_accuracy = accuracy_score(
        supported_y_true,
        supported_y_pred,
    )

    supported_macro_f1 = f1_score(
        supported_y_true,
        supported_y_pred,
        labels=supported_intents,
        average="macro",
        zero_division=0,
    )

    # ---------------------------------------------------------
    # "other" analysis
    # ---------------------------------------------------------

    other_mask = (
        df["gold_intent"] == "other"
    )

    other_count = int(
        other_mask.sum()
    )

    other_predictions = (
        df.loc[
            other_mask,
            "predicted_intent",
        ]
        .value_counts()
        .to_dict()
    )

    # ---------------------------------------------------------
    # Per-intent report
    # ---------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=INTENTS,
        output_dict=True,
        zero_division=0,
    )

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=INTENTS,
    )

    confusion = {
        actual: {
            predicted: int(
                matrix[i][j]
            )
            for j, predicted in enumerate(INTENTS)
        }
        for i, actual in enumerate(INTENTS)
    }

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    prediction_file = Path(
        "data/processed/"
        "intent_classifier_predictions.csv"
    )

    prediction_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        prediction_file,
        index=False,
    )

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    results = {
        "dataset": {
            "golden_examples": len(df),
            "trained_intents": supported_intents,
            "all_intents": INTENTS,
        },
        "overall": {
            "accuracy": float(
                overall_accuracy
            ),
            "macro_f1": float(
                overall_macro_f1
            ),
        },
        "supported_intents_only": {
            "examples": len(
                supported_df
            ),
            "accuracy": float(
                supported_accuracy
            ),
            "macro_f1": float(
                supported_macro_f1
            ),
        },
        "other_analysis": {
            "gold_other_examples": other_count,
            "predictions_for_other_examples": {
                key: int(value)
                for key, value
                in other_predictions.items()
            },
        },
        "per_intent": report,
        "confusion_matrix": confusion,
    }

    import json

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("Intent Classifier Evaluation")
    print("=" * 70)

    print(
        f"\nOverall accuracy: "
        f"{overall_accuracy:.4f}"
    )

    print(
        f"Overall macro-F1: "
        f"{overall_macro_f1:.4f}"
    )

    print(
        "\nSupported intents only "
        "(excluding other):"
    )

    print(
        f"  Examples: "
        f"{len(supported_df)}"
    )

    print(
        f"  Accuracy: "
        f"{supported_accuracy:.4f}"
    )

    print(
        f"  Macro-F1: "
        f"{supported_macro_f1:.4f}"
    )

    print(
        "\n'other' examples:"
    )

    print(
        f"  Gold examples: "
        f"{other_count}"
    )

    print(
        "  Model predictions:"
    )

    for intent, count in sorted(
        other_predictions.items(),
        key=lambda item: -item[1],
    ):
        print(
            f"    {intent}: {count}"
        )

    print(
        "\nPer-intent results:"
    )

    for intent in INTENTS:
        if intent not in report:
            continue

        metrics = report[intent]

        print(
            f"  {intent:22s} "
            f"precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} "
            f"f1={metrics['f1-score']:.3f} "
            f"support={int(metrics['support'])}"
        )

    print(
        "\nSaved predictions to:"
    )

    print(
        f"  {prediction_file}"
    )

    print(
        "\nSaved results to:"
    )

    print(
        f"  {RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()