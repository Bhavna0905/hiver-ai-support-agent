import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from evaluation.baselines import AlwaysNoEscalationBaseline
from evaluation.baselines import MajorityIntentBaseline
from evaluation.baselines import build_tfidf_escalation_baseline
from evaluation.baselines import build_tfidf_intent_baseline
from evaluation.metrics import binary_metrics
from evaluation.metrics import classification_metrics
from src.analysis.llm_annotator import INTENTS


GOLDEN_FILE = Path("data/golden/golden_dataset.csv")
OUTPUT_FILE = Path("reports/baseline_results.json")


def parse_bool(value):
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def load_golden_dataset(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find golden dataset: {path}"
        )

    df = pd.read_csv(path)

    required_columns = [
        "customer_message",
        "gold_intent",
        "expected_escalation",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    for column in required_columns:
        df[column] = df[column].fillna("").astype(str)

    labelled = df[
        (df["gold_intent"].str.strip() != "")
        & (df["expected_escalation"].str.strip() != "")
    ].copy()

    labelled["expected_escalation_bool"] = labelled[
        "expected_escalation"
    ].map(parse_bool)

    return labelled.reset_index(drop=True)


def cross_validated_predictions(
    texts,
    labels,
    build_model,
    n_splits=5,
):
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    predictions = [None] * len(texts)

    for train_index, test_index in splitter.split(
        texts,
        labels,
    ):
        model = build_model()

        train_texts = [
            texts[index]
            for index in train_index
        ]
        train_labels = [
            labels[index]
            for index in train_index
        ]
        test_texts = [
            texts[index]
            for index in test_index
        ]

        model.fit(train_texts, train_labels)
        fold_predictions = model.predict(test_texts)

        for index, prediction in zip(
            test_index,
            fold_predictions,
        ):
            predictions[index] = prediction

    return predictions


def evaluate(df):
    texts = df["customer_message"].tolist()
    intent_labels = df["gold_intent"].tolist()
    escalation_labels = df[
        "expected_escalation_bool"
    ].tolist()

    majority_intent_predictions = cross_validated_predictions(
        texts=texts,
        labels=intent_labels,
        build_model=MajorityIntentBaseline,
    )

    tfidf_intent_predictions = cross_validated_predictions(
        texts=texts,
        labels=intent_labels,
        build_model=build_tfidf_intent_baseline,
    )

    always_no_escalation_predictions = (
        AlwaysNoEscalationBaseline()
        .fit(texts, escalation_labels)
        .predict(texts)
    )

    tfidf_escalation_predictions = cross_validated_predictions(
        texts=texts,
        labels=escalation_labels,
        build_model=build_tfidf_escalation_baseline,
    )

    return {
        "dataset": {
            "rows": int(len(df)),
            "intent_distribution": {
                str(key): int(value)
                for key, value in df[
                    "gold_intent"
                ].value_counts().items()
            },
            "escalation_distribution": {
                str(key): int(value)
                for key, value in df[
                    "expected_escalation_bool"
                ].value_counts().items()
            },
        },
        "intent": {
            "trivial_majority": classification_metrics(
                intent_labels,
                majority_intent_predictions,
                labels=INTENTS,
            ),
            "tfidf_logistic_regression": classification_metrics(
                intent_labels,
                tfidf_intent_predictions,
                labels=INTENTS,
            ),
        },
        "escalation": {
            "trivial_always_no": binary_metrics(
                escalation_labels,
                always_no_escalation_predictions,
            ),
            "tfidf_logistic_regression": binary_metrics(
                escalation_labels,
                tfidf_escalation_predictions,
            ),
        },
    }


def print_summary(results):
    print("Dataset")
    print(f"  Rows: {results['dataset']['rows']}")

    print("\nIntent baselines")

    for name, metrics in results["intent"].items():
        print(
            f"  {name}: "
            f"accuracy={metrics['accuracy']}, "
            f"macro_f1={metrics['macro_f1']}"
        )

    print("\nEscalation baselines")

    for name, metrics in results["escalation"].items():
        print(
            f"  {name}: "
            f"accuracy={metrics['accuracy']}, "
            f"f1={metrics['f1']}, "
            f"false_auto_handle_rate="
            f"{metrics['false_auto_handle_rate']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate support-agent baselines on the "
            "golden dataset."
        )
    )

    parser.add_argument(
        "--golden-file",
        type=Path,
        default=GOLDEN_FILE,
        help="Path to the labelled golden CSV.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=OUTPUT_FILE,
        help="Where to save JSON results.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = load_golden_dataset(args.golden_file)
    results = evaluate(df)

    args.output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_file.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print_summary(results)
    print(f"\nSaved results to: {args.output_file}")


if __name__ == "__main__":
    main()
