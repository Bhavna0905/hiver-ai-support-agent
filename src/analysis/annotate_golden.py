import argparse
from pathlib import Path

import pandas as pd

from src.analysis.llm_annotator import DEFAULT_CONFIDENCE_THRESHOLD
from src.analysis.llm_annotator import DEFAULT_MODEL
from src.analysis.llm_annotator import AnnotationError
from src.analysis.llm_annotator import annotate_with_ollama


GOLDEN_FILE = Path("data/golden/golden_dataset.csv")

ANNOTATION_COLUMNS = [
    "ai_intent",
    "ai_escalation",
    "ai_confidence",
    "ai_reason",
    "ai_source",
    "ai_disagrees_with_candidate",
    "ai_low_confidence",
    "annotation_status",
    "gold_intent",
    "expected_escalation",
    "annotation_notes",
]

BULK_ACCEPTED_NOTE = (
    "Bulk accepted from AI suggestion; not individually human-reviewed."
)


def is_blank(value):
    return str(value).strip() == ""


def bool_to_csv(value):
    return "true" if value else "false"


def ensure_annotation_columns(df):
    for column in ANNOTATION_COLUMNS:
        if column not in df.columns:
            df[column] = ""

        df[column] = df[column].fillna("").astype(str)

    for index, row in df.iterrows():
        if not is_blank(row["annotation_status"]):
            continue

        note = str(row["annotation_notes"])

        if BULK_ACCEPTED_NOTE in note:
            df.at[index, "annotation_status"] = "ai_bulk_accepted"
        elif not is_blank(row["gold_intent"]):
            df.at[index, "annotation_status"] = "human_reviewed"
        elif not is_blank(row["ai_intent"]):
            df.at[index, "annotation_status"] = "ai_suggested"


def has_ai_prediction(df, index):
    return (
        not is_blank(df.at[index, "ai_intent"])
        and not is_blank(df.at[index, "ai_escalation"])
        and not is_blank(df.at[index, "ai_confidence"])
    )


def save_prediction(df, index, suggestion):
    df.at[index, "ai_intent"] = suggestion.intent
    df.at[index, "ai_escalation"] = bool_to_csv(
        suggestion.escalation
    )
    df.at[index, "ai_confidence"] = str(
        suggestion.confidence
    )
    df.at[index, "ai_reason"] = suggestion.reason
    df.at[index, "ai_source"] = suggestion.source
    df.at[index, "ai_disagrees_with_candidate"] = bool_to_csv(
        suggestion.disagrees_with_candidate
    )
    df.at[index, "ai_low_confidence"] = bool_to_csv(
        suggestion.low_confidence
    )

    # Preserve the provenance of an existing gold annotation.
    current_status = str(
        df.at[index, "annotation_status"]
    ).strip()

    if current_status in {
        "human_reviewed",
        "ai_bulk_accepted",
    }:
        return

    df.at[index, "annotation_status"] = "ai_suggested"


def save_failure(df, index, error):
    df.at[index, "annotation_status"] = "ai_failed"
    df.at[index, "ai_reason"] = str(error)


def summarize_annotations(df):
    ai_predicted = (
        df["ai_intent"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
    gold_present = (
        df["gold_intent"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
    disagreement_count = (
        df["ai_disagrees_with_candidate"]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )
    low_confidence_count = (
        df["ai_low_confidence"]
        .fillna("")
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )

    confidences = pd.to_numeric(
        df["ai_confidence"],
        errors="coerce",
    )

    print("\nAnnotation summary")
    print(f"  Rows: {len(df)}")
    print(f"  AI predictions present: {ai_predicted}")
    print(f"  Gold labels present: {gold_present}")
    print(f"  Candidate disagreements: {disagreement_count}")
    print(f"  Low-confidence predictions: {low_confidence_count}")

    if confidences.notna().any():
        print(
            "  Confidence mean/min/max: "
            f"{confidences.mean():.1f}/"
            f"{confidences.min():.0f}/"
            f"{confidences.max():.0f}"
        )

    print("\nStatus counts:")
    print(
        df["annotation_status"]
        .fillna("")
        .replace("", "unset")
        .value_counts()
        .to_string()
    )

    print("\nAI intent distribution:")
    print(
        df["ai_intent"]
        .fillna("")
        .replace("", "unset")
        .value_counts()
        .to_string()
    )


def annotate_dataset(
    df,
    output_file,
    model,
    limit=None,
    force=False,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
):
    processed = 0
    skipped = 0
    failed = 0

    for index, row in df.iterrows():
        if limit is not None and processed >= limit:
            break

        if has_ai_prediction(df, index) and not force:
            skipped += 1
            continue

        print(
            f"Annotating row {index + 1}/{len(df)} "
            f"with {model}..."
        )

        try:
            suggestion = annotate_with_ollama(
                customer_message=row["customer_message"],
                candidate_intent=row["candidate_intent"],
                model=model,
                confidence_threshold=confidence_threshold,
            )
            save_prediction(df, index, suggestion)
            processed += 1
        except Exception as error:
            save_failure(df, index, error)
            failed += 1

        df.to_csv(
            output_file,
            index=False,
        )

    print("\nRun summary")
    print(f"  New/updated AI predictions: {processed}")
    print(f"  Skipped existing predictions: {skipped}")
    print(f"  Failed rows: {failed}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Automatically annotate the golden set using "
            "a local Ollama model."
        )
    )

    parser.add_argument(
        "--golden-file",
        type=Path,
        default=GOLDEN_FILE,
        help="Path to the golden CSV.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Ollama model to use. Can also be configured "
            "with OLLAMA_MODEL."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process this many rows that need AI prediction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run rows that already have AI predictions.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=int,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence below this value is flagged for review.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print annotation summary without calling Ollama.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.golden_file.exists():
        raise FileNotFoundError(
            f"Could not find {args.golden_file}"
        )

    df = pd.read_csv(args.golden_file)
    ensure_annotation_columns(df)

    # Save once after migration so annotation_status exists, while keeping
    # existing gold labels and notes untouched.
    df.to_csv(
        args.golden_file,
        index=False,
    )

    if not args.summary_only:
        annotate_dataset(
            df=df,
            output_file=args.golden_file,
            model=args.model,
            limit=args.limit,
            force=args.force,
            confidence_threshold=args.confidence_threshold,
        )

    summarize_annotations(df)


if __name__ == "__main__":
    main()
