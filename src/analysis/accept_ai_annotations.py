from pathlib import Path

import pandas as pd


GOLDEN_FILE = Path("data/golden/golden_dataset.csv")

BULK_NOTE = (
    "Bulk accepted from AI suggestion; not individually human-reviewed."
)


def is_blank(value):
    return str(value).strip() == ""


def accept_ai_annotations(df):
    required_columns = [
        "ai_intent",
        "ai_escalation",
        "gold_intent",
        "expected_escalation",
        "annotation_notes",
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

    accepted = 0
    skipped = 0

    for index, row in df.iterrows():
        if not is_blank(row["gold_intent"]):
            skipped += 1
            continue

        if is_blank(row["ai_intent"]):
            skipped += 1
            continue

        df.at[index, "gold_intent"] = row["ai_intent"]
        df.at[index, "expected_escalation"] = str(
            row["ai_escalation"]
        ).strip().lower()

        if is_blank(row["annotation_notes"]):
            df.at[index, "annotation_notes"] = BULK_NOTE

        accepted += 1

    return accepted, skipped


def main():
    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {GOLDEN_FILE}"
        )

    df = pd.read_csv(GOLDEN_FILE)

    for column in df.columns:
        df[column] = df[column].fillna("").astype(str)

    accepted, skipped = accept_ai_annotations(df)

    df.to_csv(
        GOLDEN_FILE,
        index=False,
    )

    labelled = (
        df["gold_intent"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    print(f"Bulk accepted AI suggestions: {accepted}")
    print(f"Skipped rows: {skipped}")
    print(f"Labelled rows now: {labelled}/{len(df)}")
    print(f"Saved: {GOLDEN_FILE}")


if __name__ == "__main__":
    main()
