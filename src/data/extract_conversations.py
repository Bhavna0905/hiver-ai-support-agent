import json
from pathlib import Path

import pandas as pd

from src.data.conversations import build_conversations


DATA_FILE = Path("data/raw/twcs.csv")
OUTPUT_FILE = Path(
    "data/processed/amazonhelp_conversations.jsonl"
)

BRAND_NAME = "AmazonHelp"
CHUNK_SIZE = 100_000


def load_brand_data():
    """Extract tweets relevant to AmazonHelp."""

    print(f"Finding {BRAND_NAME} tweets...")

    brand_tweet_ids = set()

    # ---------------------------------------------------------
    # Pass 1: find AmazonHelp tweet IDs
    # ---------------------------------------------------------

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_FILE,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
            ],
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        mask = (
            ~chunk["inbound"]
            & chunk["author_id"].eq(BRAND_NAME)
        )

        brand_tweet_ids.update(
            chunk.loc[mask, "tweet_id"]
            .astype(int)
            .tolist()
        )

        print(
            f"Pass 1 - chunk {chunk_number}"
        )

    print(
        f"Found {len(brand_tweet_ids):,} "
        f"AmazonHelp tweets."
    )

    # ---------------------------------------------------------
    # Pass 2: extract directly connected tweets
    # ---------------------------------------------------------

    relevant_chunks = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_FILE,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
                "created_at",
                "text",
                "response_tweet_id",
                "in_response_to_tweet_id",
            ],
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        brand_mask = (
            ~chunk["inbound"]
            & chunk["author_id"].eq(BRAND_NAME)
        )

        parent_is_brand = (
            chunk["in_response_to_tweet_id"]
            .isin(brand_tweet_ids)
        )

        has_brand_response = chunk[
            "response_tweet_id"
        ].fillna("").apply(
            lambda value: any(
                int(response_id.strip())
                in brand_tweet_ids
                for response_id in str(value).split(",")
                if response_id.strip().isdigit()
            )
        )

        relevant_mask = (
            brand_mask
            | parent_is_brand
            | has_brand_response
        )

        relevant = chunk[relevant_mask]

        if not relevant.empty:
            relevant_chunks.append(relevant)

        print(
            f"Pass 2 - chunk {chunk_number}"
        )

    if not relevant_chunks:
        return pd.DataFrame()

    result = pd.concat(
        relevant_chunks,
        ignore_index=True,
    )

    result = result.drop_duplicates(
        subset=["tweet_id"]
    ).reset_index(drop=True)

    return result


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    print("Loading AmazonHelp interaction data...")

    df = load_brand_data()

    print(
        f"\nRelevant tweets: {len(df):,}"
    )

    print("Reconstructing conversations...")

    conversations = build_conversations(
        df,
        BRAND_NAME,
    )

    print(
        f"Reconstructed conversations: "
        f"{len(conversations):,}"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for conversation in conversations:
            json.dump(
                conversation,
                file,
                ensure_ascii=False,
                default=str,
            )

            file.write("\n")

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()