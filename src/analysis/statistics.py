from pathlib import Path

import pandas as pd

from src.data.conversations import build_conversations


DATA_FILE = Path("data/raw/twcs.csv")

CHUNK_SIZE = 100_000


def load_brand_data(brand_name: str) -> pd.DataFrame:
    """
    Load tweets relevant to a single brand.

    We keep:
    - brand tweets
    - customer tweets directly replying to brand tweets
    - customer tweets that have a direct brand response
    """

    # ---------------------------------------------------------
    # Pass 1: identify brand tweet IDs
    # ---------------------------------------------------------

    brand_tweet_ids = set()

    print(f"\nFinding tweets for {brand_name}...")

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
            & chunk["author_id"].eq(brand_name)
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
        f"brand tweets."
    )

    # ---------------------------------------------------------
    # Pass 2: collect brand + directly connected tweets
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
        # Brand tweets
        brand_mask = (
            ~chunk["inbound"]
            & chunk["author_id"].eq(brand_name)
        )

        # Customer -> brand
        parent_is_brand = (
            chunk["in_response_to_tweet_id"]
            .isin(brand_tweet_ids)
        )

        # Customer tweet has a brand response
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

    # A tweet can be discovered through more than one
    # relationship, so deduplicate by tweet_id.
    result = result.drop_duplicates(
        subset=["tweet_id"]
    ).reset_index(drop=True)

    return result


def analyze_brand(brand_name: str):
    """Calculate conversation-level statistics for a brand."""

    df = load_brand_data(brand_name)

    if df.empty:
        return {
            "brand": brand_name,
            "tweets": 0,
            "conversations": 0,
        }

    print(
        f"\nBuilding conversations for "
        f"{brand_name}..."
    )

    conversations = build_conversations(
        df,
        brand_name,
    )

    if not conversations:
        return {
            "brand": brand_name,
            "tweets": len(df),
            "conversations": 0,
        }

    turn_counts = pd.Series(
        [
            conversation["turn_count"]
            for conversation in conversations
        ]
    )

    return {
        "brand": brand_name,
        "tweets": len(df),
        "conversations": len(conversations),
        "avg_turns": round(
            turn_counts.mean(),
            2,
        ),
        "median_turns": float(
            turn_counts.median()
        ),
        "max_turns": int(
            turn_counts.max()
        ),
        "conversations_3plus": int(
            (turn_counts >= 3).sum()
        ),
        "conversations_6plus": int(
            (turn_counts >= 6).sum()
        ),
    }


def main():
    brands = [
        "AmazonHelp",
        "AppleSupport",
        "Uber_Support",
    ]

    results = []

    for brand in brands:
        result = analyze_brand(brand)
        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("CONVERSATION-LEVEL BRAND COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False
        )
    )

    output_file = Path(
        "data/processed/"
        "brand_conversation_statistics.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nSaved results to: {output_file}"
    )


if __name__ == "__main__":
    main()