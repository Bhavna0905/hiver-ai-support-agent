from collections import defaultdict
from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/raw/twcs.csv")
CHUNK_SIZE = 100_000


def analyze_brands():
    """
    Analyze brand activity in the TWCS dataset.

    A brand is identified from outbound tweets (inbound=False).
    Inbound customer messages are associated with a brand through
    their direct response tweets.
    """

    outbound_counts = defaultdict(int)
    inbound_counts = defaultdict(int)

    # Map brand response tweet IDs -> brand.
    brand_tweet_to_brand = {}

    print(
        f"Reading dataset in chunks of {CHUNK_SIZE:,} rows..."
    )

    # ---------------------------------------------------------
    # Pass 1: identify brand tweets
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
        outbound_chunk = chunk[~chunk["inbound"]]

        for row in outbound_chunk.itertuples(index=False):
            brand = str(row.author_id)

            outbound_counts[brand] += 1
            brand_tweet_to_brand[int(row.tweet_id)] = brand

        print(
            f"Pass 1 - processed chunk {chunk_number}"
        )

    print(
        f"\nIdentified {len(outbound_counts)} potential brands."
    )

       # ---------------------------------------------------------
    # Pass 2: associate inbound tweets with brands
    # ---------------------------------------------------------

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_FILE,
            usecols=[
                "tweet_id",
                "inbound",
                "in_response_to_tweet_id",
                "response_tweet_id",
            ],
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        inbound_chunk = chunk[chunk["inbound"]]

        for row in inbound_chunk.itertuples(index=False):
            associated_brands = set()

            # -------------------------------------------------
            # Case 1:
            # Customer tweet directly replies to a brand tweet.
            # -------------------------------------------------

            parent_id = row.in_response_to_tweet_id

            if pd.notna(parent_id):
                try:
                    parent_id = int(parent_id)
                except (ValueError, TypeError):
                    parent_id = None

                if parent_id is not None:
                    brand = brand_tweet_to_brand.get(parent_id)

                    if brand is not None:
                        associated_brands.add(brand)

            # -------------------------------------------------
            # Case 2:
            # Customer tweet's response_tweet_id contains
            # a brand response.
            #
            # This captures relationships represented in the
            # opposite direction in the dataset.
            # -------------------------------------------------

            if pd.notna(row.response_tweet_id):
                response_ids = str(
                    row.response_tweet_id
                ).split(",")

                for response_id in response_ids:
                    response_id = response_id.strip()

                    if not response_id:
                        continue

                    try:
                        response_id = int(response_id)
                    except ValueError:
                        continue

                    brand = brand_tweet_to_brand.get(
                        response_id
                    )

                    if brand is not None:
                        associated_brands.add(brand)

            # -------------------------------------------------
            # Count the inbound interaction once per brand.
            # -------------------------------------------------

            for brand in associated_brands:
                inbound_counts[brand] += 1

        print(
            f"Pass 2 - processed chunk {chunk_number}"
        )
    # ---------------------------------------------------------
    # Build results
    # ---------------------------------------------------------

    brands = sorted(
        outbound_counts.keys()
    )

    results = []

    for brand in brands:
        inbound = inbound_counts.get(
            brand,
            0,
        )

        outbound = outbound_counts.get(
            brand,
            0,
        )

        results.append(
            {
                "brand": brand,
                "inbound": inbound,
                "outbound": outbound,
                "total": inbound + outbound,
            }
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        [
            "inbound",
            "outbound",
        ],
        ascending=False,
    )

    return results_df


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    results = analyze_brands()

    print("\n" + "=" * 70)
    print("BRAND ANALYSIS")
    print("=" * 70)

    print(
        results.head(20).to_string(
            index=False
        )
    )

    print(
        f"\nTotal brands identified: "
        f"{len(results)}"
    )

    output_file = Path(
        "data/processed/brand_statistics.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nSaved results to: {output_file}"
    )


if __name__ == "__main__":
    main()