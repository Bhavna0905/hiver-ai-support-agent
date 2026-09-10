from collections import defaultdict

import pandas as pd

from src.data.load import iter_data


CANDIDATE_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "response_tweet_id",
]

RESPONSE_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "in_response_to_tweet_id",
]


def find_brand_candidates(
    chunksize: int = 100_000,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Identify likely brand/support accounts from outbound tweets.

    In the TWCS dataset, inbound=False represents a message
    sent by a support account to a customer.
    """

    outbound_counts = defaultdict(int)

    total_rows = 0
    total_inbound = 0
    total_outbound = 0

    print("Scanning dataset for brand candidates...")

    for chunk_number, chunk in enumerate(
        iter_data(
            chunksize=chunksize,
            usecols=CANDIDATE_COLUMNS,
        ),
        start=1,
    ):
        total_rows += len(chunk)

        inbound_count = int(chunk["inbound"].sum())
        outbound_count = len(chunk) - inbound_count

        total_inbound += inbound_count
        total_outbound += outbound_count

        outbound = chunk[~chunk["inbound"]]

        counts = outbound["author_id"].value_counts()

        for author_id, count in counts.items():
            outbound_counts[author_id] += int(count)

        print(
            f"Processed chunk {chunk_number}: "
            f"{total_rows:,} rows"
        )

    candidates = (
        pd.DataFrame(
            [
                {
                    "brand": brand,
                    "outbound_tweets": count,
                }
                for brand, count in outbound_counts.items()
            ]
        )
        .sort_values(
            "outbound_tweets",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    print("\nDataset summary")
    print("----------------")
    print(f"Total rows:      {total_rows:,}")
    print(f"Inbound tweets:  {total_inbound:,}")
    print(f"Outbound tweets: {total_outbound:,}")

    return candidates


def analyze_brand_responses(
    candidates: pd.DataFrame,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """
    Analyze how candidate brands responded to customer messages.

    We use the explicit tweet relationship:
        brand tweet -> in_response_to_tweet_id -> customer tweet

    This is more reliable than assuming adjacent CSV rows belong
    to the same conversation.
    """

    candidate_brands = set(candidates["brand"])

    # Maps customer tweet ID -> brands that directly responded to it.
    customer_response_map = defaultdict(set)

    # Number of outbound responses by brand.
    outbound_counts = defaultdict(int)

    # Number of unique customer tweets directly answered by each brand.
    answered_customer_tweets = defaultdict(set)

    # Number of unique customers represented.
    customer_authors = defaultdict(set)

    print("\nBuilding customer-response relationships...")

    for chunk_number, chunk in enumerate(
        iter_data(
            chunksize=chunksize,
            usecols=RESPONSE_COLUMNS,
        ),
        start=1,
    ):
        outbound = chunk[
            (~chunk["inbound"])
            & chunk["author_id"].isin(candidate_brands)
        ]

        for row in outbound.itertuples(index=False):
            brand = row.author_id

            outbound_counts[brand] += 1

            if pd.notna(row.in_response_to_tweet_id):
                customer_tweet_id = str(
                    int(row.in_response_to_tweet_id)
                )

                answered_customer_tweets[brand].add(
                    customer_tweet_id
                )

        if chunk_number % 5 == 0:
            print(
                f"Processed response chunk {chunk_number}..."
            )

    print("\nCounting customer activity...")

    # Read inbound tweets and determine which brands directly
    # answered them using the relationship built above.
    answered_ids = set()

    for ids in answered_customer_tweets.values():
        answered_ids.update(ids)

    inbound_tweet_ids = set()

    for chunk_number, chunk in enumerate(
        iter_data(
            chunksize=chunksize,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
            ],
        ),
        start=1,
    ):
        inbound = chunk[chunk["inbound"]]

        for row in inbound.itertuples(index=False):
            tweet_id = str(row.tweet_id)

            if tweet_id in answered_ids:
                inbound_tweet_ids.add(tweet_id)

                # The exact brand will be assigned below using
                # the direct-response mapping.
                for brand, ids in answered_customer_tweets.items():
                    if tweet_id in ids:
                        customer_authors[brand].add(
                            str(row.author_id)
                        )

        if chunk_number % 5 == 0:
            print(
                f"Processed inbound chunk {chunk_number}..."
            )

    result = candidates.copy()

    result["answered_customer_tweets"] = (
        result["brand"]
        .map(
            lambda brand: len(
                answered_customer_tweets.get(brand, set())
            )
        )
        .fillna(0)
        .astype(int)
    )

    result["unique_customers"] = (
        result["brand"]
        .map(
            lambda brand: len(
                customer_authors.get(brand, set())
            )
        )
        .fillna(0)
        .astype(int)
    )

    result["responses_per_answered_customer_tweet"] = (
        result["outbound_tweets"]
        / result["answered_customer_tweets"].replace(0, pd.NA)
    )

    return result


def main():
    candidates = find_brand_candidates()

    print("\nTop brand candidates")
    print("--------------------")
    print(
        candidates.to_string(index=False)
    )

    results = analyze_brand_responses(candidates)

    print("\nBrand response analysis")
    print("----------------------")

    display_columns = [
        "brand",
        "outbound_tweets",
        "answered_customer_tweets",
        "unique_customers",
        "responses_per_answered_customer_tweet",
    ]

    print(
        results[display_columns]
        .to_string(index=False)
    )

    print("\nTop 10 candidates")
    print("-----------------")

    print(
        results[display_columns]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()