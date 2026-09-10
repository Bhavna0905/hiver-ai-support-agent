from collections import defaultdict

import pandas as pd

from src.data.load import iter_data


TOP_BRANDS = {
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
}


def analyze_conversation_depth(
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """
    Analyze conversation depth for the strongest brand candidates.

    A conversation is approximated by following the
    in_response_to_tweet_id relationship.

    The analysis focuses on conversations containing at least
    one inbound customer tweet and one outbound brand reply.
    """

    print("Loading tweets for candidate brands...")

    candidate_tweets = {}

    # First pass: collect relevant tweets.
    for chunk_number, chunk in enumerate(
        iter_data(
            chunksize=chunksize,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
                "text",
                "in_response_to_tweet_id",
            ],
        ),
        start=1,
    ):
        relevant = chunk[
            chunk["author_id"].isin(TOP_BRANDS)
            | chunk["inbound"]
        ]

        for row in relevant.itertuples(index=False):
            candidate_tweets[str(row.tweet_id)] = {
                "author_id": row.author_id,
                "inbound": bool(row.inbound),
                "text": str(row.text),
                "parent_id": (
                    None
                    if pd.isna(row.in_response_to_tweet_id)
                    else str(int(row.in_response_to_tweet_id))
                ),
            }

        if chunk_number % 5 == 0:
            print(
                f"Processed chunk {chunk_number}..."
            )

    print(
        f"\nCollected {len(candidate_tweets):,} relevant tweets."
    )

    # Build parent -> children relationships.
    children = defaultdict(list)

    for tweet_id, tweet in candidate_tweets.items():
        parent_id = tweet["parent_id"]

        if parent_id in candidate_tweets:
            children[parent_id].append(tweet_id)

    # Find conversations by starting from tweets without
    # a parent inside our relevant subset.
    conversation_lengths = defaultdict(list)

    visited = set()

    for tweet_id, tweet in candidate_tweets.items():

        if tweet_id in visited:
            continue

        # Only start from inbound customer tweets.
        if not tweet["inbound"]:
            continue

        parent_id = tweet["parent_id"]

        if parent_id in candidate_tweets:
            continue

        # Traverse the conversation.
        stack = [tweet_id]
        conversation = []

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            conversation.append(current)

            stack.extend(
                children.get(current, [])
            )

        if len(conversation) < 2:
            continue

        # Determine the brand(s) participating in the conversation.
        brands = {
            candidate_tweets[t]["author_id"]
            for t in conversation
            if not candidate_tweets[t]["inbound"]
            and candidate_tweets[t]["author_id"] in TOP_BRANDS
        }

        for brand in brands:
            conversation_lengths[brand].append(
                len(conversation)
            )

    rows = []

    for brand in TOP_BRANDS:
        lengths = conversation_lengths[brand]

        if not lengths:
            continue

        series = pd.Series(lengths)

        rows.append(
            {
                "brand": brand,
                "conversations": len(lengths),
                "avg_turns": round(series.mean(), 2),
                "median_turns": round(series.median(), 2),
                "max_turns": int(series.max()),
                "2_plus_turns": int((series >= 2).sum()),
                "4_plus_turns": int((series >= 4).sum()),
                "6_plus_turns": int((series >= 6).sum()),
                "10_plus_turns": int((series >= 10).sum()),
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values(
            "conversations",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return result


def main():
    result = analyze_conversation_depth()

    print("\nConversation depth comparison")
    print("============================")

    print(
        result.to_string(index=False)
    )


if __name__ == "__main__":
    main()