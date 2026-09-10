from collections import defaultdict

import pandas as pd


def build_reply_maps(df: pd.DataFrame):
    """
    Build parent -> children and tweet_id -> row mappings.

    The TWCS dataset represents conversations using:
      - tweet_id
      - in_response_to_tweet_id
      - response_tweet_id
    """

    tweet_map = {
        row.tweet_id: row
        for row in df.itertuples(index=False)
    }

    children = defaultdict(list)

    for row in df.itertuples(index=False):
        parent_id = row.in_response_to_tweet_id

        if pd.notna(parent_id):
            children[int(parent_id)].append(row.tweet_id)

    return tweet_map, children


def get_thread(tweet_id, tweet_map, children):
    """
    Follow the direct reply chain starting from a tweet.

    Important:
    We follow one reply path rather than merging every child of a tweet.
    This prevents unrelated customers replying to the same brand tweet
    from becoming one conversation.
    """

    thread = []
    current_id = tweet_id
    visited = set()

    while current_id in tweet_map and current_id not in visited:
        visited.add(current_id)

        row = tweet_map[current_id]
        thread.append(row)

        child_ids = children.get(current_id, [])

        if not child_ids:
            break

        # Pick the earliest direct reply.
        # We will later improve this using customer identity and
        # conversation branching logic.
        current_id = child_ids[0]

    return thread


def build_conversations(df: pd.DataFrame, brand_name: str):
    """
    Build candidate support conversations for a single brand.

    Only conversations containing the selected brand are considered.
    """

    df = df.copy()

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
        utc=True,
    )

    # Keep only tweets involving the selected brand.
    brand_mask = (
        df["author_id"].eq(brand_name)
        | df["text"].str.contains(
            f"@{brand_name}",
            case=False,
            na=False,
        )
    )

    brand_df = df[brand_mask].copy()

    tweet_map, children = build_reply_maps(brand_df)

    conversations = []

    for tweet_id in tweet_map:
        thread = get_thread(
            tweet_id,
            tweet_map,
            children,
        )

        if len(thread) < 2:
            continue

        messages = []

        for row in thread:
            messages.append(
                {
                    "tweet_id": int(row.tweet_id),
                    "author_id": row.author_id,
                    "inbound": bool(row.inbound),
                    "created_at": row.created_at,
                    "text": row.text,
                }
            )

        conversations.append(
            {
                "conversation_id": f"{brand_name}_{tweet_id}",
                "messages": messages,
                "turn_count": len(messages),
            }
        )

    return conversations