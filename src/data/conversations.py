from collections import defaultdict
import pandas as pd


def _parse_response_ids(value):
    """Parse response_tweet_id into a list of integer tweet IDs."""
    if pd.isna(value):
        return []

    return [
        int(tweet_id.strip())
        for tweet_id in str(value).split(",")
        if tweet_id.strip()
    ]


def build_reply_maps(df: pd.DataFrame):
    """Build tweet lookup and parent -> children mappings."""
    tweet_map = {
        int(row.tweet_id): row
        for row in df.itertuples(index=False)
    }

    children = defaultdict(list)

    for row in df.itertuples(index=False):
        parent_id = row.in_response_to_tweet_id

        if pd.notna(parent_id):
            children[int(parent_id)].append(int(row.tweet_id))

    return tweet_map, children


def _is_brand(row, brand_name):
    return str(row.author_id).lower() == brand_name.lower()


# def _same_customer(row, customer_id):
#     return (
#         not _is_brand(row, customer_id)
#         and str(row.author_id) == str(customer_id)
#     )


def _find_customer_id(row, tweet_map, brand_name):
    """Find the customer involved in a brand/customer interaction."""
    parent_id = row.in_response_to_tweet_id

    if pd.notna(parent_id):
        parent = tweet_map.get(int(parent_id))

        if parent is not None and not _is_brand(parent, brand_name):
            return parent.author_id

    return None


def _trace_backward(
    start_id,
    tweet_map,
    brand_name,
    customer_id,
):
    """
    Trace a conversation backwards.

    Only follow:
      customer -> brand
      brand -> same customer

    This prevents unrelated customers from being merged.
    """
    chain = []
    current_id = start_id
    visited = set()

    while current_id in tweet_map and current_id not in visited:
        visited.add(current_id)

        row = tweet_map[current_id]
        chain.append(row)

        parent_id = row.in_response_to_tweet_id

        if pd.isna(parent_id):
            break

        parent = tweet_map.get(int(parent_id))

        if parent is None:
            break

        if _is_brand(parent, brand_name):
            current_id = int(parent.tweet_id)

        elif str(parent.author_id) == str(customer_id):
            current_id = int(parent.tweet_id)

        else:
            # Another customer: stop.
            break

    chain.reverse()
    return chain


def _trace_forward(
    start_id,
    tweet_map,
    children,
    brand_name,
    customer_id,
):
    """
    Trace the conversation forwards while keeping the same customer.

    When several replies exist, choose the earliest matching reply.
    This gives us a deterministic primary conversation path.
    """
    chain = []
    current_id = start_id
    visited = set()

    while current_id in tweet_map and current_id not in visited:
        visited.add(current_id)

        current = tweet_map[current_id]
        child_ids = children.get(current_id, [])

        candidates = []

        for child_id in child_ids:
            child = tweet_map[child_id]

            if _is_brand(current, brand_name):
                # From brand -> only follow the same customer.
                if str(child.author_id) == str(customer_id):
                    candidates.append(child)

            else:
                # From customer -> only follow brand replies.
                if _is_brand(child, brand_name):
                    candidates.append(child)

        if not candidates:
            break

        candidates.sort(
            key=lambda row: (
                pd.Timestamp(row.created_at)
                if not pd.isna(row.created_at)
                else pd.Timestamp.max
            )
        )

        next_row = candidates[0]

        if int(next_row.tweet_id) in visited:
            break

        chain.append(next_row)
        current_id = int(next_row.tweet_id)

    return chain


def _rows_to_messages(rows):
    """Convert tweet rows into the public conversation format."""
    messages = []

    for row in rows:
        messages.append(
            {
                "tweet_id": int(row.tweet_id),
                "author_id": row.author_id,
                "inbound": bool(row.inbound),
                "created_at": row.created_at,
                "text": row.text,
            }
        )

    return messages


def build_conversations(df: pd.DataFrame, brand_name: str):
    """
    Build customer-specific support conversations.

    A conversation is never allowed to switch from one customer
    to another, even when multiple customers reply to the same
    brand tweet.
    """
    df = df.copy()

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
        utc=True,
    )

    tweet_map, children = build_reply_maps(df)

    conversations = []
    seen_roots = set()

    # Start from customer tweets that have a direct brand response.
    for row in df.itertuples(index=False):

        if bool(row.inbound) is not True:
            continue

        customer_id = row.author_id

        response_ids = _parse_response_ids(row.response_tweet_id)

        brand_responses = [
            response_id
            for response_id in response_ids
            if response_id in tweet_map
            and _is_brand(tweet_map[response_id], brand_name)
        ]

        if not brand_responses:
            continue

        # Trace backwards to find the beginning of this
        # customer-specific conversation.
        backward = _trace_backward(
            int(row.tweet_id),
            tweet_map,
            brand_name,
            customer_id,
        )

        if not backward:
            continue

        root_id = int(backward[0].tweet_id)

        if root_id in seen_roots:
            continue

        # Continue forward from the root.
        forward = _trace_forward(
            root_id,
            tweet_map,
            children,
            brand_name,
            customer_id,
        )

        full_thread = backward + forward

        # Remove accidental duplicate tweet IDs.
        unique_rows = []
        seen_ids = set()

        for thread_row in full_thread:
            tweet_id = int(thread_row.tweet_id)

            if tweet_id not in seen_ids:
                unique_rows.append(thread_row)
                seen_ids.add(tweet_id)

        if len(unique_rows) < 2:
            continue

        seen_roots.add(root_id)

        conversations.append(
            {
                "conversation_id": f"{brand_name}_{root_id}",
                "customer_id": str(customer_id),
                "messages": _rows_to_messages(unique_rows),
                "turn_count": len(unique_rows),
            }
        )

    return conversations