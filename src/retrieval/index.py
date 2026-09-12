import json
from pathlib import Path

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


CONVERSATIONS_FILE = Path(
    "data/processed/amazonhelp_conversations.jsonl"
)

GOLDEN_FILE = Path(
    "data/golden/golden_dataset.csv"
)

INDEX_FILE = Path(
    "data/processed/amazonhelp_resolutions.faiss"
)

METADATA_FILE = Path(
    "data/processed/amazonhelp_resolutions.jsonl"
)

MODEL_NAME = "all-MiniLM-L6-v2"


def load_golden_tweet_ids():
    """Load tweet IDs used by the golden evaluation set."""

    if not GOLDEN_FILE.exists():
        return set()

    golden = pd.read_csv(
        GOLDEN_FILE
    ).fillna("")

    if "tweet_id" not in golden.columns:
        raise ValueError(
            "Golden dataset must contain tweet_id."
        )

    return set(
        golden["tweet_id"]
        .astype(str)
        .str.strip()
    )


def load_conversations():
    """Read reconstructed AmazonHelp conversations."""

    if not CONVERSATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Conversations file not found: "
            f"{CONVERSATIONS_FILE}"
        )

    conversations = []

    with CONVERSATIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            conversations.append(
                json.loads(line)
            )

    return conversations


def contains_golden_tweet(
    conversation,
    golden_ids,
):
    """Return True if any tweet in the conversation is in the golden set."""

    for message in conversation.get(
        "messages",
        [],
    ):
        tweet_id = str(
            message.get("tweet_id", "")
        ).strip()

        if tweet_id in golden_ids:
            return True

    return False


def build_resolution_pairs(
    conversations,
    golden_ids,
):
    """
    Extract customer -> AmazonHelp resolution pairs.

    A pair consists of:
      customer message
      AmazonHelp response

    Entire conversations containing golden examples are
    excluded to prevent retrieval leakage.
    """

    pairs = []

    excluded_conversations = 0

    for conversation in conversations:

        if contains_golden_tweet(
            conversation,
            golden_ids,
        ):
            excluded_conversations += 1
            continue

        messages = conversation.get(
            "messages",
            [],
        )

        for index in range(
            len(messages) - 1
        ):
            customer_message = messages[index]
            agent_message = messages[index + 1]

            customer_inbound = bool(
                customer_message.get(
                    "inbound",
                    False,
                )
            )

            agent_inbound = bool(
                agent_message.get(
                    "inbound",
                    False,
                )
            )

            # We only want customer -> AmazonHelp
            # response pairs.
            if not customer_inbound:
                continue

            if agent_inbound:
                continue

            customer_text = str(
                customer_message.get(
                    "text",
                    "",
                )
            ).strip()

            agent_text = str(
                agent_message.get(
                    "text",
                    "",
                )
            ).strip()

            if not customer_text:
                continue

            if not agent_text:
                continue

            pairs.append(
                {
                    "conversation_id": conversation.get(
                        "conversation_id"
                    ),
                    "customer_tweet_id": customer_message.get(
                        "tweet_id"
                    ),
                    "agent_tweet_id": agent_message.get(
                        "tweet_id"
                    ),
                    "customer_message": customer_text,
                    "historical_response": agent_text,
                }
            )

    return pairs, excluded_conversations


def main():
    print("Loading golden tweet IDs...")

    golden_ids = load_golden_tweet_ids()

    print(
        f"Golden tweet IDs: "
        f"{len(golden_ids):,}"
    )

    print("\nLoading conversations...")

    conversations = load_conversations()

    print(
        f"Conversations loaded: "
        f"{len(conversations):,}"
    )

    print("\nBuilding historical resolution pairs...")

    pairs, excluded = build_resolution_pairs(
        conversations,
        golden_ids,
    )

    print(
        f"Resolution pairs: "
        f"{len(pairs):,}"
    )

    print(
        f"Conversations excluded because "
        f"they contain golden examples: "
        f"{excluded:,}"
    )

    if not pairs:
        raise RuntimeError(
            "No historical resolution pairs were found."
        )

    print(
        "\nLoading embedding model:"
    )

    print(
        f"  {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    customer_messages = [
        pair["customer_message"]
        for pair in pairs
    ]

    print(
        "\nCreating embeddings..."
    )

    embeddings = model.encode(
        customer_messages,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    print(
        f"Embedding dimension: "
        f"{dimension}"
    )

    # Inner product on normalized embeddings
    # is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    INDEX_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(INDEX_FILE),
    )

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        for pair in pairs:
            json.dump(
                pair,
                file,
                ensure_ascii=False,
            )

            file.write("\n")

    print("\n" + "=" * 70)
    print("Retrieval index built")
    print("=" * 70)

    print(
        f"Indexed resolutions: "
        f"{index.ntotal:,}"
    )

    print(
        f"FAISS index: "
        f"{INDEX_FILE}"
    )

    print(
        f"Metadata: "
        f"{METADATA_FILE}"
    )


if __name__ == "__main__":
    main()