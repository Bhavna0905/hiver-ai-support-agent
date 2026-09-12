import json
import random
from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/amazonhelp_conversations.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/amazonhelp_message_sample.csv"
)

SAMPLE_SIZE = 500
RANDOM_SEED = 42


def load_customer_messages():
    """Load inbound customer messages from conversations."""

    messages = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            conversation = json.loads(line)

            for message in conversation["messages"]:
                if message["inbound"]:
                    text = str(message["text"]).strip()

                    if len(text) < 10:
                        continue

                    messages.append(
                        {
                            "conversation_id": conversation[
                                "conversation_id"
                            ],
                            "tweet_id": message["tweet_id"],
                            "created_at": message[
                                "created_at"
                            ],
                            "text": text,
                        }
                    )

    return pd.DataFrame(messages)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print("Loading customer messages...")

    df = load_customer_messages()

    print(
        f"Usable customer messages: "
        f"{len(df):,}"
    )

    sample_size = min(
        SAMPLE_SIZE,
        len(df),
    )

    sample = df.sample(
        n=sample_size,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Sampled messages: {len(sample):,}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()