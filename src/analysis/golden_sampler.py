import json
import random
import re
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR


INPUT_FILE = PROCESSED_DATA_DIR / "amazonhelp_conversations.jsonl"
OUTPUT_FILE = PROCESSED_DATA_DIR / "golden_candidates.csv"

RANDOM_SEED = 42
CANDIDATES_PER_INTENT = 40


INTENT_KEYWORDS = {
    "delivery_problem": [
        "delivery",
        "delivered",
        "package",
        "shipping",
        "courier",
        "arrive",
        "arrived",
        "late",
        "tracking",
        "shipment",
        "driver",
    ],

    "order_management": [
        "order",
        "cancel",
        "cancellation",
        "ordered",
        "purchase",
        "modify",
        "change order",
    ],

    "returns_refunds": [
        "refund",
        "return",
        "returned",
        "returning",
        "money back",
        "refunds",
        "pickup",
    ],

    "product_issue": [
        "broken",
        "damaged",
        "defective",
        "wrong item",
        "wrong product",
        "faulty",
        "doesn't work",
        "not working",
        "missing",
    ],

    "payment_billing": [
        "payment",
        "charged",
        "charge",
        "credit card",
        "debit card",
        "billing",
        "bill",
        "card",
        "transaction",
    ],

    "amazon_pay": [
        "amazon pay",
        "cashback",
        "cash back",
        "wallet",
        "balance",
        "recharge",
    ],

    "account_access": [
        "account",
        "login",
        "log in",
        "password",
        "email",
        "locked",
        "security",
        "hack",
        "hacked",
    ],

    "prime_membership": [
        "prime",
        "membership",
        "member",
        "subscription",
        "prime video membership",
    ],

    "digital_content": [
        "kindle",
        "prime video",
        "video",
        "movie",
        "ebook",
        "book",
        "audible",
        "streaming",
    ],

    "product_information": [
        "warranty",
        "availability",
        "available",
        "price",
        "exchange",
        "specification",
        "size",
        "edition",
    ],

    "technical_issue": [
        "app",
        "website",
        "error",
        "bug",
        "crash",
        "loading",
        "checkout",
        "login error",
        "not working",
    ],

    "support_followup": [
        "customer service",
        "customer care",
        "support",
        "agent",
        "representative",
        "callback",
        "contact",
        "response",
        "reply",
    ],

    "other": [],
}


def load_messages():
    """
    Load customer messages from the reconstructed conversations.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. "
            "Run the conversation extraction step first."
        )

    messages = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            conversation = json.loads(line)

            conversation_id = conversation.get(
                "conversation_id"
            )

            conversation_messages = conversation.get(
                "messages", []
            )

            for message in conversation_messages:

                # Only customer messages.
                if not message.get("inbound", False):
                    continue

                text = str(message.get("text", "")).strip()

                if not text:
                    continue

                messages.append(
                    {
                        "conversation_id": conversation_id,
                        "tweet_id": message.get("tweet_id"),
                        "customer_message": text,
                        "turn_count": conversation.get(
                            "turn_count", 1
                        ),
                    }
                )

    return messages


def keyword_score(text, keywords):
    """
    Count how many intent keywords appear in a message.
    """

    text = text.lower()

    score = 0

    for keyword in keywords:
        if keyword in text:
            score += 1

    return score


def guess_candidate_intents(text):
    """
    Return intents that have at least one keyword match.
    """

    scores = {}

    for intent, keywords in INTENT_KEYWORDS.items():

        if not keywords:
            continue

        score = keyword_score(text, keywords)

        if score > 0:
            scores[intent] = score

    if not scores:
        return [("other", 0)]

    return sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )


def main():

    random.seed(RANDOM_SEED)

    print("Loading customer messages...")

    messages = load_messages()

    print(f"Customer messages loaded: {len(messages):,}")

    candidates = []

    # Group messages by candidate intent.
    buckets = {
        intent: []
        for intent in INTENT_KEYWORDS
    }

    for message in messages:

        guesses = guess_candidate_intents(
            message["customer_message"]
        )

        # Put the message into its strongest matching bucket.
        best_intent, score = guesses[0]

        buckets[best_intent].append(
            {
                **message,
                "candidate_intent": best_intent,
                "keyword_score": score,
            }
        )

    print("\nCandidate availability:")

    for intent, bucket in buckets.items():
        print(
            f"{intent:25} {len(bucket):>6,}"
        )

    # Sample from each bucket.
    for intent, bucket in buckets.items():

        if not bucket:
            continue

        # Shuffle deterministically.
        random.shuffle(bucket)

        selected = bucket[:CANDIDATES_PER_INTENT]

        candidates.extend(selected)

    # Shuffle final candidate set.
    random.shuffle(candidates)

    df = pd.DataFrame(candidates)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Generated {len(df):,} golden-set candidates."
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("\nCandidate distribution:")

    print(
        df["candidate_intent"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()