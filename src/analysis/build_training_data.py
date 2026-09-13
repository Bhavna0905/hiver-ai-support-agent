from pathlib import Path

import pandas as pd


DATA_FILE = Path("data/raw/twcs.csv")
GOLDEN_FILE = Path("data/golden/golden_dataset.csv")
OUTPUT_FILE = Path("data/processed/intent_training.csv")

CHUNK_SIZE = 100_000
MAX_OTHER_EXAMPLES = 2_000


INTENT_RULES = {
    "delivery_problem": [
        "not delivered",
        "not received",
        "where is my package",
        "where is my order",
        "delivery delayed",
        "delivery delay",
        "late delivery",
        "package late",
        "order late",
        "delivery date",
        "delivery hasn't",
        "delivery hasnt",
        "package hasn't",
        "package hasnt",
        "still waiting for my package",
    ],

    "order_management": [
        "cancel my order",
        "cancel order",
        "change my order",
        "modify my order",
        "change delivery address",
        "wrong address",
        "change shipping address",
        "order status",
    ],

    "returns_refunds": [
        "return this",
        "return my",
        "return an item",
        "want a refund",
        "need a refund",
        "refund my",
        "refund hasn't",
        "refund hasnt",
        "refund not",
        "replacement",
    ],

    "payment_billing": [
        "payment declined",
        "payment failed",
        "card declined",
        "credit card",
        "debit card",
        "charged twice",
        "charged me",
        "billing",
        "billing issue",
        "payment issue",
    ],

    "amazon_pay": [
        "amazon pay",
        "amazonpay",
        "pay balance",
        "amazon pay balance",
        "amazon pay cashback",
        "amazon pay cash",
        "recharge amazon pay",
        "top up amazon pay",
    ],

    "account_access": [
        "can't login",
        "cant login",
        "cannot login",
        "can't sign in",
        "cant sign in",
        "cannot sign in",
        "password",
        "account locked",
        "locked out",
        "verification code",
        "otp",
        "suspicious login",
        "someone accessed my account",
    ],

    "prime_membership": [
        "prime membership",
        "prime member",
        "prime subscription",
        "cancel prime",
        "prime charge",
        "prime fee",
        "prime benefits",
    ],

    "digital_content": [
        "prime video",
        "primevideo",
        "kindle",
        "ebook",
        "e-book",
        "audible",
        "digital content",
        "movie on prime",
        "movie coming to prime",
        "when is the movie",
        "streaming",
    ],

    "technical_issue": [
        "app is not working",
        "app isn't working",
        "app isnt working",
        "website not working",
        "website isn't working",
        "website isnt working",
        "error message",
        "technical problem",
        "technical issue",
        "not working",
        "doesn't work",
        "doesnt work",
        "cannot connect",
        "can't connect",
        "cant connect",
        "alexa app",
    ],

    "product_information": [
        "how much",
        "what is the price",
        "price of",
        "is this available",
        "availability",
        "specifications",
        "specs",
        "warranty",
        "when will this be available",
        "does this product",
    ],

    "product_issue": [
        "broken",
        "damaged",
        "defective",
        "doesn't work",
        "doesnt work",
        "not working",
        "wrong item",
        "faulty",
        "stopped working",
    ],

    "support_followup": [
        "still waiting for support",
        "no response",
        "nobody replied",
        "contacted support",
        "customer service",
        "customer care",
        "call me",
        "callback",
        "follow up",
        "follow-up",
    ],
}


# These are deliberately high-precision examples of messages
# that are not asking for one of the operational support intents.
OTHER_RULES = [
    "thank you amazon",
    "thanks amazon",
    "thank you amazonhelp",
    "thanks amazonhelp",
    "great service",
    "excellent service",
    "love amazon",
    "love amazon prime",
    "good job amazon",
    "well done amazon",
    "i love amazon",
    "please add",
    "please bring back",
    "feature request",
    "can you add",
    "you should add",
    "suggestion for amazon",
    "feedback for amazon",
]


def normalize_text(text):
    return " ".join(
        str(text)
        .lower()
        .strip()
        .split()
    )


def find_matching_intents(text):
    normalized = normalize_text(text)

    matches = []

    for intent, phrases in INTENT_RULES.items():
        if any(
            phrase in normalized
            for phrase in phrases
        ):
            matches.append(intent)

    return matches


def is_other_message(text):
    normalized = normalize_text(text)

    return any(
        phrase in normalized
        for phrase in OTHER_RULES
    )


def label_message(text):
    """
    Assign an intent only when there is exactly one
    high-confidence operational match.

    Otherwise, return None. The separate OTHER_RULES
    path handles a small set of high-confidence
    non-operational examples.
    """
    matches = find_matching_intents(text)

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0 and is_other_message(text):
        return "other"

    return None


def load_golden_ids():
    if not GOLDEN_FILE.exists():
        return set()

    golden = pd.read_csv(GOLDEN_FILE)

    return set(
        golden["tweet_id"]
        .astype(str)
        .str.strip()
    )


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    golden_ids = load_golden_ids()

    print(
        f"Golden tweet IDs excluded: "
        f"{len(golden_ids):,}"
    )

    training_chunks = []

    total_amazon_messages = 0
    total_labeled = 0
    total_other = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_FILE,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
                "text",
            ],
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        mask = (
            chunk["inbound"].astype(bool)
            & chunk["text"].notna()
            & chunk["text"].astype(str).str.strip().ne("")
        )

        chunk = chunk[mask].copy()

        total_amazon_messages += len(chunk)

        # Never allow golden examples into training.
        chunk = chunk[
            ~chunk["tweet_id"]
            .astype(str)
            .isin(golden_ids)
        ].copy()

        chunk["intent"] = chunk["text"].apply(
            label_message
        )

        # Keep only messages with a high-confidence label.
        chunk = chunk[
            chunk["intent"].notna()
        ].copy()

        if not chunk.empty:
            other_rows = chunk[
                chunk["intent"] == "other"
            ]

            normal_rows = chunk[
                chunk["intent"] != "other"
            ]

            # Keep the other class controlled.
            remaining_other = (
                MAX_OTHER_EXAMPLES - total_other
            )

            if remaining_other <= 0:
                other_rows = other_rows.iloc[0:0]
            elif len(other_rows) > remaining_other:
                other_rows = other_rows.sample(
                    n=remaining_other,
                    random_state=42,
                )

            total_other += len(other_rows)

            chunk = pd.concat(
                [
                    normal_rows,
                    other_rows,
                ],
                ignore_index=True,
            )

            if not chunk.empty:
                training_chunks.append(
                    chunk[
                        [
                            "tweet_id",
                            "text",
                            "intent",
                        ]
                    ]
                )

                total_labeled += len(chunk)

        print(
            f"Processed chunk {chunk_number}: "
            f"{total_labeled:,} labeled examples "
            f"({total_other:,} other)"
        )

    if not training_chunks:
        raise RuntimeError(
            "No training examples were generated."
        )

    training_df = pd.concat(
        training_chunks,
        ignore_index=True,
    )

    training_df = training_df.drop_duplicates(
        subset=["text"]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nTraining dataset")
    print("=" * 60)

    print(
        f"AmazonHelp customer messages inspected: "
        f"{total_amazon_messages:,}"
    )

    print(
        f"High-confidence labeled examples: "
        f"{len(training_df):,}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print("\nExamples per intent:")

    print(
        training_df["intent"]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()