import json
from pathlib import Path

import pandas as pd

from src.retrieval.retrieve import ResolutionRetriever


GOLDEN_PATH = Path("data/golden/golden_dataset.csv")


def normalize_intent(intent: str) -> str:
    """Normalize intent labels for comparison."""
    return str(intent).strip().lower()


def build_intent_lexicon():
    """
    Lightweight intent lexicon used only for retrieval evaluation.

    This is intentionally separate from the production classifier.
    The goal is to determine whether retrieved historical examples
    belong to the same broad support workflow as the golden query.
    """
    return {
        "delivery_problem": [
            "delivery",
            "delivered",
            "package",
            "parcel",
            "shipping",
            "shipment",
            "arrived",
            "arrive",
            "delivery date",
            "tracking",
        ],
        "order_management": [
            "cancel order",
            "cancel my order",
            "order status",
            "modify order",
            "change order",
            "order",
        ],
        "returns_refunds": [
            "refund",
            "return",
            "returned",
            "money back",
            "refund money",
        ],
        "product_issue": [
            "broken",
            "damaged",
            "defective",
            "faulty",
            "not working",
            "doesn't work",
            "does not work",
        ],
        "payment_billing": [
            "payment",
            "paid",
            "charged",
            "credit card",
            "debit card",
            "billing",
            "declined",
        ],
        "amazon_pay": [
            "amazon pay",
            "pay balance",
            "amazonpay",
        ],
        "account_access": [
            "account",
            "password",
            "login",
            "log in",
            "sign in",
            "locked out",
        ],
        "prime_membership": [
            "prime membership",
            "prime member",
            "amazon prime",
            "prime subscription",
            "prime",
        ],
        "digital_content": [
            "kindle",
            "ebook",
            "movie",
            "video",
            "music",
            "prime video",
            "audible",
            "stream",
        ],
        "product_information": [
            "price",
            "available",
            "availability",
            "specification",
            "specs",
            "when will",
            "do you have",
        ],
        "technical_issue": [
            "app",
            "website",
            "error",
            "crash",
            "not loading",
            "technical",
            "alexa",
            "device",
        ],
        "support_followup": [
            "still waiting",
            "follow up",
            "follow-up",
            "any update",
            "update",
            "contacted",
            "response",
            "reply",
        ],
    }


def infer_intent(text: str, lexicon: dict) -> str:
    """
    Infer a broad intent from a historical customer message.

    Returns 'unknown' when there is insufficient evidence.
    """
    text = str(text).lower()

    scores = {}

    for intent, keywords in lexicon.items():
        score = 0

        for keyword in keywords:
            if keyword in text:
                # Longer phrases are stronger evidence.
                score += 2 if " " in keyword else 1

        if score > 0:
            scores[intent] = score

    if not scores:
        return "unknown"

    return max(scores, key=scores.get)


def reciprocal_rank(rank):
    if rank is None:
        return 0.0

    return 1.0 / rank


def evaluate_retrieval(top_k: int = 5):
    df = pd.read_csv(GOLDEN_PATH)

    retriever = ResolutionRetriever()
    lexicon = build_intent_lexicon()

    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []
    reciprocal_ranks = []

    evaluated_examples = 0
    unknown_examples = 0

    for _, row in df.iterrows():
        query = str(row["customer_message"])
        gold_intent = normalize_intent(row["gold_intent"])

        # We cannot meaningfully evaluate rows without a valid gold label.
        if not gold_intent or gold_intent == "nan":
            continue

        results = retriever.retrieve(query, top_k=top_k)

        ranked_intents = []

        for result in results:
            historical_message = result["customer_message"]
            inferred_intent = infer_intent(
                historical_message,
                lexicon,
            )
            ranked_intents.append(inferred_intent)

        # Find first retrieved example matching the golden intent.
        rank = None

        for position, intent in enumerate(ranked_intents, start=1):
            if intent == gold_intent:
                rank = position
                break

        recall_at_1.append(
            1 if rank is not None and rank <= 1 else 0
        )

        recall_at_3.append(
            1 if rank is not None and rank <= 3 else 0
        )

        recall_at_5.append(
            1 if rank is not None and rank <= 5 else 0
        )

        reciprocal_ranks.append(reciprocal_rank(rank))

        if "unknown" in ranked_intents:
            unknown_examples += 1

        evaluated_examples += 1

    results = {
        "examples": evaluated_examples,
        "recall_at_1": sum(recall_at_1) / len(recall_at_1),
        "recall_at_3": sum(recall_at_3) / len(recall_at_3),
        "recall_at_5": sum(recall_at_5) / len(recall_at_5),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "examples_with_unknown_retrieved_intent": unknown_examples,
        "evaluation_note": (
            "Retrieved historical customer messages are assigned a broad "
            "intent using an evaluation-only keyword lexicon. These labels "
            "are not production ground truth and should be interpreted as "
            "approximate retrieval relevance."
        ),
    }

    print("\nRetrieval Evaluation")
    print("=" * 60)
    print(f"Examples:   {results['examples']}")
    print(f"Recall@1:   {results['recall_at_1']:.4f}")
    print(f"Recall@3:   {results['recall_at_3']:.4f}")
    print(f"Recall@5:   {results['recall_at_5']:.4f}")
    print(f"MRR:        {results['mrr']:.4f}")
    print(
        "Retrieved examples containing unknown intent:",
        results["examples_with_unknown_retrieved_intent"],
    )

    output_path = Path("reports/retrieval_results.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    evaluate_retrieval()