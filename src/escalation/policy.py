from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str

    def __getitem__(self, key):
        if key == "escalate":
            return self.escalate
        if key == "reason":
            return self.reason
        raise KeyError(key)


class EscalationPolicy:
    """
    Decide whether a customer request should be handled automatically
    or escalated to a human agent.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        retrieval_threshold: float = 0.55,
    ):
        self.confidence_threshold = confidence_threshold
        self.retrieval_threshold = retrieval_threshold

        # High-risk situations that should always receive human review.
        self.high_risk_phrases: List[str] = [
            "fraud",
            "scam",
            "hacked",
            "someone accessed my account",
            "unauthorized",
            "stolen",
            "identity theft",
            "security breach",
            "chargeback",
            "legal action",
            "lawsuit",
            "police",
        ]

        # Signals that the customer has already tried to get help
        # and the issue remains unresolved.
        self.previous_support_phrases: List[str] = [
            "already talked to customer service",
            "already contacted customer service",
            "already contacted support",
            "already talked to support",
            "contacted customer service",
            "contacted support",
            "called customer service",
            "called support",
            "sent mail to cs",
            "emailed customer service",
            "emailed support",
            "still not working",
            "still doesn't work",
            "still does not work",
            "still haven't received",
            "still have not received",
            "we are stuck",
            "we're stuck",
            "multiple times",
            "twice",
            "three times",
        ]

        # Signals that the customer repeatedly failed to complete
        # an action, such as placing an order.
        self.repeated_attempt_phrases: List[str] = [
            "2 attempts",
            "2 attempt",
            "two attempts",
            "3 attempts",
            "3 attempt",
            "three attempts",
            "multiple attempts",
            "multiple tries",
            "failed attempts",
            "failed attempt",
        ]

        # Narrow signals for unresolved financial issues.
        self.unresolved_financial_phrases: List[str] = [
            "amazon pay cashback",
            "cashback since",
            "cashback for",
            "cashback not received",
            "cashback hasn't arrived",
            "cashback has not arrived",
        ]

    def decide(
        self,
        customer_message: str,
        intent: str,
        confidence: float,
        retrieved_examples: Optional[list] = None,
        top_similarity: Optional[float] = None,
    ) -> EscalationDecision:
        """
        Decide whether to escalate.

        Supports both:
        - retrieved_examples: list of retrieval results
        - top_similarity: precomputed similarity score

        retrieved_examples is kept for compatibility with the existing
        tests and agent code.
        """

        message = customer_message.lower()

        # Derive retrieval similarity when only retrieved examples
        # were provided.
        if top_similarity is None:
            if retrieved_examples:
                first = retrieved_examples[0]

                if isinstance(first, dict):
                    top_similarity = float(
                        first.get("similarity", 0.0)
                    )
                else:
                    top_similarity = 0.0
            else:
                top_similarity = 0.0

        # ---------------------------------------------------------
        # 1. High-risk/security issues always go to a human.
        # ---------------------------------------------------------
        for phrase in self.high_risk_phrases:
            if phrase in message:
                return EscalationDecision(
                    escalate=True,
                    reason=(
                        f"High-risk issue detected: '{phrase}'. "
                        "Human review required."
                    ),
                )

        # ---------------------------------------------------------
        # 2. Account-access problems are sensitive.
        # ---------------------------------------------------------
        if intent == "account_access":
            return EscalationDecision(
                escalate=True,
                reason="Account-access issue requires human review.",
            )

        # ---------------------------------------------------------
        # 3. Previous unsuccessful support attempts.
        # ---------------------------------------------------------
        for phrase in self.previous_support_phrases:
            if phrase in message:
                return EscalationDecision(
                    escalate=True,
                    reason=(
                        f"Customer indicates previous unsuccessful "
                        f"support interaction: '{phrase}'. "
                        "Human follow-up is appropriate."
                    ),
                )

        # ---------------------------------------------------------
        # 4. Repeated failed attempts indicate that self-service
        #    handling has already failed.
        # ---------------------------------------------------------
        for phrase in self.repeated_attempt_phrases:
            if phrase in message:
                return EscalationDecision(
                    escalate=True,
                    reason=(
                        f"Customer reports repeated failed attempts: "
                        f"'{phrase}'. Human follow-up is appropriate."
                    ),
                )

        # ---------------------------------------------------------
        # 5. Unresolved financial issues can require human review,
        #    especially when the customer reports missing cashback.
        # ---------------------------------------------------------
        for phrase in self.unresolved_financial_phrases:
            if phrase in message:
                return EscalationDecision(
                    escalate=True,
                    reason=(
                        f"Unresolved financial issue detected: "
                        f"'{phrase}'. Human review is appropriate."
                    ),
                )

        # ---------------------------------------------------------
        # 6. Unsupported/catch-all intent is not safe to automate.
        # ---------------------------------------------------------
        if intent == "other":
            return EscalationDecision(
                escalate=True,
                reason=(
                    "Intent could not be safely mapped to a "
                    "supported workflow."
                ),
            )

        # ---------------------------------------------------------
        # 7. Low confidence + weak evidence.
        #
        # Low confidence alone is not sufficient. This avoids
        # escalating straightforward requests simply because the
        # classifier is uncertain.
        # ---------------------------------------------------------
        if (
            confidence < self.confidence_threshold
            and top_similarity < self.retrieval_threshold
        ):
            return EscalationDecision(
                escalate=True,
                reason=(
                    f"Low classifier confidence ({confidence:.2f}) "
                    f"and weak historical evidence "
                    f"(similarity {top_similarity:.2f})."
                ),
            )

        # ---------------------------------------------------------
        # 8. Weak historical evidence.
        # ---------------------------------------------------------
        if top_similarity < self.retrieval_threshold:
            return EscalationDecision(
                escalate=True,
                reason=(
                    "Historical evidence is weak "
                    f"(similarity {top_similarity:.2f} below "
                    f"threshold {self.retrieval_threshold:.2f})."
                ),
            )

        # ---------------------------------------------------------
        # 9. Strong enough evidence for automated handling.
        # ---------------------------------------------------------
        return EscalationDecision(
            escalate=False,
            reason=(
                "Intent confidence and historical evidence are "
                "strong enough for automated handling."
            ),
        )