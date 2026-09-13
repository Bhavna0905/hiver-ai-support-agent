
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
        # 3. Unsupported/catch-all intent is not safe to automate.
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
        # 4. Low classifier confidence.
        # ---------------------------------------------------------
        if confidence < self.confidence_threshold:
            return EscalationDecision(
                escalate=True,
                reason=(
                    f"Low intent confidence ({confidence:.2f}) "
                    f"below threshold ({self.confidence_threshold:.2f})."
                ),
            )

        # ---------------------------------------------------------
        # 5. Weak historical evidence.
        # ---------------------------------------------------------
        if top_similarity < self.retrieval_threshold:
            return EscalationDecision(
                escalate=True,
                reason=(
                    f"Historical evidence is weak "
                    f"(similarity {top_similarity:.2f} below "
                    f"threshold {self.retrieval_threshold:.2f})."
                ),
            )

        # ---------------------------------------------------------
        # 6. Strong enough evidence for automated handling.
        # ---------------------------------------------------------
        return EscalationDecision(
            escalate=False,
            reason=(
                "Intent confidence and historical evidence are "
                "strong enough for automated handling."
            ),
        )