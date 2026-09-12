from typing import Any


class EscalationPolicy:
    """
    Rule-based policy for deciding whether a customer request
    should be handled automatically or escalated to a human.

    The policy intentionally favors safety over aggressive automation.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        retrieval_threshold: float = 0.55,
    ):
        self.confidence_threshold = confidence_threshold
        self.retrieval_threshold = retrieval_threshold

    def decide(
        self,
        customer_message: str,
        intent: str,
        confidence: float,
        retrieved_examples: list[dict[str, Any]],
    ) -> dict[str, Any]:

        message = customer_message.lower()

        # ---------------------------------------------------------
        # Rule 1: Very low classifier confidence
        # ---------------------------------------------------------
        if confidence < self.confidence_threshold:
            return {
                "escalate": True,
                "reason": (
                    f"Low intent confidence ({confidence:.2f}) "
                    f"below threshold ({self.confidence_threshold:.2f})."
                ),
            }

        # ---------------------------------------------------------
        # Rule 2: No useful historical evidence
        # ---------------------------------------------------------
        if not retrieved_examples:
            return {
                "escalate": True,
                "reason": "No historical resolution evidence was retrieved.",
            }

        top_similarity = float(
            retrieved_examples[0].get("similarity", 0.0)
        )

        if top_similarity < self.retrieval_threshold:
            return {
                "escalate": True,
                "reason": (
                    f"Retrieved evidence is weak "
                    f"(top similarity {top_similarity:.2f})."
                ),
            }

        # ---------------------------------------------------------
        # Rule 3: Sensitive / potentially high-risk requests
        # ---------------------------------------------------------
        high_risk_phrases = [
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

        for phrase in high_risk_phrases:
            if phrase in message:
                return {
                    "escalate": True,
                    "reason": (
                        f"Potentially high-risk request detected "
                        f"('{phrase}')."
                    ),
                }

        # ---------------------------------------------------------
        # Rule 4: Account access problems
        # ---------------------------------------------------------
        if intent == "account_access":
            return {
                "escalate": True,
                "reason": (
                    "Account-access issues can involve account security "
                    "and should be reviewed by a human."
                ),
            }

        # ---------------------------------------------------------
        # Rule 5: Other / unclear requests
        # ---------------------------------------------------------
        if intent == "other":
            return {
                "escalate": True,
                "reason": (
                    "Request does not match a supported automated "
                    "support intent."
                ),
            }

        # ---------------------------------------------------------
        # Otherwise: safe enough to auto-handle
        # ---------------------------------------------------------
        return {
            "escalate": False,
            "reason": (
                "Intent confidence and historical evidence are strong "
                "enough for automated handling."
            ),
        }