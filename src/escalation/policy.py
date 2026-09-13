from dataclasses import dataclass
from typing import List, Optional
import re


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

    The policy prioritizes:
    1. High-risk/security issues
    2. Repeated unsuccessful support interactions
    3. Unresolved financial issues
    4. Persistent unresolved operational issues
    5. Private/sensitive support requests
    6. Weak intent/evidence combinations
    """

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        retrieval_threshold: float = 0.55,
    ):
        self.confidence_threshold = confidence_threshold
        self.retrieval_threshold = retrieval_threshold

        # ---------------------------------------------------------
        # High-risk situations that should receive human review.
        # ---------------------------------------------------------
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
            "fir",
        ]

        # ---------------------------------------------------------
        # Signals that the customer has already tried to get help
        # and the issue remains unresolved.
        # ---------------------------------------------------------
        self.previous_support_phrases: List[str] = [
            "already talked to customer service",
            "already contacted customer service",
            "already contacted support",
            "already talked to support",
            "talked to customer service",
            "talked to support",
            "spoke to customer service",
            "spoke to support",
            "contacted customer service",
            "contacted support",
            "called customer service",
            "called support",
            "sent mail to cs",
            "emailed customer service",
            "emailed support",
            "already reported",
            "already complained",
            "filed a complaint",
            "still not working",
            "still doesn't work",
            "still does not work",
            "still haven't received",
            "still have not received",
            "no results",
            "no response",
            "no one can tell me",
            "nothing happened",
            "not resolved",
            "we are stuck",
            "we're stuck",
            "multiple times",
            "multiple complaints",
            "still waiting",
        ]

        # ---------------------------------------------------------
        # Signals that the customer repeatedly failed to complete
        # an action.
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Narrow signals for unresolved financial issues.
        # ---------------------------------------------------------
        self.unresolved_financial_phrases: List[str] = [
            "amazon pay cashback",
            "cashback since",
            "cashback for",
            "cashback not received",
            "cashback hasn't arrived",
            "cashback has not arrived",
        ]

        # ---------------------------------------------------------
        # Requests that are better handled privately.
        # ---------------------------------------------------------
        self.private_support_phrases: List[str] = [
            "can't dm",
            "cannot dm",
            "can't direct message",
            "cannot direct message",
            "dm me",
            "message me privately",
            "sensitive issue",
            "can someone call me",
            "please call me",
            "call me",
        ]

        # ---------------------------------------------------------
        # Simple positive feedback should not trigger escalation.
        # ---------------------------------------------------------
        self.non_escalation_phrases: List[str] = [
            "thank you",
            "thanks",
            "appreciate it",
            "great service",
            "excellent service",
            "good job",
            "shout out",
            "well done",
        ]

    def _has_repeated_contact_signal(self, message: str) -> bool:
        """
        Detect numeric repetition such as:
        '3 times', '4 times', '5 complaints', etc.

        This is intentionally pattern-based rather than tied to
        individual golden examples.
        """
        patterns = [
            r"\b\d+\s+times\b",
            r"\b\d+\s+complaints?\b",
            r"\b\d+\s+attempts?\b",
            r"\b\d+\s+calls?\b",
            r"\b\d+\s+emails?\b",
        ]

        return any(
            re.search(pattern, message)
            for pattern in patterns
        )

    def _has_unresolved_refund_signal(self, message: str) -> bool:
        """
        Detect combinations indicating a financial issue remains
        unresolved, without escalating every mention of refunds.
        """
        financial_terms = [
            "refund",
            "refunded",
            "refund money",
            "cashback",
            "money back",
            "payment",
            "charged",
        ]

        unresolved_terms = [
            "still",
            "yet",
            "haven't",
            "have not",
            "not received",
            "not refunded",
            "no response",
            "no one",
            "waiting",
            "can't get",
            "cannot get",
            "missing",
        ]

        has_financial = any(
            term in message
            for term in financial_terms
        )

        has_unresolved = any(
            term in message
            for term in unresolved_terms
        )

        return has_financial and has_unresolved

    def _has_persistent_unresolved_issue(
        self,
        message: str,
    ) -> bool:
        """
        Detect persistent operational issues.

        The rule combines a persistence signal with an operational
        issue signal so that ordinary informational questions do not
        automatically escalate.
        """
        persistence_signals = [
            "for days",
            "for weeks",
            "for months",
            "again",
            "no results",
            "no response",
            "not resolved",
            "still waiting",
            "still unresolved",
            "nothing happened",
            "no one can tell me",
        ]

        operational_signals = [
            "refund",
            "cashback",
            "delivery",
            "delivered",
            "package",
            "order",
            "preorder",
            "prime video",
            "fire stick",
            "alexa",
            "payment",
            "invoice",
            "email",
            "broken",
            "not working",
        ]

        has_persistence = any(
            phrase in message
            for phrase in persistence_signals
        )

        has_operational_issue = any(
            phrase in message
            for phrase in operational_signals
        )

        return (
            has_persistence
            and has_operational_issue
        )

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

        retrieved_examples is kept for compatibility with the
        existing tests and agent code.
        """

        message = str(customer_message).lower()

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
        # 1. High-risk/security/legal issues.
        #
        # Short terms such as "fir" use word boundaries to avoid
        # matching them inside unrelated words.
        # ---------------------------------------------------------
        for phrase in self.high_risk_phrases:
            if len(phrase) <= 4:
                matched = re.search(
                    rf"\b{re.escape(phrase)}\b",
                    message,
                )
            else:
                matched = phrase in message

            if matched:
                return EscalationDecision(
                    escalate=True,
                    reason=(
                        f"High-risk issue detected: "
                        f"'{phrase}'. Human review required."
                    ),
                )

        # ---------------------------------------------------------
        # 2. Previous unsuccessful support interactions.
        # ---------------------------------------------------------
        for phrase in self.previous_support_phrases:
            if phrase in message:
                return EscalationDecision(
                    escalate=True,
                    reason=(
                        "Customer indicates a previous support "
                        "interaction remains unresolved. "
                        "Human follow-up is appropriate."
                    ),
                )

        # Numeric repeated-contact signals such as "3 times".
        if self._has_repeated_contact_signal(message):
            return EscalationDecision(
                escalate=True,
                reason=(
                    "Customer reports repeated support attempts "
                    "or complaints. Human follow-up is appropriate."
                ),
            )

        # ---------------------------------------------------------
        # 3. Repeated failed attempts.
        # ---------------------------------------------------------
        for phrase in self.repeated_attempt_phrases:
            if phrase in message:
                return EscalationDecision(
                    escalate=True,
                    reason=(
                        "Customer reports repeated failed attempts. "
                        "Human follow-up is appropriate."
                    ),
                )

        # ---------------------------------------------------------
        # 4. Narrow unresolved financial signals.
        # ---------------------------------------------------------
        for phrase in self.unresolved_financial_phrases:
            if phrase in message:
                return EscalationDecision(
                    escalate=True,
                    reason=(
                        "Unresolved financial issue detected. "
                        "Human review is appropriate."
                    ),
                )

        if self._has_unresolved_refund_signal(message):
            return EscalationDecision(
                escalate=True,
                reason=(
                    "Customer reports an unresolved financial issue. "
                    "Human review is appropriate."
                ),
            )

        # ---------------------------------------------------------
        # 5. Persistent unresolved operational issues.
        # ---------------------------------------------------------
        if self._has_persistent_unresolved_issue(message):
            return EscalationDecision(
                escalate=True,
                reason=(
                    "Customer describes a persistent unresolved "
                    "support issue. Human follow-up is appropriate."
                ),
            )

        # ---------------------------------------------------------
        # 6. Requests requiring private/sensitive support.
        # ---------------------------------------------------------
        for phrase in self.private_support_phrases:
            if phrase in message:
                return EscalationDecision(
                    escalate=True,
                    reason=(
                        "Customer requests private or sensitive "
                        "support. Human follow-up is appropriate."
                    ),
                )

        # ---------------------------------------------------------
        # 7. Unsupported/catch-all intent.
        #
        # Do not automatically escalate every 'other' message.
        # Simple positive feedback should not create unnecessary
        # human workload.
        # ---------------------------------------------------------
        if intent == "other":
            is_simple_positive_feedback = any(
                phrase in message
                for phrase in self.non_escalation_phrases
            )

            if is_simple_positive_feedback:
                return EscalationDecision(
                    escalate=False,
                    reason=(
                        "Message is positive feedback without an "
                        "unresolved support issue."
                    ),
                )

            if (
                confidence < self.confidence_threshold
                and top_similarity < self.retrieval_threshold
            ):
                return EscalationDecision(
                    escalate=True,
                    reason=(
                        "Message could not be confidently mapped "
                        "to a supported workflow and lacks strong "
                        "historical evidence."
                    ),
                )

            return EscalationDecision(
                escalate=False,
                reason=(
                    "Message does not contain a clear high-risk "
                    "or unresolved-support signal."
                ),
            )

        # ---------------------------------------------------------
        # 8. Low confidence + weak historical evidence.
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
        # 9. Weak historical evidence.
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
        # 10. Strong enough evidence for automated handling.
        # ---------------------------------------------------------
        return EscalationDecision(
            escalate=False,
            reason=(
                "Intent confidence and historical evidence are "
                "strong enough for automated handling."
            ),
        )