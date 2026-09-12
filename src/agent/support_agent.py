from typing import Any

from src.escalation.policy import EscalationPolicy
from src.generation.response_generator import ResponseGenerator
from src.intent.classifier import IntentClassifier
from src.retrieval.retrieve import ResolutionRetriever


class SupportAgent:
    """
    End-to-end AmazonHelp support agent.

    Pipeline:
        customer message
            -> intent classification
            -> historical resolution retrieval
            -> grounded response generation
            -> escalation decision
    """

    def __init__(
        self,
        classifier: IntentClassifier | None = None,
        retriever: ResolutionRetriever | None = None,
        generator: ResponseGenerator | None = None,
        escalation_policy: EscalationPolicy | None = None,
    ):
        self.classifier = classifier or IntentClassifier.load()
        self.retriever = retriever or ResolutionRetriever()
        self.generator = generator or ResponseGenerator()
        self.escalation_policy = escalation_policy or EscalationPolicy()

    def handle(self, customer_message: str) -> dict[str, Any]:
        """
        Process one customer message end-to-end.
        """

        if not customer_message.strip():
            raise ValueError("customer_message cannot be empty.")

        # ---------------------------------------------------------
        # 1. Intent classification
        # ---------------------------------------------------------
        intent_result = self.classifier.predict_with_confidence(
            customer_message
        )

        intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        # ---------------------------------------------------------
        # 2. Historical resolution retrieval
        # ---------------------------------------------------------
        retrieved = self.retriever.retrieve(
            customer_message,
            top_k=3,
        )

        # ---------------------------------------------------------
        # 3. Generate grounded response
        # ---------------------------------------------------------
        generation_result = self.generator.generate(
            customer_message=customer_message,
            intent=intent,
            historical_examples=retrieved,
        )

        # ---------------------------------------------------------
        # 4. Escalation decision
        # ---------------------------------------------------------
        escalation_result = self.escalation_policy.decide(
            customer_message=customer_message,
            intent=intent,
            confidence=confidence,
            retrieved_examples=retrieved,
        )

        return {
            "customer_message": customer_message,
            "intent": intent,
            "confidence": confidence,
            "response": generation_result["response"],
            "evidence": retrieved,
            "escalate": escalation_result["escalate"],
            "escalation_reason": escalation_result["reason"],
        }