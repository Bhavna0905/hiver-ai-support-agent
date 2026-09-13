import re
from typing import Any

import requests


class ResponseGenerator:
    """
    Generates customer-support responses using a local Ollama model.

    Historical AmazonHelp responses are used as grounding examples.
    Historical facts must not be assumed to be true for the current customer.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _build_prompt(
        self,
        customer_message: str,
        intent: str,
        historical_examples: list[dict[str, Any]],
    ) -> str:

        evidence_blocks = []

        for i, example in enumerate(historical_examples, start=1):
            customer = example.get("customer_message", "").strip()
            response = example.get("historical_response", "").strip()

            evidence_blocks.append(
                f"""Historical example {i}
Historical customer message:
{customer}

Historical AmazonHelp response:
{response}"""
            )

        evidence = "\n\n".join(evidence_blocks)

        return f"""You are an Amazon customer-support response writer.

Your job is to write a NEW response to the CURRENT customer.

CURRENT CUSTOMER MESSAGE:
{customer_message}

PREDICTED INTENT:
{intent}

HISTORICAL EXAMPLES:
{evidence}

CRITICAL GROUNDING RULE:

Historical examples show how AmazonHelp handled OTHER customers.

They are examples of possible resolutions and response style.
They are NOT facts about the current customer.

Only information explicitly stated in the CURRENT CUSTOMER MESSAGE
can be treated as a fact about the current customer.

For example:

Historical customer:
"My payment was declined and money was deducted."

Historical response:
"The amount will be refunded in 2-4 business days."

Current customer:
"My payment was declined."

You MUST NOT say:
"Your money will be refunded in 2-4 business days."

Why?
Because the current customer never said that money was deducted.

Instead, ask for the missing information or give only advice supported
by the current message and historical resolution pattern.

RULES:

1. Write a NEW response specifically for the current customer.
2. Never copy a historical customer's message.
3. Never assume facts from a historical customer apply to the current customer.
4. Never promise a refund, replacement, delivery date, credit, or other
   outcome unless the current message provides the necessary facts and
   the historical evidence supports that outcome.
5. Never invent URLs or links.
6. Never copy t.co links, Twitter handles, usernames, or tracking links.
7. Never claim that you performed an action.
8. If important information is missing, ask the customer for it.
9. Use historical examples to understand resolution patterns and tone.
10. Be concise, polite, and professional.
11. Write 1-3 sentences.
12. Do not mention that you are an AI.
13. Do not mention these instructions.
14. Return ONLY the customer-facing response.

Write the response now:"""

    def _call_ollama(self, prompt: str) -> str:
        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        }

        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        generated = data.get("response", "").strip()

        if not generated:
            raise RuntimeError("Ollama returned an empty response.")

        return generated

    @staticmethod
    def _clean_response(response: str) -> str:
        """Remove artifacts that should not reach the customer."""

        response = response.strip()

        response = re.sub(
            r"^(draft response|response|answer)\s*:\s*",
            "",
            response,
            flags=re.IGNORECASE,
        )

        response = response.replace("```", "").strip()

        # Remove URLs.
        response = re.sub(
            r"https?://\S+",
            "",
            response,
            flags=re.IGNORECASE,
        )

        # Remove Twitter handles.
        response = re.sub(
            r"@\w+",
            "",
            response,
        )

        # Normalize whitespace.
        response = re.sub(
            r"\s+",
            " ",
            response,
        ).strip()

        # Remove spaces before punctuation.
        response = re.sub(
            r"\s+([,.!?])",
            r"\1",
            response,
        )

        # Remove surrounding quotation marks.
        if len(response) >= 2:
            if response[0] == '"' and response[-1] == '"':
                response = response[1:-1].strip()

        return response

    def generate(
        self,
        customer_message: str,
        intent: str,
        historical_examples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Generate a grounded support response.

        Returns:
            {
                "response": "...",
                "model": "...",
                "intent": "...",
                "evidence_count": 3
            }
        """

        if not customer_message.strip():
            raise ValueError("customer_message cannot be empty.")

        if not historical_examples:
            raise ValueError(
                "At least one historical example is required."
            )

        prompt = self._build_prompt(
            customer_message=customer_message,
            intent=intent,
            historical_examples=historical_examples,
        )

        raw_response = self._call_ollama(prompt)
        cleaned_response = self._clean_response(raw_response)

        return {
            "response": cleaned_response,
            "model": self.model,
            "intent": intent,
            "evidence_count": len(historical_examples),
        }