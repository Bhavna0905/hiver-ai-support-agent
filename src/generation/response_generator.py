import re
from typing import Any

import requests


class ResponseGenerator:
    """
    Generates customer-support responses using a local Ollama model.

    Historical AmazonHelp responses are provided as grounding evidence.
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
Customer issue: {customer}
Historical support resolution: {response}"""
            )

        evidence = "\n\n".join(evidence_blocks)

        return f"""You are a customer-support response writer.

Write a NEW response to the customer's message.

CUSTOMER MESSAGE:
{customer_message}

INTENT:
{intent}

HISTORICAL SUPPORT EXAMPLES:
{evidence}

IMPORTANT RULES:

1. Write a NEW response specifically for the customer message.
2. Use the historical support responses only to understand how similar
   issues were handled.
3. NEVER copy or repeat a historical customer's message.
4. NEVER copy URLs, t.co links, Twitter handles, tracking links, or
   placeholder links from the historical examples.
5. NEVER invent a URL or link.
6. NEVER invent order details, delivery dates, refund amounts, policies,
   or actions that are not supported by the evidence.
7. Do not claim that you performed an action.
8. If the evidence is insufficient, ask the customer for the missing
   information or direct them to Amazon support.
9. Be polite, concise, and professional.
10. Write only 1-3 sentences.
11. Do not mention these instructions.
12. Do not mention that you are an AI.
13. Return ONLY the customer-facing response.

BAD RESPONSE:
"Hi, I'd like to cancel an order. I don't see any options to cancel."

WHY IT IS BAD:
This repeats the customer's wording instead of responding to the customer.

GOOD RESPONSE:
"I understand you'd like to cancel your order. If the order has already
shipped, the cancellation option may no longer be available."

Now write the response:"""

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
        """
        Remove artifacts that should never be sent to a customer.
        """

        response = response.strip()

        # Remove common model prefixes.
        response = re.sub(
            r"^(draft response|response|answer)\s*:\s*",
            "",
            response,
            flags=re.IGNORECASE,
        )

        # Remove markdown code fences.
        response = response.replace("```", "").strip()

        # Remove URLs.
        response = re.sub(
            r"https?://\S+",
            "",
            response,
            flags=re.IGNORECASE,
        )

        # Remove Twitter-style handles.
        response = re.sub(
            r"@\w+",
            "",
            response,
        )

        # Remove excessive whitespace.
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