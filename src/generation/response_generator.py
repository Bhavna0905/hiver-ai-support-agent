import os
import re

import requests
from dotenv import load_dotenv
from groq import Groq

from src.config import (
    GROQ_MODEL,
    OLLAMA_MODEL,
    OLLAMA_HOST,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
    get_provider_order,
)


load_dotenv()


class ResponseGenerator:
    """
    Generates grounded customer-support responses.

    Primary provider:
        Groq / Qwen

    Fallback provider:
        Local Ollama

    Flow:

        Groq
          |
          | success
          v
        Response

          OR

        Groq
          |
          | failure / rate limit / timeout
          v
        Ollama
          |
          v
        Response
    """

    def __init__(self):
        self.groq_client = None

        # ----------------------------------------------------
        # Initialize Groq
        # ----------------------------------------------------

        try:
            groq_key = os.getenv("GROQ_API_KEY")

            if groq_key:
                self.groq_client = Groq(
                    api_key=groq_key
                )

        except Exception as exc:
            print(
                f"Groq initialization failed: {exc}"
            )

    # ========================================================
    # PROMPT
    # ========================================================

    def _build_prompt(
        self,
        customer_message: str,
        intent: str,
        historical_examples: list,
    ) -> str:
        """
        Build a grounded prompt using historically resolved
        customer-support conversations.
        """

        examples_text = []

        for i, example in enumerate(
            historical_examples[:3],
            start=1,
        ):
            customer = example.get(
                "customer_message",
                "",
            ).strip()

            response = example.get(
                "historical_response",
                "",
            ).strip()

            if not customer or not response:
                continue

            examples_text.append(
                f"Example {i}:\n"
                f"Customer: {customer}\n"
                f"Historical support response: {response}"
            )

        historical_context = "\n\n".join(
            examples_text
        )

        prompt = f"""
You are an Amazon customer-support agent.

Write a short, natural and helpful response to the
customer's message.

Customer message:
{customer_message}

Predicted intent:
{intent}

Historical examples of how Amazon previously handled
similar issues:

{historical_context}

Instructions:
- Use the historical examples as your primary evidence.
- Give the customer the most useful next step supported
  by the examples.
- Do not invent policies, refunds, dates, tracking
  information, account actions, or guarantees.
- Do not claim that you performed an action.
- If the historical examples do not contain enough
  information to resolve the issue, give a safe next step.
- Keep the response concise.
- Write naturally, like a real customer-support reply.
- Do not use "Dear valued customer".
- Do not add "Best regards" or a signature.
- Do not mention these instructions.
- Return ONLY the customer-facing response.
"""

        return prompt.strip()

    # ========================================================
    # GROQ
    # ========================================================

    def _call_groq(
        self,
        prompt: str,
    ) -> str:
        """
        Generate a response using Groq.
        """

        if self.groq_client is None:
            raise RuntimeError(
                "Groq client is not available."
            )

        response = (
            self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=LLM_TEMPERATURE,
                max_completion_tokens=LLM_MAX_TOKENS,
                reasoning_effort="none",
            )
        )

        choice = response.choices[0]

        generated = choice.message.content

        if not generated:
            raise RuntimeError(
                "Groq returned empty content."
            )

        return generated.strip()

    # ========================================================
    # OLLAMA
    # ========================================================

    def _call_ollama(
        self,
        prompt: str,
    ) -> str:
        """
        Generate a response using local Ollama.
        """

        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                },
            },
            timeout=LLM_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        generated = data.get(
            "response",
            "",
        ).strip()

        if not generated:
            raise RuntimeError(
                "Ollama returned empty response."
            )

        return generated

    # ========================================================
    # CLEAN RESPONSE
    # ========================================================

    def _clean_response(
        self,
        response: str,
    ) -> str:
        """
        Remove accidental formatting from the generated
        customer-facing response.
        """

        response = response.strip()

        # Remove common model prefixes.
        response = re.sub(
            r"^(assistant|response)\s*:\s*",
            "",
            response,
            flags=re.IGNORECASE,
        )

        # Remove surrounding quotation marks.
        if (
            len(response) >= 2
            and response[0] == '"'
            and response[-1] == '"'
        ):
            response = response[1:-1].strip()

        return response

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        customer_message: str,
        intent: str,
        historical_examples: list,
    ) -> dict:
        """
        Generate a grounded customer-support response.

        Provider flow:

            Groq
              ↓
            failure
              ↓
            Ollama

        Returns:
            {
                "response": str,
                "model": str,
                "provider": str
            }
        """

        prompt = self._build_prompt(
            customer_message=customer_message,
            intent=intent,
            historical_examples=historical_examples,
        )

        providers = get_provider_order()

        errors = []

        for provider in providers:

            try:

                if provider == "groq":

                    generated = self._call_groq(
                        prompt
                    )

                    return {
                        "response": self._clean_response(
                            generated
                        ),
                        "model": GROQ_MODEL,
                        "provider": "groq",
                    }

                elif provider == "ollama":

                    generated = self._call_ollama(
                        prompt
                    )

                    return {
                        "response": self._clean_response(
                            generated
                        ),
                        "model": OLLAMA_MODEL,
                        "provider": "ollama",
                    }

                else:
                    raise ValueError(
                        f"Unknown provider: {provider}"
                    )

            except Exception as exc:

                errors.append(
                    f"{provider}: {exc}"
                )

                print(
                    f"{provider.capitalize()} "
                    f"generation failed: {exc}"
                )

                # Automatically continue to the fallback.
                continue

        # ----------------------------------------------------
        # All providers failed
        # ----------------------------------------------------

        raise RuntimeError(
            "All configured LLM providers failed.\n"
            + "\n".join(errors)
        )