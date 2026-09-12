import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass


DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_CONFIDENCE_THRESHOLD = 70
MAX_RETRIES = 3


INTENTS = [
    "delivery_problem",
    "order_management",
    "returns_refunds",
    "product_issue",
    "payment_billing",
    "amazon_pay",
    "account_access",
    "prime_membership",
    "digital_content",
    "product_information",
    "technical_issue",
    "support_followup",
    "other",
]


INTENT_DEFINITIONS = {
    "delivery_problem": (
        "Late, missing, wrong-address, tracking, package, shipping, "
        "or carrier issues."
    ),
    "order_management": (
        "Placing, cancelling, changing, or managing an order."
    ),
    "returns_refunds": (
        "Returns, refunds, replacements, refund status, or return pickup."
    ),
    "product_issue": (
        "Damaged, defective, wrong item, missing parts, "
        "or physical product quality issues."
    ),
    "payment_billing": (
        "Charges, cards, billing, invoices, payment failures, "
        "or payment verification."
    ),
    "amazon_pay": (
        "Amazon Pay, Pay balance, wallet, recharge, UPI, "
        "or Amazon Pay cashback issues."
    ),
    "account_access": (
        "Login, password, locked account, email, verification, "
        "suspicious account activity, or security access problems."
    ),
    "prime_membership": (
        "Prime subscription, Prime benefits, Prime charges, "
        "or membership."
    ),
    "digital_content": (
        "Kindle, ebooks, Prime Video, Audible, apps, playback, "
        "or digital purchases."
    ),
    "product_information": (
        "Price, availability, warranty, sizing, specifications, "
        "offers, release information, or pre-purchase information."
    ),
    "technical_issue": (
        "App, website, checkout, error pages, broken links, "
        "device integration, or software/technical failures."
    ),
    "support_followup": (
        "Contacting support, callbacks, agents, or unresolved "
        "prior support cases."
    ),
    "other": (
        "Thanks, jokes, general praise/complaints with no actionable issue, "
        "feature requests, spam, unsupported language, or messages that "
        "do not fit the taxonomy."
    ),
}


class AnnotationError(Exception):
    """Raised when an LLM annotation cannot be validated."""


@dataclass
class AnnotationSuggestion:
    intent: str
    escalation: bool
    confidence: int
    reason: str
    source: str
    disagrees_with_candidate: bool
    low_confidence: bool
    retry_count: int


def build_prompt(customer_message, candidate_intent, previous_error=None):
    definitions = "\n".join(
        f"- {intent}: {definition}"
        for intent, definition in INTENT_DEFINITIONS.items()
    )

    retry_instruction = ""

    if previous_error:
        retry_instruction = (
            "\nYour previous response was invalid because: "
            f"{previous_error}\n"
            "Return corrected JSON only. Do not add any explanation."
        )

    return f"""
You are labelling AmazonHelp customer-support tweets.

Choose exactly one intent from this taxonomy:

{definitions}

IMPORTANT CLASSIFICATION RULES:

1. Classify the PRIMARY ACTIONABLE PROBLEM.
   Do not classify based only on a product, service, or keyword mentioned
   in the message.

2. If a message mentions Prime but the actual problem is a late, missing,
   damaged, or incorrectly delivered package, use delivery_problem.

3. If a message mentions Prime but the actual problem is cancelling,
   changing, placing, or managing an order, use order_management.

4. If the message explicitly concerns Amazon Pay, Pay balance, wallet,
   recharge, UPI, or Amazon Pay cashback, use amazon_pay.

5. If the actual problem is a card charge, billing charge, invoice,
   payment failure, or payment verification unrelated specifically to
   Amazon Pay, use payment_billing.

6. If the customer reports a damaged, defective, broken, wrong, or
   physically problematic product, use product_issue.

7. If the problem is an app, website, error, broken link, checkout,
   device integration, or software malfunction, use technical_issue.

8. If the problem concerns Kindle, Prime Video, Audible, ebooks,
   digital purchases, or missing/playback digital content, use
   digital_content.

9. If the customer wants information before or around purchasing,
   such as price, availability, specifications, warranty, offers,
   release information, or product capabilities, use
   product_information.

10. Use account_access for login, password, account lockout, verification,
    suspicious account activity, or account/security access problems.

11. Use returns_refunds when the main request is a return, refund,
    replacement, refund status, or return pickup.

12. Use support_followup when the main issue is an unresolved previous
    support interaction, callback, agent contact, or contacting support.

13. Use prime_membership when the main issue is the Prime subscription,
    Prime membership benefits, Prime-specific charges, or a direct
    membership complaint.

14. Use other for thanks, jokes, general praise/complaints with no
    actionable issue, feature requests, spam, or messages that do not
    fit the taxonomy.

15. When multiple intents are present, select the one that represents
    the customer's main actionable problem.

16. Do not invent details that are not present in the message.

17. The candidate intent is only a weak heuristic hint and may be wrong.

18. The existing gold label is NOT provided to you and must not be
    inferred or copied.

ESCALATION RULES:

Set escalation to true only when the case likely needs a human because
it involves:

- account or security risk
- fraud or phishing
- billing verification
- sensitive personal information
- repeated unresolved support failure
- legal or safety concerns
- unusually complex context
- a situation where automated handling could reasonably cause harm

Otherwise set escalation to false.

IMPORTANT:
A customer being angry, rude, sarcastic, or frustrated does NOT by itself
mean escalation is required.

OUTPUT FORMAT:

Return only valid JSON with exactly these four keys:

{{
  "intent": "one_of_the_allowed_intents",
  "escalation": false,
  "confidence": 0,
  "reason": "One short sentence explaining the primary problem."
}}

Rules:
- intent must be one of the listed intent names.
- escalation must be true or false.
- confidence must be an integer from 0 to 100.
- confidence represents certainty in the classification, NOT severity.
- use lower confidence when the message is ambiguous or lacks context.
- do not use confidence 0 unless the message is genuinely impossible
  to classify.
- reason must be one short sentence.
- do not mention the candidate intent in the reason.
- do not add markdown or additional text.

Candidate intent: {candidate_intent}

Customer message:
{customer_message}

{retry_instruction}
""".strip()


def extract_json_object(text):
    """
    Extract and parse a JSON object from an LLM response.

    The model is instructed to return JSON only, but this also handles
    responses that accidentally contain surrounding text.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise AnnotationError("No JSON object found.")

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise AnnotationError(
            f"Invalid JSON: {error.msg}"
        ) from error


def parse_bool(value):
    """Convert common boolean representations into a real bool."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        lowered = value.strip().lower()

        if lowered in {"true", "yes", "y", "1"}:
            return True

        if lowered in {"false", "no", "n", "0"}:
            return False

    raise AnnotationError("escalation must be a boolean.")


def validate_annotation(
    data,
    candidate_intent,
    source,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    retry_count=0,
):
    """Validate an LLM response and convert it to AnnotationSuggestion."""

    if not isinstance(data, dict):
        raise AnnotationError("Response must be a JSON object.")

    required_keys = {
        "intent",
        "escalation",
        "confidence",
        "reason",
    }

    missing_keys = required_keys - set(data)

    if missing_keys:
        raise AnnotationError(
            "Missing keys: "
            + ", ".join(sorted(missing_keys))
        )

    intent = str(data["intent"]).strip()

    if intent not in INTENTS:
        raise AnnotationError(
            f"Invalid intent: {intent}"
        )

    escalation = parse_bool(data["escalation"])

    try:
        confidence = int(data["confidence"])
    except (TypeError, ValueError) as error:
        raise AnnotationError(
            "confidence must be an integer."
        ) from error

    if confidence < 0 or confidence > 100:
        raise AnnotationError(
            "confidence must be between 0 and 100."
        )

    reason = str(data["reason"]).strip()

    if not reason:
        raise AnnotationError("reason must be non-empty.")

    return AnnotationSuggestion(
        intent=intent,
        escalation=escalation,
        confidence=confidence,
        reason=reason,
        source=source,
        disagrees_with_candidate=(
            intent != str(candidate_intent).strip()
        ),
        low_confidence=confidence < confidence_threshold,
        retry_count=retry_count,
    )


def call_ollama(
    prompt,
    model,
    host,
    timeout_seconds,
):
    """Call the local Ollama generate API."""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
        },
    }

    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )
    except urllib.error.HTTPError as error:
        raise AnnotationError(
            f"Ollama HTTP error {error.code}."
        ) from error
    except urllib.error.URLError as error:
        raise AnnotationError(
            f"Could not connect to Ollama: {error.reason}"
        ) from error
    except TimeoutError as error:
        raise AnnotationError(
            "Ollama request timed out."
        ) from error

    response_text = str(response_data.get("response", "")).strip()

    if not response_text:
        raise AnnotationError(
            "Ollama returned an empty response."
        )

    return response_text


def annotate_with_ollama(
    customer_message,
    candidate_intent,
    model=None,
    host=None,
    timeout_seconds=60,
    max_retries=MAX_RETRIES,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
):
    """
    Generate and validate an annotation using a local Ollama model.

    Retries malformed/invalid responses and returns a validated
    AnnotationSuggestion.
    """

    model = model or os.getenv(
        "OLLAMA_MODEL",
        DEFAULT_MODEL,
    )

    host = host or os.getenv(
        "OLLAMA_HOST",
        DEFAULT_HOST,
    )

    source = f"ollama:{model}"
    previous_error = None

    for attempt in range(max_retries + 1):
        prompt = build_prompt(
            customer_message=customer_message,
            candidate_intent=candidate_intent,
            previous_error=previous_error,
        )

        try:
            response_text = call_ollama(
                prompt=prompt,
                model=model,
                host=host,
                timeout_seconds=timeout_seconds,
            )

            parsed = extract_json_object(response_text)

            return validate_annotation(
                parsed,
                candidate_intent=candidate_intent,
                source=source,
                confidence_threshold=confidence_threshold,
                retry_count=attempt,
            )

        except AnnotationError as error:
            previous_error = str(error)

    raise AnnotationError(
        f"Could not get valid annotation after "
        f"{max_retries} retries: {previous_error}"
    )