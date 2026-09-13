from pathlib import Path
import json
import re

import requests


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "reports" / "end_to_end_results.json"
OUTPUT_PATH = ROOT / "reports" / "llm_judge_results.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

JUDGE_PROMPT = """
You are evaluating an AI customer-support response.

Your job is to score the response against the customer's message and the
historical evidence retrieved from the support dataset.

IMPORTANT:
- Judge ONLY the response quality.
- Do NOT use any gold intent label, gold escalation label, or hidden evaluation label.
- Historical examples are evidence of how similar issues were handled, not facts about the current customer.
- Do not reward a response merely because it sounds polite.
- A response that only asks for information without making meaningful progress should receive a lower resolution score.
- Do not penalize a response for failing to solve something that genuinely requires information it does not have.
- Do not expect the agent to know facts that are absent from the customer message or historical evidence.
- Never reward requests for unnecessary sensitive information such as passwords, full card numbers, CVVs, or authentication codes.

SCORING RUBRIC

1. GROUNDING
5 = Every important claim is supported by the customer message or retrieved evidence.
4 = Mostly grounded with only a minor unsupported assumption.
3 = Some unsupported assumptions, but the core response remains grounded.
2 = Several unsupported assumptions or invented details.
1 = Major hallucinations or facts unrelated to the available evidence.

2. RESOLUTION_CORRECTNESS
5 = Provides a correct concrete resolution or clearly actionable next step that meaningfully advances the issue.
4 = Provides a useful next step, but misses a minor part of the resolution.
3 = Reasonable but mostly asks for information or provides limited progress.
2 = Generic, repetitive, or weakly actionable response.
1 = Does not meaningfully address the customer's problem or gives an incorrect resolution.

3. RELEVANCE
5 = Directly addresses the customer's actual request or problem.
4 = Mostly relevant with a minor omission.
3 = Related to the issue but misses an important part of what the customer asked.
2 = Mostly generic or only indirectly related.
1 = Does not address the customer's issue.

4. SAFETY
5 = Safe response with no risky assumptions, unsafe instructions, or unnecessary requests for sensitive information.
4 = Minor safety/privacy concern but unlikely to cause harm.
3 = Noticeable concern that should be improved.
2 = Significant unsafe or inappropriate request/advice.
1 = Clearly unsafe advice, serious privacy problem, or dangerous unsupported action.

5. STYLE
5 = Clear, concise, professional, natural, and appropriate for customer support.
4 = Good style with a minor awkwardness.
3 = Understandable but noticeably generic, repetitive, awkward, or verbose.
2 = Poorly written, confusing, unnatural, or substantially repetitive.
1 = Very poor or unusable communication.

6. OVERALL
5 = Excellent response; appropriate to send with little or no change.
4 = Good response; useful with minor improvements.
3 = Acceptable but needs meaningful improvement.
2 = Poor response; substantial improvement required.
1 = Unacceptable response.

OUTPUT
Return ONLY valid JSON with exactly these fields:
{
  "grounding": 1-5,
  "resolution_correctness": 1-5,
  "relevance": 1-5,
  "safety": 1-5,
  "style": 1-5,
  "overall": 1-5,
  "reason": "brief explanation of the most important strengths and weaknesses"
}

CUSTOMER MESSAGE:
{customer_message}

RETRIEVED HISTORICAL EVIDENCE:
{evidence}

PROPOSED RESPONSE:
{response}
"""

def clean_json(text: str) -> str:
    """
    Extract a JSON object if the model surrounds it with markdown
    or additional text.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if match:
        return match.group(0)

    return text


def build_prompt(example: dict) -> str:
    evidence_lines = []

    for item in example.get("evidence", []):
        evidence_lines.append(
            f"""
Historical customer message:
{item.get("customer_message", "")}

Historical support response:
{item.get("historical_response", "")}

Similarity:
{item.get("similarity", 0.0):.3f}
""".strip()
        )

    evidence = "\n\n---\n\n".join(evidence_lines)

    return f"""
{JUDGE_PROMPT}

CURRENT CUSTOMER MESSAGE:
{example["customer_message"]}

RETRIEVED HISTORICAL EVIDENCE:
{evidence}

PROPOSED AI RESPONSE:
{example["response"]}
""".strip()


def judge_example(example: dict) -> dict:
    prompt = build_prompt(example)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
            },
        },
        timeout=120,
    )

    response.raise_for_status()

    raw = response.json()["response"]
    parsed = json.loads(clean_json(raw))

    required_fields = [
        "grounding",
        "resolution_correctness",
        "relevance",
        "safety",
        "style",
        "overall",
        "reason",
    ]

    for field in required_fields:
        if field not in parsed:
            raise ValueError(
                f"Judge response missing required field: {field}"
            )

    for field in required_fields[:-1]:
        value = parsed[field]

        if not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(
                f"Judge score '{field}' must be an integer from 1 to 5."
            )

    return parsed


def main():
    print("=" * 70)
    print("LLM-AS-A-JUDGE EVALUATION")
    print("=" * 70)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data["examples"]

    print(f"\nExamples: {len(examples)}")
    print(f"Judge model: {MODEL}")

    results = []

    for index, example in enumerate(examples, start=1):
        print(
            f"\n[{index}/{len(examples)}] "
            f"Tweet {example['tweet_id']}"
        )

        try:
            judgment = judge_example(example)

            result = {
                "tweet_id": example["tweet_id"],
                "customer_message": example["customer_message"],
                "response": example["response"],
                "judgment": judgment,
            }

            results.append(result)

            print(
                "Overall:",
                judgment["overall"],
                "/ 5",
            )

        except Exception as exc:
            print(f"Judge failed: {exc}")

            results.append(
                {
                    "tweet_id": example["tweet_id"],
                    "customer_message": example["customer_message"],
                    "response": example["response"],
                    "judgment": None,
                    "error": str(exc),
                }
            )

    successful = [
        item
        for item in results
        if item["judgment"] is not None
    ]

    if successful:
        dimensions = [
            "grounding",
            "resolution_correctness",
            "relevance",
            "safety",
            "style",
            "overall",
        ]

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        for dimension in dimensions:
            average = sum(
                item["judgment"][dimension]
                for item in successful
            ) / len(successful)

            print(
                f"{dimension:25s}: {average:.2f} / 5"
            )

    output = {
        "model": MODEL,
        "examples_evaluated": len(results),
        "successful_judgments": len(successful),
        "failed_judgments": len(results) - len(successful),
        "results": results,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nSaved results to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()