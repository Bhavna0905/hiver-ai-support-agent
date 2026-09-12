from src.escalation.policy import EscalationPolicy


def main():
    policy = EscalationPolicy()

    test_cases = [
        {
            "name": "Normal delivery issue",
            "message": "My package has not arrived yet.",
            "intent": "delivery_problem",
            "confidence": 0.85,
            "similarity": 0.76,
        },
        {
            "name": "Low confidence",
            "message": "Something is wrong with my order.",
            "intent": "order_management",
            "confidence": 0.35,
            "similarity": 0.80,
        },
        {
            "name": "Weak retrieval evidence",
            "message": "My package has a strange problem.",
            "intent": "delivery_problem",
            "confidence": 0.85,
            "similarity": 0.40,
        },
        {
            "name": "Security concern",
            "message": "I think someone hacked my Amazon account.",
            "intent": "account_access",
            "confidence": 0.90,
            "similarity": 0.85,
        },
        {
            "name": "Unsupported intent",
            "message": "I have a completely unusual request.",
            "intent": "other",
            "confidence": 0.90,
            "similarity": 0.85,
        },
    ]

    for case in test_cases:
        retrieved_examples = [
            {
                "similarity": case["similarity"],
                "customer_message": "Historical customer issue",
                "historical_response": "Historical support response",
            }
        ]

        result = policy.decide(
            customer_message=case["message"],
            intent=case["intent"],
            confidence=case["confidence"],
            retrieved_examples=retrieved_examples,
        )

        print("\n" + "=" * 70)
        print(case["name"])
        print("=" * 70)
        print(f"Message:     {case['message']}")
        print(f"Intent:      {case['intent']}")
        print(f"Confidence:  {case['confidence']}")
        print(f"Similarity:  {case['similarity']}")
        print(f"Escalate:    {result['escalate']}")
        print(f"Reason:      {result['reason']}")


if __name__ == "__main__":
    main()