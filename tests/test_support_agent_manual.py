from src.agent.support_agent import SupportAgent


def main():
    agent = SupportAgent()

    test_messages = [
        "My package has not arrived yet.",
        "I want to cancel my order.",
        "My Amazon Pay balance is not showing.",
        "My payment was declined.",
        "My Alexa app is not working.",
        "I think someone hacked my Amazon account.",
    ]

    for message in test_messages:
        print("\n" + "=" * 80)
        print(f"CUSTOMER: {message}")
        print("=" * 80)

        result = agent.handle(message)

        print(f"\nINTENT: {result['intent']}")
        print(f"CONFIDENCE: {result['confidence']:.4f}")
        print(f"\nRESPONSE:\n{result['response']}")
        print(f"\nESCALATE: {result['escalate']}")
        print(f"ESCALATION REASON: {result['escalation_reason']}")

        print("\nTOP EVIDENCE:")
        for i, evidence in enumerate(result["evidence"], start=1):
            print(
                f"{i}. "
                f"similarity={evidence['similarity']:.4f} | "
                f"{evidence['customer_message']}"
            )


if __name__ == "__main__":
    main()