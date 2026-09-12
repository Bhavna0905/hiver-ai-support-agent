from src.generation.response_generator import ResponseGenerator
from src.retrieval.retrieve import ResolutionRetriever


def main():
    retriever = ResolutionRetriever()
    generator = ResponseGenerator()

    test_messages = [
        "My package has not arrived yet.",
        "I want to cancel my order.",
        "My Amazon Pay balance is not showing.",
        "My payment was declined.",
        "My Alexa app is not working.",
    ]

    for message in test_messages:
        print("\n" + "=" * 80)
        print(f"CUSTOMER: {message}")
        print("=" * 80)

        # Retrieve historical resolutions.
        retrieved = retriever.retrieve(message, top_k=3)

        # Generate a response using those historical examples.
        result = generator.generate(
            customer_message=message,
            intent="unknown",
            historical_examples=retrieved,
        )

        print("\nGENERATED RESPONSE:")
        print(result["response"])

        print("\nMODEL:", result["model"])
        print("EVIDENCE USED:", result["evidence_count"])


if __name__ == "__main__":
    main()