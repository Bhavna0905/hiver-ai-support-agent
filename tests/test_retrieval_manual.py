from src.retrieval.retrieve import ResolutionRetriever


def main():
    retriever = ResolutionRetriever()

    test_messages = [
        "My package has not arrived yet.",
        "I want to cancel my order.",
        "My Amazon Pay balance is not showing.",
        "My payment was declined.",
        "My Alexa app is not working.",
    ]

    for message in test_messages:
        print("\n" + "=" * 80)
        print(f"QUERY: {message}")
        print("=" * 80)

        results = retriever.retrieve(
            message,
            top_k=3,
        )

        for number, result in enumerate(
            results,
            start=1,
        ):
            print(f"\nResult {number}")
            print(
                f"Similarity: "
                f"{result['similarity']:.4f}"
            )
            print(
                f"Customer: "
                f"{result['customer_message']}"
            )
            print(
                f"AmazonHelp: "
                f"{result['historical_response']}"
            )


if __name__ == "__main__":
    main()