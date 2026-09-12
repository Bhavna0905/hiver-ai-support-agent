import csv
import json
import re
from collections import Counter
from pathlib import Path

from src.config import PROCESSED_DATA_DIR


INPUT_FILE = PROCESSED_DATA_DIR / "amazonhelp_conversations.jsonl"
TERMS_OUTPUT = PROCESSED_DATA_DIR / "amazonhelp_top_terms.csv"
PHRASES_OUTPUT = PROCESSED_DATA_DIR / "amazonhelp_top_phrases.csv"

TOP_N = 100

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "at", "by", "from",
    "this", "that", "it", "its", "i", "me", "my", "we", "you",
    "your", "our", "they", "them", "their", "he", "she",
    "do", "does", "did", "have", "has", "had",
    "can", "could", "would", "should", "will",
    "please", "just", "get", "got", "getting",
    "amazon", "amazonhelp",
}


TOKEN_RE = re.compile(r"[a-z][a-z']+")


def normalize(text: str) -> str:
    """Normalize a customer message for phrase discovery."""

    text = text.lower()

    # Remove URLs.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove Twitter mentions.
    text = re.sub(r"@\w+", " ", text)

    # Replace numbers with spaces.
    text = re.sub(r"\d+", " ", text)

    # Keep alphabetic text.
    text = re.sub(r"[^a-z'\s]", " ", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_tokens(text: str):
    tokens = TOKEN_RE.findall(text)

    return [
        token
        for token in tokens
        if token not in STOPWORDS and len(token) > 2
    ]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. "
            "Run the conversation extraction step first."
        )

    unigram_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()

    message_count = 0
    conversation_count = 0

    print(f"Reading conversations from: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            conversation = json.loads(line)
            conversation_count += 1

            for message in conversation.get("messages", []):
                # We only want customer messages.
                if not message.get("inbound", False):
                    continue

                text = message.get("text", "")

                normalized = normalize(text)
                tokens = get_tokens(normalized)

                if not tokens:
                    continue

                message_count += 1

                unigram_counts.update(tokens)

                if len(tokens) >= 2:
                    bigram_counts.update(
                        zip(tokens, tokens[1:])
                    )

                if len(tokens) >= 3:
                    trigram_counts.update(
                        zip(tokens, tokens[1:], tokens[2:])
                    )

    print()
    print("Analysis complete.")
    print(f"Conversations: {conversation_count:,}")
    print(f"Customer messages: {message_count:,}")

    # Save individual terms.
    with TERMS_OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "count"])

        for term, count in unigram_counts.most_common(TOP_N):
            writer.writerow([term, count])

    # Save phrases.
    with PHRASES_OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["phrase", "count"])

        for phrase, count in bigram_counts.most_common(TOP_N):
            writer.writerow([" ".join(phrase), count])

        for phrase, count in trigram_counts.most_common(TOP_N):
            writer.writerow([" ".join(phrase), count])

    print()
    print(f"Saved: {TERMS_OUTPUT}")
    print(f"Saved: {PHRASES_OUTPUT}")

    print("\nTop 30 terms:")
    for term, count in unigram_counts.most_common(30):
        print(f"{term:25} {count:,}")

    print("\nTop 30 phrases:")
    for phrase, count in bigram_counts.most_common(30):
        print(f"{' '.join(phrase):35} {count:,}")


if __name__ == "__main__":
    main()