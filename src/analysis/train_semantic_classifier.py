from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data" / "processed" / "intent_training.csv"
MODEL_PATH = ROOT / "data" / "processed" / "intent_classifier_minilm.joblib"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_PER_CLASS = 5_000
RANDOM_STATE = 42
BATCH_SIZE = 64


def build_balanced_training_data(df):
    parts = []

    for intent, group in df.groupby("intent"):
        if len(group) > MAX_PER_CLASS:
            group = group.sample(
                n=MAX_PER_CLASS,
                random_state=RANDOM_STATE,
            )

        parts.append(group)

    balanced = pd.concat(parts, ignore_index=True)

    return balanced.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print(f"Original examples: {len(df):,}")

    df = build_balanced_training_data(df)

    print(f"Balanced examples: {len(df):,}")

    print("\nClass distribution:")
    print(df["intent"].value_counts())

    texts = df["text"].astype(str).tolist()
    labels = df["intent"].tolist()

    # ---------------------------------------------------------
    # 1. Semantic features
    # ---------------------------------------------------------
    print(
        f"\nLoading embedding model: {EMBEDDING_MODEL}"
    )

    encoder = SentenceTransformer(EMBEDDING_MODEL)

    print("\nCreating MiniLM embeddings...")

    embeddings = encoder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    print(
        f"Embedding matrix: {embeddings.shape}"
    )

    # ---------------------------------------------------------
    # 2. Lexical features
    # ---------------------------------------------------------
    print("\nCreating TF-IDF features...")

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=30_000,
        sublinear_tf=True,
    )

    tfidf_features = tfidf.fit_transform(texts)

    print(
        f"TF-IDF matrix: {tfidf_features.shape}"
    )

    # ---------------------------------------------------------
    # 3. Combine semantic + lexical features
    # ---------------------------------------------------------
    print("\nCombining MiniLM + TF-IDF features...")

    combined_features = hstack(
        [
            csr_matrix(embeddings),
            tfidf_features,
        ],
        format="csr",
    )

    print(
        f"Combined feature matrix: {combined_features.shape}"
    )

    # ---------------------------------------------------------
    # 4. Train classifier
    # ---------------------------------------------------------
    print("\nTraining Logistic Regression...")

    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    classifier.fit(
        combined_features,
        labels,
    )

    artifact = {
        "encoder_name": EMBEDDING_MODEL,
        "classifier": classifier,
        "tfidf_vectorizer": tfidf,
        "feature_type": "minilm_tfidf_hybrid",
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    print("\nTraining complete.")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()