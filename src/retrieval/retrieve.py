import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


INDEX_FILE = Path(
    "data/processed/amazonhelp_resolutions.faiss"
)

METADATA_FILE = Path(
    "data/processed/amazonhelp_resolutions.jsonl"
)

MODEL_NAME = "all-MiniLM-L6-v2"


class ResolutionRetriever:
    """
    Retrieve historically similar AmazonHelp
    customer-support resolutions.
    """

    def __init__(
        self,
        index_file=INDEX_FILE,
        metadata_file=METADATA_FILE,
        model_name=MODEL_NAME,
    ):
        self.index_file = Path(
            index_file
        )

        self.metadata_file = Path(
            metadata_file
        )

        if not self.index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found: "
                f"{self.index_file}"
            )

        if not self.metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: "
                f"{self.metadata_file}"
            )

        print(
            f"Loading embedding model: "
            f"{model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

        self.index = faiss.read_index(
            str(self.index_file)
        )

        self.metadata = []

        with self.metadata_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                self.metadata.append(
                    json.loads(line)
                )

        if self.index.ntotal != len(
            self.metadata
        ):
            raise ValueError(
                "FAISS index size does not match "
                "metadata size."
            )

    def retrieve(
        self,
        customer_message,
        top_k=5,
    ):
        """
        Retrieve the most similar historical
        customer-support resolutions.
        """

        if not customer_message or not str(
            customer_message
        ).strip():
            raise ValueError(
                "customer_message cannot be empty."
            )

        top_k = min(
            int(top_k),
            self.index.ntotal,
        )

        embedding = self.model.encode(
            [str(customer_message)],
            normalize_embeddings=True,
        )

        embedding = embedding.astype(
            "float32"
        )

        scores, indices = self.index.search(
            embedding,
            top_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index < 0:
                continue

            result = dict(
                self.metadata[index]
            )

            result["similarity"] = float(
                score
            )

            results.append(
                result
            )

        return results