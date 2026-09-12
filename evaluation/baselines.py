from collections import Counter

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class MajorityIntentBaseline:
    def __init__(self):
        self.majority_intent = None

    def fit(self, texts, labels):
        del texts
        self.majority_intent = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, texts):
        return [self.majority_intent for _ in texts]


class AlwaysNoEscalationBaseline:
    def fit(self, texts, labels):
        del texts, labels
        return self

    def predict(self, texts):
        return [False for _ in texts]


def build_tfidf_intent_baseline():
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=20000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_tfidf_escalation_baseline():
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=20000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def build_dummy_escalation_baseline():
    return DummyClassifier(
        strategy="most_frequent",
    )
