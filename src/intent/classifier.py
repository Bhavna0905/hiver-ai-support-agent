from pathlib import Path
from typing import List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


DEFAULT_MODEL_PATH = Path("data/processed/intent_classifier.joblib")


class IntentClassifier:
    """
    TF-IDF + Logistic Regression intent classifier.

    A small high-precision security override is applied before the
    statistical classifier for explicit account-compromise messages.
    This prevents safety-critical security issues from being
    misclassified by the general-purpose intent model.

    The classifier is intentionally kept independent from the
    evaluation harness so it can later be used by the support agent.
    """

    SECURITY_PHRASES: List[str] = [
        "account hacked",
        "my account was hacked",
        "my account got hacked",
        "someone hacked my account",
        "someone has hacked my account",
        "someone accessed my account",
        "someone has accessed my account",
        "unauthorized access",
        "unauthorized login",
        "unauthorized transaction",
        "account compromised",
        "account was compromised",
        "account has been compromised",
        "identity theft",
        "someone is using my account",
        "someone used my account",
    ]

    def __init__(
        self,
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_iter=1000,
        class_weight="balanced",
    ):
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=ngram_range,
                        min_df=min_df,
                        max_features=max_features,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=max_iter,
                        class_weight=class_weight,
                        random_state=42,
                    ),
                ),
            ]
        )

        self._is_fitted = False

    def fit(self, messages, intents):
        """
        Train the classifier.
        """
        messages = list(messages)
        intents = list(intents)

        if not messages:
            raise ValueError("Training messages cannot be empty.")

        if len(messages) != len(intents):
            raise ValueError(
                "messages and intents must contain the same number of items."
            )

        if len(set(intents)) < 2:
            raise ValueError(
                "Training requires at least two different intent classes."
            )

        self.pipeline.fit(messages, intents)
        self._is_fitted = True

        return self

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "IntentClassifier has not been fitted yet. "
                "Call fit() or load() first."
            )

    def _security_override(self, message):
        """
        Return account_access for explicit account-compromise messages.

        Returns None when no high-confidence security phrase is found.
        """
        normalized = " ".join(
            str(message).lower().split()
        )

        # Flexible account-compromise detection.
        #
        # We require both:
        #   1. a security signal
        #   2. an account signal
        #
        # This prevents unrelated uses of "hacked" from automatically
        # becoming account_access.
        security_signal = (
            "hacked" in normalized
            or "compromised" in normalized
            or "unauthorized access" in normalized
            or "unauthorized login" in normalized
        )

        account_signal = (
            "account" in normalized
            or "amazon account" in normalized
        )

        if security_signal and account_signal:
            return {
                "intent": "account_access",
                "confidence": 1.0,
            }

        for phrase in self.SECURITY_PHRASES:
            if phrase in normalized:
                return {
                    "intent": "account_access",
                    "confidence": 1.0,
                }

        return None

    def predict(self, messages):
        """
        Predict the intent for one or more messages.
        """
        self._check_fitted()

        if isinstance(messages, str):
            messages = [messages]

        messages = list(messages)

        predictions = []

        for message in messages:
            override = self._security_override(message)

            if override is not None:
                predictions.append(
                    override["intent"]
                )
            else:
                predictions.append(
                    self.pipeline.predict([message])[0]
                )

        return predictions

    def predict_one(self, message):
        """
        Predict a single message.
        """
        return str(
            self.predict([message])[0]
        )

    def predict_proba(self, messages):
        """
        Return class probabilities for one or more messages.

        For security overrides, a synthetic probability vector is
        returned with probability 1.0 for account_access.
        """
        self._check_fitted()

        if isinstance(messages, str):
            messages = [messages]

        messages = list(messages)

        classes = (
            self.pipeline
            .named_steps["classifier"]
            .classes_
        )

        class_to_index = {
            class_name: index
            for index, class_name in enumerate(classes)
        }

        results = []

        for message in messages:
            override = self._security_override(message)

            if override is not None:
                probabilities = [0.0] * len(classes)

                if "account_access" in class_to_index:
                    probabilities[
                        class_to_index["account_access"]
                    ] = 1.0

                results.append(probabilities)
            else:
                results.append(
                    self.pipeline.predict_proba(
                        [message]
                    )[0]
                )

        return results

    def predict_with_confidence(self, message):
        """
        Predict an intent together with confidence.
        """
        self._check_fitted()

        override = self._security_override(message)

        if override is not None:
            return override

        probabilities = self.predict_proba(
            [message]
        )[0]

        classes = (
            self.pipeline
            .named_steps["classifier"]
            .classes_
        )

        best_index = probabilities.argmax()

        return {
            "intent": str(
                classes[best_index]
            ),
            "confidence": float(
                probabilities[best_index]
            ),
        }

    def save(self, path=DEFAULT_MODEL_PATH):
        """
        Save the trained classifier to disk.
        """
        self._check_fitted()

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            self,
            path,
        )

    @classmethod
    def load(cls, path=DEFAULT_MODEL_PATH):
        """
        Load a previously trained classifier.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Classifier model not found: {path}"
            )

        model = joblib.load(path)

        if not isinstance(model, cls):
            raise TypeError(
                f"Expected {cls.__name__}, "
                f"but loaded {type(model).__name__}."
            )

        model._is_fitted = True

        return model