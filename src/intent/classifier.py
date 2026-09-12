from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


DEFAULT_MODEL_PATH = Path("data/processed/intent_classifier.joblib")


class IntentClassifier:
    """
    TF-IDF + Logistic Regression intent classifier.

    The classifier is intentionally kept independent from the
    evaluation harness so it can later be used by the support agent.
    """

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

        Parameters
        ----------
        messages:
            Iterable of customer messages.

        intents:
            Iterable of intent labels.
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

    def predict(self, messages):
        """
        Predict the intent for one or more messages.
        """
        self._check_fitted()

        if isinstance(messages, str):
            messages = [messages]

        return self.pipeline.predict(list(messages))

    def predict_one(self, message):
        """
        Predict a single message.
        """
        return str(self.predict([message])[0])

    def predict_proba(self, messages):
        """
        Return class probabilities for one or more messages.
        """
        self._check_fitted()

        if isinstance(messages, str):
            messages = [messages]

        return self.pipeline.predict_proba(list(messages))

    def predict_with_confidence(self, message):
        """
        Predict an intent together with the model's probability
        for its predicted class.
        """
        self._check_fitted()

        probabilities = self.predict_proba([message])[0]
        classes = self.pipeline.named_steps["classifier"].classes_

        best_index = probabilities.argmax()

        return {
            "intent": str(classes[best_index]),
            "confidence": float(probabilities[best_index]),
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

        joblib.dump(self, path)

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