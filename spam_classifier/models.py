"""Candidate models: two baselines and three linear text classifiers."""

from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

SEED = 42


class KeywordRule(BaseEstimator, ClassifierMixin):
    """Hand-written baseline: spam if the message contains typical spam phrases.

    It has nothing to learn; it exists to show how much the trained models
    add over common sense.
    """

    PATTERN = re.compile(
        r"\b(free|win|won|winner|prize|claim|urgent|cash|txt|text \w+ to|call 0\d{6,}|£\d+|ringtone)\b",
        re.IGNORECASE,
    )

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return np.array([len(self.PATTERN.findall(text)) for text in X], dtype=float)

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def word_features() -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def word_and_char_features() -> FeatureUnion:
    # Character n-grams catch obfuscated spam ("FR33", "txt2win") that word tokens miss.
    return FeatureUnion([
        ("word", word_features()),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])


def _logreg(c: float) -> Pipeline:
    return Pipeline([
        ("tfidf", word_and_char_features()),
        ("clf", LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=SEED)),
    ])


def _svm(c: float) -> Pipeline:
    return Pipeline([
        ("tfidf", word_and_char_features()),
        ("clf", LinearSVC(C=c, class_weight="balanced", random_state=SEED)),
    ])


def candidates() -> dict[str, BaseEstimator]:
    """Every model compared during model selection, ordered from simplest to most complex.

    The order matters: when several models are statistically tied, selection
    prefers the earliest (most regularised) one. A smaller C means stronger
    regularisation.
    """
    return {
        "baseline: always ham": DummyClassifier(strategy="most_frequent"),
        "baseline: keyword rule": KeywordRule(),
        "naive bayes (words)": Pipeline([("tfidf", word_features()), ("clf", MultinomialNB(alpha=0.1))]),
        "linear SVM (words+chars, C=0.1)": _svm(0.1),
        "logistic regression (words+chars, C=1)": _logreg(1.0),
        "linear SVM (words+chars, C=1)": _svm(1.0),
        "logistic regression (words+chars, C=10)": _logreg(10.0),
        "logistic regression (words+chars, C=100)": _logreg(100.0),
    }


def scores(model, X) -> np.ndarray:
    """Higher = more likely spam. Probability when the model has one, else a decision score."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def score_type(model) -> str:
    return "probability" if hasattr(model, "predict_proba") else "decision score"


def top_spam_features(model, text: str, k: int = 5) -> list[tuple[str, float]]:
    """For linear pipelines: the features in `text` that pushed hardest towards spam.

    Contribution = TF-IDF value × learned weight, which is exactly how a linear
    model builds its score, so this explanation is faithful rather than approximate.
    """
    if not isinstance(model, Pipeline):
        return []
    vectorizer, classifier = model[0], model[-1]
    if hasattr(classifier, "coef_"):  # logistic regression, linear SVM
        weights = classifier.coef_.ravel()
    elif hasattr(classifier, "feature_log_prob_"):  # naive Bayes: log P(word|spam) - log P(word|ham)
        weights = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    else:
        return []
    row = vectorizer.transform([text]).tocsr()
    contributions = row.data * weights[row.indices]
    names = vectorizer.get_feature_names_out()[row.indices]
    order = np.argsort(contributions)[::-1]
    return [(_readable(str(names[i])), round(float(contributions[i]), 4)) for i in order[:k] if contributions[i] > 0]


def _readable(feature: str) -> str:
    """'word__call now' -> 'word: call now'; 'char__ fre' -> 'chars: " fre"'."""
    kind, sep, value = feature.partition("__")
    if not sep:  # a plain word vectorizer (naive Bayes) has no prefix
        return f"word: {feature}"
    return f'chars: "{value}"' if kind == "char" else f"word: {value}"
