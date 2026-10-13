import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from spam_classifier import evaluate, models
from spam_classifier.train import select


# --- metrics ---------------------------------------------------------------------------------


def test_metrics_at_threshold():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.7, 0.8, 0.4])
    m = evaluate.metrics_at(y, scores, threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["false_positive_rate"] == pytest.approx(1 / 3)


def test_pr_auc_perfect_and_imbalance_aware():
    y = np.array([0] * 9 + [1])
    assert evaluate.pr_auc(y, np.arange(10) / 10) == 1.0
    assert evaluate.pr_auc(y, np.zeros(10)) == pytest.approx(0.1)  # no skill = spam rate


def test_choose_threshold_meets_precision_target_with_max_recall():
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    scores = np.arange(10) / 10
    threshold = evaluate.choose_threshold(y, scores, target_precision=0.98)
    m = evaluate.metrics_at(y, scores, threshold)
    assert m["precision"] >= 0.98
    assert threshold == pytest.approx(0.6)  # the lowest cut with no false positives


def test_choose_threshold_falls_back_to_best_f1():
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])  # precision 1.0 only at the very top
    threshold = evaluate.choose_threshold(y, scores, target_precision=1.01)  # unreachable
    assert evaluate.metrics_at(y, scores, threshold)["f1"] == pytest.approx(0.8)


# --- models -----------------------------------------------------------------------------------


def test_keyword_rule():
    rule = models.KeywordRule().fit([], [])
    assert list(rule.predict(["You WON a prize, call 09061701461", "see you at lunch"])) == [1, 0]


def test_candidates_are_ordered_simplest_first():
    names = list(models.candidates())
    assert names[:3] == ["baseline: always ham", "baseline: keyword rule", "naive bayes (words)"]
    assert names.index("logistic regression (words+chars, C=1)") < names.index("logistic regression (words+chars, C=100)")


def test_scores_and_explanations(messages):
    texts = [m.text for m in messages]
    y = np.array([m.label for m in messages])
    logreg = models.candidates()["logistic regression (words+chars, C=1)"].fit(texts, y)
    svm = models.candidates()["linear SVM (words+chars, C=0.1)"].fit(texts, y)
    assert models.score_type(logreg) == "probability" and models.score_type(svm) == "decision score"
    assert 0 <= models.scores(logreg, ["win cash prize"])[0] <= 1

    explained = models.top_spam_features(svm, "urgent win cash prize lunch")
    assert explained and all(value > 0 for _, value in explained)
    assert all(name.startswith(("word: ", "chars: ")) for name, _ in explained)
    assert explained[0][1] == max(value for _, value in explained)
    assert models.top_spam_features(DummyClassifier().fit(texts, y), "x") == []

    bayes = models.candidates()["naive bayes (words)"].fit(texts, y)
    assert [name for name, _ in models.top_spam_features(bayes, "lunch win prize")][:2] in (
        ["word: win", "word: prize"], ["word: prize", "word: win"])


# --- model selection ----------------------------------------------------------------------------


def test_one_standard_error_rule_prefers_simpler_model():
    comparison = [
        {"model": "simple", "cv_pr_auc_mean": 0.980, "cv_pr_auc_se": 0.004},
        {"model": "complex", "cv_pr_auc_mean": 0.983, "cv_pr_auc_se": 0.004},
    ]
    assert select(comparison)["model"] == "simple"  # within one SE of the best
    comparison[0]["cv_pr_auc_mean"] = 0.970
    assert select(comparison)["model"] == "complex"  # clearly worse, so not chosen


def test_baselines_are_never_selected():
    comparison = [
        {"model": "baseline: keyword rule", "cv_pr_auc_mean": 1.0, "cv_pr_auc_se": 0.0},
        {"model": "naive bayes (words)", "cv_pr_auc_mean": 0.95, "cv_pr_auc_se": 0.01},
    ]
    assert select(comparison)["model"] == "naive bayes (words)"
