"""Train, compare and evaluate. Usage: python -m spam_classifier.train

1. Load, deduplicate and split off a 15% test set (template-grouped, stratified).
2. On the remaining 85% ("development" data), run 5-fold grouped cross-validation
   for every candidate and collect out-of-fold scores.
3. Select a model with the one-standard-error rule: among candidates whose mean
   PR-AUC is within one standard error of the best, take the simplest.
4. Choose the decision threshold on the out-of-fold scores (precision >= 98%).
5. Refit the chosen model on all development data; evaluate ONCE on test.
6. Save the model and write reports/metrics.json + reports/REPORT.md.

Why cross-validation: an earlier version tuned the threshold on a single
776-message validation set (~98 spam). Validation precision was 0.989 but test
precision 0.914; the threshold was fitted to noise. See reports/history/.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from spam_classifier import data, evaluate, models

ROOT = Path(__file__).resolve().parent.parent
DATA_ZIP = ROOT / "data" / "raw" / "sms+spam+collection.zip"
MODEL_PATH = ROOT / "models" / "spam_model.joblib"
REPORTS = ROOT / "reports"


CV_FOLDS = 5


def cross_validate(model, X: list[str], y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold scores for every development message, plus PR-AUC per fold."""
    folds = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    out_of_fold = np.zeros(len(X))
    fold_scores = []
    for train_idx, val_idx in folds.split(np.zeros(len(X)), y, groups):
        fitted = clone(model).fit([X[i] for i in train_idx], y[train_idx])
        scores = models.scores(fitted, [X[i] for i in val_idx])
        out_of_fold[val_idx] = scores
        fold_scores.append(evaluate.pr_auc(y[val_idx], scores))
    return out_of_fold, fold_scores


def select(comparison: list[dict]) -> dict:
    """One-standard-error rule: the simplest candidate within one SE of the best mean PR-AUC.

    `comparison` must be in candidate order (simplest first). Baselines are reference
    points only and are never selected for deployment.
    """
    eligible = [row for row in comparison if not row["model"].startswith("baseline")]
    best = max(eligible, key=lambda row: row["cv_pr_auc_mean"])
    bar = best["cv_pr_auc_mean"] - best["cv_pr_auc_se"]
    return next(row for row in eligible if row["cv_pr_auc_mean"] >= bar)


def run(messages: list[data.Message], seed: int = 42) -> dict:
    """The whole experiment on already-loaded messages. Returns the chosen model and results."""
    unique = data.deduplicate(messages)
    parts = data.split(unique, seed=seed)
    development = parts.train + parts.validation  # the test split is never touched until the end
    X_dev, y_dev = data.texts(development), data.labels(development)
    groups = np.array([data.template_key(t) for t in X_dev])
    X_test, y_test = data.texts(parts.test), data.labels(parts.test)

    comparison, candidates = [], models.candidates()
    for name, model in candidates.items():
        out_of_fold, fold_scores = cross_validate(model, X_dev, y_dev, groups, seed)
        threshold = evaluate.choose_threshold(y_dev, out_of_fold)
        at_threshold = evaluate.metrics_at(y_dev, out_of_fold, threshold)
        comparison.append({
            "model": name,
            "cv_pr_auc_mean": float(np.mean(fold_scores)),
            "cv_pr_auc_se": float(np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores))),
            "cv_threshold": threshold,
            **{f"cv_{k}": at_threshold[k] for k in ("precision", "recall", "f1", "false_positive_rate")},
        })

    chosen = select(comparison)
    best_name, threshold = chosen["model"], chosen["cv_threshold"]
    best_model = clone(candidates[best_name]).fit(X_dev, y_dev)

    test_scores = models.scores(best_model, X_test)
    test_metrics = {"pr_auc": evaluate.pr_auc(y_test, test_scores), **evaluate.metrics_at(y_test, test_scores, threshold)}

    predicted = test_scores >= threshold
    errors = {
        "false_positives": [t for t, y, p in zip(X_test, y_test, predicted) if p and y == 0],
        "false_negatives": [t for t, y, p in zip(X_test, y_test, predicted) if not p and y == 1],
    }
    return {
        "model": best_model,
        "model_name": best_name,
        "threshold": threshold,
        "score_type": models.score_type(best_model),
        "comparison": comparison,
        "test": test_metrics,
        "errors": errors,
        "sizes": {
            "raw": len(messages), "after_dedup": len(unique),
            "development": len(development), "test": len(parts.test),
            "spam_rate": round(float(data.labels(unique).mean()), 4),
        },
        "seed": seed,
    }


def _clip(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def write_report(result: dict, path: Path, dataset_sha256: str) -> None:
    t, s = result["test"], result["sizes"]
    cm = t["confusion_matrix"]
    lines = [
        "# Training report",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `python -m spam_classifier.train` "
        f"(seed {result['seed']}, scikit-learn {sklearn.__version__}, dataset SHA-256 `{dataset_sha256[:12]}…`).",
        "",
        "## Data",
        "",
        f"{s['raw']} messages → {s['after_dedup']} after removing exact duplicates "
        f"(spam rate {s['spam_rate']:.1%}). Split by template group, stratified: "
        f"development {s['development']} (used for {CV_FOLDS}-fold grouped cross-validation), "
        f"test {s['test']} (held out until the end).",
        "",
        f"## Model comparison ({CV_FOLDS}-fold cross-validation on development data)",
        "",
        "PR-AUC is the mean ± standard error over folds. Precision, recall and F1 are computed on the "
        f"out-of-fold predictions at each model's own threshold (highest recall with precision ≥ "
        f"{evaluate.TARGET_PRECISION:.0%}). Models are listed simplest first; the selected model is the "
        "simplest one within one standard error of the best PR-AUC.",
        "",
        "| Model | PR-AUC | Precision | Recall | F1 | Ham flagged as spam |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["comparison"]:
        marker = " **(selected)**" if row["model"] == result["model_name"] else ""
        lines.append(
            f"| {row['model']}{marker} | {row['cv_pr_auc_mean']:.3f} ± {row['cv_pr_auc_se']:.3f} | "
            f"{row['cv_precision']:.3f} | {row['cv_recall']:.3f} | {row['cv_f1']:.3f} | "
            f"{row['cv_false_positive_rate']:.2%} |"
        )
    lines += [
        "",
        f"## Test set (evaluated once): {result['model_name']}",
        "",
        f"Refitted on all development data. Threshold {result['threshold']:.4f} ({result['score_type']}), "
        "chosen on out-of-fold development predictions.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| PR-AUC | {t['pr_auc']:.3f} |",
        f"| Precision (spam) | {t['precision']:.3f} |",
        f"| Recall (spam) | {t['recall']:.3f} |",
        f"| F1 (spam) | {t['f1']:.3f} |",
        f"| Accuracy | {t['accuracy']:.3f} |",
        f"| Ham flagged as spam | {t['false_positive_rate']:.2%} |",
        "",
        "Confusion matrix (rows = actual, columns = predicted):",
        "",
        "| | predicted ham | predicted spam |",
        "|---|---|---|",
        f"| **actual ham** | {cm['tn']} | {cm['fp']} |",
        f"| **actual spam** | {cm['fn']} | {cm['tp']} |",
        "",
        "## Error analysis (test set)",
        "",
        f"**Real messages flagged as spam ({len(result['errors']['false_positives'])}):**",
        "",
    ]
    lines += [f"- {_clip(m)}" for m in result["errors"]["false_positives"]] or ["- none"]
    lines += ["", f"**Spam that got through ({len(result['errors']['false_negatives'])}):**", ""]
    lines += [f"- {_clip(m)}" for m in result["errors"]["false_negatives"]] or ["- none"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save(result: dict, model_path: Path, reports_dir: Path, dataset_sha256: str) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": result["model_name"],
        "threshold": result["threshold"],
        "score_type": result["score_type"],
        "test": result["test"],
        "sizes": result["sizes"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "dataset_sha256": dataset_sha256,
    }
    joblib.dump({"model": result["model"], "metadata": metadata}, model_path)
    report = {k: v for k, v in result.items() if k not in ("model", "errors")}
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_report(result, reports_dir / "REPORT.md", dataset_sha256)


def main() -> int:
    if not DATA_ZIP.is_file():
        print(f"Dataset not found at {DATA_ZIP}. Run: python scripts/download_data.py", file=sys.stderr)
        return 1
    sha = hashlib.sha256(DATA_ZIP.read_bytes()).hexdigest()
    result = run(data.load_zip(DATA_ZIP))
    save(result, MODEL_PATH, REPORTS, sha)
    t = result["test"]
    print(f"Selected: {result['model_name']}")
    print(f"Test: precision {t['precision']:.3f}  recall {t['recall']:.3f}  F1 {t['f1']:.3f}  "
          f"PR-AUC {t['pr_auc']:.3f}  ham flagged {t['false_positive_rate']:.2%}")
    print(f"Saved {MODEL_PATH.relative_to(ROOT)} and reports/REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
