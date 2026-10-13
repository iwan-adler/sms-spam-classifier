# Training report

Generated 2026-10-13 17:00 UTC by `python -m spam_classifier.train` (seed 42, scikit-learn 1.9.1, dataset SHA-256 `1587ea43e58e…`).

## Data

5574 messages → 5171 after removing exact duplicates (spam rate 12.6%). Split by template group, stratified: development 4394 (used for 5-fold grouped cross-validation), test 777 (held out until the end).

## Model comparison (5-fold cross-validation on development data)

PR-AUC is the mean ± standard error over folds. Precision, recall and F1 are computed on the out-of-fold predictions at each model's own threshold (highest recall with precision ≥ 98%). Models are listed simplest first; the selected model is the simplest one within one standard error of the best PR-AUC.

| Model | PR-AUC | Precision | Recall | F1 | Ham flagged as spam |
|---|---|---|---|---|---|
| baseline: always ham | 0.126 ± 0.000 | 0.126 | 1.000 | 0.224 | 100.00% |
| baseline: keyword rule | 0.713 ± 0.026 | 0.980 | 0.442 | 0.609 | 0.13% |
| naive bayes (words) | 0.969 ± 0.004 | 0.980 | 0.901 | 0.939 | 0.26% |
| linear SVM (words+chars, C=0.1) **(selected)** | 0.983 ± 0.004 | 0.981 | 0.930 | 0.955 | 0.26% |
| logistic regression (words+chars, C=1) | 0.981 ± 0.004 | 0.981 | 0.919 | 0.949 | 0.26% |
| linear SVM (words+chars, C=1) | 0.983 ± 0.004 | 0.981 | 0.935 | 0.957 | 0.26% |
| logistic regression (words+chars, C=10) | 0.982 ± 0.004 | 0.981 | 0.930 | 0.955 | 0.26% |
| logistic regression (words+chars, C=100) | 0.983 ± 0.004 | 0.981 | 0.939 | 0.959 | 0.26% |

## Test set (evaluated once): linear SVM (words+chars, C=0.1)

Refitted on all development data. Threshold 0.0282 (decision score), chosen on out-of-fold development predictions.

| Metric | Value |
|---|---|
| PR-AUC | 0.988 |
| Precision (spam) | 0.941 |
| Recall (spam) | 0.970 |
| F1 (spam) | 0.955 |
| Accuracy | 0.988 |
| Ham flagged as spam | 0.88% |

Confusion matrix (rows = actual, columns = predicted):

| | predicted ham | predicted spam |
|---|---|---|
| **actual ham** | 672 | 6 |
| **actual spam** | 3 | 96 |

## Error analysis (test set)

**Real messages flagged as spam (6):**

- Are you free now?can i call now?
- We have sent JD for Customer Service cum Accounts Executive to ur mail id, For details co…
- Dont you have message offer
- Hi, Mobile no. &lt;#&gt; has added you in their contact list on www.fullonsms.com It s a …
- I liked the new mobile
- I'm vivek:)i got call from your number.

**Spam that got through (3):**

- Burger King - Wanna play footy at a top stadium? Get 2 Burger King before 1st Sept and go…
- LIFE has never been this much fun and great until you came in. You made it truly special …
- Latest News! Police station toilet stolen, cops have nothing to go on!
