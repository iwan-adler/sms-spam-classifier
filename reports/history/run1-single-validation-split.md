# Training report

Generated 2026-10-13 11:29 UTC by `python -m spam_classifier.train` (seed 42, scikit-learn 1.9.1, dataset SHA-256 `1587ea43e58e…`).

## Data

5574 messages → 5171 after removing exact duplicates (spam rate 12.6%). Split by template group, stratified: train 3618, validation 776, test 777.

## Model comparison (validation set)

Threshold per model: highest recall with precision ≥ 98% on validation.

| Model | PR-AUC | Precision | Recall | F1 | Ham flagged as spam |
|---|---|---|---|---|---|
| logistic regression (words+chars, C=100) **(selected)** | 0.990 | 0.989 | 0.959 | 0.974 | 0.15% |
| linear SVM (words+chars, C=1) | 0.990 | 0.989 | 0.959 | 0.974 | 0.15% |
| linear SVM (words+chars, C=0.1) | 0.990 | 0.990 | 0.969 | 0.979 | 0.15% |
| logistic regression (words+chars, C=10) | 0.988 | 0.989 | 0.959 | 0.974 | 0.15% |
| logistic regression (words+chars, C=1) | 0.987 | 0.990 | 0.969 | 0.979 | 0.15% |
| naive bayes (words) | 0.980 | 0.989 | 0.908 | 0.947 | 0.15% |
| baseline: keyword rule | 0.762 | 1.000 | 0.439 | 0.610 | 0.00% |
| baseline: always ham | 0.126 | 0.126 | 1.000 | 0.224 | 100.00% |

## Test set (evaluated once): logistic regression (words+chars, C=100)

Threshold 0.3124 (probability), chosen on validation.

| Metric | Value |
|---|---|
| PR-AUC | 0.987 |
| Precision (spam) | 0.914 |
| Recall (spam) | 0.970 |
| F1 (spam) | 0.941 |
| Accuracy | 0.985 |
| Ham flagged as spam | 1.33% |

Confusion matrix (rows = actual, columns = predicted):

| | predicted ham | predicted spam |
|---|---|---|
| **actual ham** | 669 | 9 |
| **actual spam** | 3 | 96 |

## Error analysis (test set)

**Real messages flagged as spam (9):**

- Hi! You just spoke to MANEESHA V. We'd like to know if you were satisfied with the experi…
- Are you free now?can i call now?
- We have sent JD for Customer Service cum Accounts Executive to ur mail id, For details co…
- Dont you have message offer
- Hi, Mobile no. &lt;#&gt; has added you in their contact list on www.fullonsms.com It s a …
- They have a thread on the wishlist section of the forums where ppl post nitro requests. S…
- Send ur birthdate with month and year, I will tel u ur LIFE PARTNER'S name. and the metho…
- I liked the new mobile
- Sorry i missed your call. Can you please call back.

**Spam that got through (3):**

- Burger King - Wanna play footy at a top stadium? Get 2 Burger King before 1st Sept and go…
- LIFE has never been this much fun and great until you came in. You made it truly special …
- Latest News! Police station toilet stolen, cops have nothing to go on!
