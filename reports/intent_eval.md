# Intent classifier evaluation

Test set: 3188 messages (108 are our template-generated account intents, reported separately because they are easier).
All intervals are 95% bootstrap confidence intervals. Temperature and threshold are fitted on the validation set only.

| Model | Banking77 accuracy | Banking77 macro-F1 | Account-intent accuracy | ECE before → after calibration | Temperature |
|---|---|---|---|---|---|
| tfidf-logreg | 0.915 [0.905, 0.924] | 0.915 [0.905, 0.925] | 0.917 [0.861, 0.963] | 0.072 → 0.010 | 0.72 |
| distilbert-finetuned | 0.903 [0.893, 0.913] | 0.890 [0.880, 0.909] | 0.852 [0.787, 0.917] | 0.127 → 0.014 | 0.73 |

## Hand-off to a human (target: 97% accuracy on auto-answered messages)

| Model | Threshold (from validation) | Test coverage (answered by bot) | Test accuracy on answered |
|---|---|---|---|
| tfidf-logreg | 0.79 | 84.5% | 0.973 [0.966, 0.978] |
| distilbert-finetuned | 0.83 | 79.0% | 0.972 [0.965, 0.978] |

Coverage vs. accuracy on test — tfidf-logreg:

| Threshold | Coverage | Accuracy |
|---|---|---|
| 0.00 | 100.0% | 0.915 |
| 0.30 | 99.0% | 0.921 |
| 0.50 | 94.7% | 0.943 |
| 0.70 | 88.6% | 0.962 |
| 0.80 | 84.0% | 0.973 |
| 0.90 | 78.1% | 0.984 |
| 0.95 | 71.7% | 0.989 |

Coverage vs. accuracy on test — distilbert-finetuned:

| Threshold | Coverage | Accuracy |
|---|---|---|
| 0.00 | 100.0% | 0.901 |
| 0.30 | 99.2% | 0.907 |
| 0.50 | 94.4% | 0.928 |
| 0.70 | 88.0% | 0.953 |
| 0.80 | 81.4% | 0.967 |
| 0.90 | 72.0% | 0.983 |
| 0.95 | 61.2% | 0.990 |

## Is distilbert-finetuned really better than tfidf-logreg?

- Accuracy difference (B − A), all test messages: **-0.013 [-0.024, -0.003]** (paired bootstrap)
- Messages only A got right: 160; only B got right: 117
- Exact McNemar test p-value: **0.011**
- For reference: detecting a 1-point difference with 80% power would need about 6,818 paired test messages (we have 3,188).

**Release decision: serve `tfidf-logreg`.** distilbert-finetuned is not significantly better, so the simpler model stays in production.
