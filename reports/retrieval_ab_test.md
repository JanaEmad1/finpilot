# Retrieval A/B test (offline, paired)

3,080 Banking77 test questions; the correct article is known from the question's intent.
Intent model used for routing: `tfidf-logreg`, routing threshold 0.79 (from validation).

| System | Top-1 article correct | Recall@3 | MRR@3 |
|---|---|---|---|
| A: TF-IDF search | 0.526 [0.509, 0.544] | 0.756 [0.740, 0.770] | 0.630 [0.615, 0.644] |
| B: intent-routed (+ TF-IDF fallback) | 0.909 [0.899, 0.919] | – | – |

B routed 85.4% of questions by intent; the rest fell back to A.

**Caveat — this is not a like-for-like fight.** A searches the article text without ever seeing a labelled question; B's classifier was trained on ~9,000 labelled Banking77 questions. A big gap is expected. The useful findings are *how big* it is, and that B's fallback path is only as good as A — so improving A (e.g. embeddings) still matters for the questions B is unsure about.

## Decision statistics

- Difference in top-1 accuracy (B − A): **0.383 [0.366, 0.400]** (95% paired bootstrap CI)
- Questions only A got right: 7; only B got right: 1186
- Exact McNemar p-value: **< 1e-300**
- Power check: to detect a difference this size (+0.383) with 80% power at α = 0.05 we need about 19 paired questions; we have 3,080.

Verdict: **Ship B**: the improvement is statistically significant and the whole CI is above zero.
