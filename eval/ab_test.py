"""Offline A/B test of two help-centre retrieval strategies. Writes reports/retrieval_ab_test.md.

  A (control):   TF-IDF keyword search over the articles.
  B (treatment): intent-routed — the intent classifier picks the article when it is
                 confident (>= hand-off threshold), otherwise fall back to A.

Ground truth comes for free: each Banking77 test question is labelled with an intent,
and each intent belongs to exactly one article. Both systems answer the SAME questions,
so we use paired statistics (McNemar test, paired bootstrap).
Run:  python -m eval.ab_test
"""
from __future__ import annotations

import json
import sys

import numpy as np

from eval import stats
from finpilot import config
from finpilot.intent.data import ACCOUNT_INTENTS, load_splits
from finpilot.intent.predict import CALIBRATION_FILE, load_intent_model
from finpilot.rag import TfidfRetriever, intent_to_article


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    test = load_splits()["test"]
    test = test[~test.label.isin(ACCOUNT_INTENTS)].reset_index(drop=True)
    mapping = intent_to_article()
    gold = np.array([mapping[label].slug for label in test.label])
    queries = list(test.text)

    retriever = TfidfRetriever()
    top3_a = retriever.search_many(queries, k=3)
    pred_a = np.array([row[0] for row in top3_a])

    model = load_intent_model()
    threshold = json.loads(CALIBRATION_FILE.read_text())[model.name]["threshold"]
    predictions = model.predict(queries)
    routed = np.array([p.confidence >= threshold and p.intent in mapping for p in predictions])
    pred_b = np.array([mapping[p.intent].slug if r else a
                       for p, r, a in zip(predictions, routed, pred_a)])

    correct_a, correct_b = pred_a == gold, pred_b == gold
    recall3_a = np.array([g in row for g, row in zip(gold, top3_a)])
    rank = np.array([row.index(g) + 1 if g in row else 0 for g, row in zip(gold, top3_a)])
    mrr_a = np.where(rank > 0, 1 / np.maximum(rank, 1), 0)

    test_result = stats.mcnemar(correct_a, correct_b)
    diff = stats.paired_difference_ci(correct_a, correct_b)
    n = len(gold)
    p_disc = (test_result["a_only"] + test_result["b_only"]) / n
    effect = float(correct_b.mean() - correct_a.mean())
    needed = stats.mcnemar_sample_size(p_disc, effect) if 0 < abs(effect) <= p_disc else None

    lines = [
        "# Retrieval A/B test (offline, paired)",
        "",
        f"{n:,} Banking77 test questions; the correct article is known from the question's intent.",
        f"Intent model used for routing: `{model.name}`, routing threshold {threshold:.2f} (from validation).",
        "",
        "| System | Top-1 article correct | Recall@3 | MRR@3 |",
        "|---|---|---|---|",
        f"| A: TF-IDF search | {stats.accuracy_ci(correct_a)} | {stats.accuracy_ci(recall3_a)} | "
        f"{stats.accuracy_ci(mrr_a)} |",
        f"| B: intent-routed (+ TF-IDF fallback) | {stats.accuracy_ci(correct_b)} | – | – |",
        "",
        f"B routed {routed.mean():.1%} of questions by intent; the rest fell back to A.",
        "",
        "**Caveat — this is not a like-for-like fight.** A searches the article text without ever seeing "
        "a labelled question; B's classifier was trained on ~9,000 labelled Banking77 questions. A big gap "
        "is expected. The useful findings are *how big* it is, and that B's fallback path is only as good "
        "as A — so improving A (e.g. embeddings) still matters for the questions B is unsure about.",
        "",
        "## Decision statistics",
        "",
        f"- Difference in top-1 accuracy (B − A): **{diff}** (95% paired bootstrap CI)",
        f"- Questions only A got right: {test_result['a_only']}; only B got right: {test_result['b_only']}",
        f"- Exact McNemar p-value: **{stats.format_p(test_result['p_value'])}**",
    ]
    if needed:
        lines.append(f"- Power check: to detect a difference this size ({effect:+.3f}) with 80% power at "
                     f"α = 0.05 we need about {needed:,} paired questions; we have {n:,}.")
    verdict = ("**Ship B**: the improvement is statistically significant and the whole CI is above zero."
               if diff.low > 0 and test_result["p_value"] < 0.05 else
               "**Keep A**: B is not reliably better." if diff.high <= 0 or test_result["p_value"] >= 0.05 else
               "Inconclusive.")
    lines += ["", f"Verdict: {verdict}"]

    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "retrieval_ab_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
