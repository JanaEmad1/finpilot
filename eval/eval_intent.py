"""Evaluate intent models and write reports/intent_eval.md.

Protocol (so the test numbers stay honest):
  * temperature and hand-off threshold are fitted on VALIDATION only;
  * the test set is used once, for the final numbers;
  * Banking77 and our synthetic account intents are reported separately.
Run:  python -m eval.eval_intent
"""
from __future__ import annotations

import json
import sys

import numpy as np
from sklearn.metrics import f1_score

from eval import stats
from finpilot import config
from finpilot.intent.data import ACCOUNT_INTENTS, label_ids, load_splits
from finpilot.intent.predict import (CALIBRATION_FILE, BaselineIntentModel, IntentModel,
                                     TransformerIntentModel)
from finpilot.intent.train import OUT_DIR

TARGET_ACCURACY = 0.97  # accuracy we want on the messages the bot answers by itself


def _macro_f1_ci(y, pred, **kw) -> stats.Estimate:
    return stats.bootstrap_ci(lambda idx: f1_score(y[idx], pred[idx], average="macro"), len(y), **kw)


def evaluate(model: IntentModel, splits, ids) -> dict:
    y_val = splits["val"].label.map(ids).to_numpy()
    y_test = splits["test"].label.map(ids).to_numpy()
    val_logits = model.logits(list(splits["val"].text))
    test_logits = model.logits(list(splits["test"].text))

    temperature = stats.fit_temperature(val_logits, y_val)
    val_probs = stats.softmax(val_logits, temperature)
    test_probs_raw = stats.softmax(test_logits, 1.0)
    test_probs = stats.softmax(test_logits, temperature)
    pred = test_probs.argmax(1)
    correct = pred == y_test

    val_correct = val_probs.argmax(1) == y_val
    threshold = stats.pick_threshold(val_probs.max(1), val_correct, TARGET_ACCURACY)
    answered = test_probs.max(1) >= threshold

    is_account = splits["test"].label.isin(ACCOUNT_INTENTS).to_numpy()
    return {
        "name": model.name,
        "correct": correct,
        "temperature": temperature,
        "threshold": threshold,
        "banking77_acc": stats.accuracy_ci(correct[~is_account]),
        "banking77_f1": _macro_f1_ci(y_test[~is_account], pred[~is_account], n_boot=500),
        "account_acc": stats.accuracy_ci(correct[is_account]),
        "ece_before": stats.expected_calibration_error(test_probs_raw, y_test),
        "ece_after": stats.expected_calibration_error(test_probs, y_test),
        "coverage": float(answered.mean()),
        "auto_accuracy": stats.accuracy_ci(correct[answered]),
        "curve": stats.coverage_curve(test_probs.max(1), correct, np.array([0, .3, .5, .7, .8, .9, .95])),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    splits = load_splits()
    ids = label_ids()
    models: list[IntentModel] = [BaselineIntentModel()]
    if OUT_DIR.exists():
        models.append(TransformerIntentModel())
    results = [evaluate(m, splits, ids) for m in models]

    n_test = len(splits["test"])
    lines = [
        "# Intent classifier evaluation",
        "",
        f"Test set: {n_test} messages ({int(splits['test'].label.isin(ACCOUNT_INTENTS).sum())} are our "
        "template-generated account intents, reported separately because they are easier).",
        "All intervals are 95% bootstrap confidence intervals. Temperature and threshold are fitted "
        "on the validation set only.",
        "",
        "| Model | Banking77 accuracy | Banking77 macro-F1 | Account-intent accuracy | ECE before → after calibration | Temperature |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['name']} | {r['banking77_acc']} | {r['banking77_f1']} | {r['account_acc']} | "
                     f"{r['ece_before']:.3f} → {r['ece_after']:.3f} | {r['temperature']:.2f} |")

    lines += ["", f"## Hand-off to a human (target: {TARGET_ACCURACY:.0%} accuracy on auto-answered messages)", "",
              "| Model | Threshold (from validation) | Test coverage (answered by bot) | Test accuracy on answered |",
              "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['threshold']:.2f} | {r['coverage']:.1%} | {r['auto_accuracy']} |")

    for r in results:
        lines += ["", f"Coverage vs. accuracy on test — {r['name']}:", "",
                  "| Threshold | Coverage | Accuracy |", "|---|---|---|"]
        lines += [f"| {c['threshold']:.2f} | {c['coverage']:.1%} | {c['accuracy']:.3f} |" for c in r["curve"]]

    if len(results) == 2:
        a, b = results
        test = stats.mcnemar(a["correct"], b["correct"])
        diff = stats.paired_difference_ci(a["correct"], b["correct"])
        p_disc = (test["a_only"] + test["b_only"]) / n_test
        lines += ["", f"## Is {b['name']} really better than {a['name']}?", "",
                  f"- Accuracy difference (B − A), all test messages: **{diff}** (paired bootstrap)",
                  f"- Messages only A got right: {test['a_only']}; only B got right: {test['b_only']}",
                  f"- Exact McNemar test p-value: **{stats.format_p(test['p_value'])}**",
                  f"- For reference: detecting a 1-point difference with 80% power would need about "
                  f"{stats.mcnemar_sample_size(max(p_disc, 0.011), 0.01):,} paired test messages "
                  f"(we have {n_test:,})."]

    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "intent_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (config.REPORTS_DIR / "intent_metrics.json").write_text(json.dumps({r["name"]: {
        "banking77_accuracy": vars(r["banking77_acc"]), "account_accuracy": vars(r["account_acc"]),
        "ece_after": r["ece_after"], "coverage": r["coverage"], "auto_accuracy": vars(r["auto_accuracy"]),
    } for r in results}, indent=2))
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(
        {r["name"]: {"temperature": r["temperature"], "threshold": r["threshold"]} for r in results}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
