"""Release gate: fail (exit code 1) if a model is below the quality bar.

We compare the LOWER end of the 95% confidence interval with the bar, not the point
estimate — "probably at least this good", not "good on average on this sample".
Run after eval.eval_intent:  python -m eval.gate [--model tfidf-logreg]
"""
from __future__ import annotations

import argparse
import json
import sys

from finpilot import config

BARS = {
    "banking77_accuracy": 0.88,  # lower CI bound of Banking77 test accuracy
    "auto_accuracy": 0.95,       # lower CI bound of accuracy on messages the bot answers itself
}
MIN_COVERAGE = 0.70              # the bot must still answer at least 70% of messages
MAX_ECE = 0.05


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tfidf-logreg")
    args = parser.parse_args()
    metrics = json.loads((config.REPORTS_DIR / "intent_metrics.json").read_text())[args.model]

    failures = [f"{name}: lower CI bound {metrics[name]['low']:.3f} < {bar}"
                for name, bar in BARS.items() if metrics[name]["low"] < bar]
    if metrics["coverage"] < MIN_COVERAGE:
        failures.append(f"coverage {metrics['coverage']:.3f} < {MIN_COVERAGE}")
    if metrics["ece_after"] > MAX_ECE:
        failures.append(f"calibration error {metrics['ece_after']:.3f} > {MAX_ECE}")

    for f in failures:
        print("GATE FAILED —", f)
    if not failures:
        print(f"Gate passed for {args.model}.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
