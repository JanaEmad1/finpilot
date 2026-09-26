"""Baseline intent classifier: TF-IDF features + logistic regression.

The fine-tuned transformer has to beat this — with statistical evidence — to be worth
its extra cost. Run:  python -m finpilot.intent.baseline
"""
from __future__ import annotations

import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from finpilot import config
from finpilot.intent.data import LABELS, label_ids, load_splits

PATH = config.MODELS_DIR / "baseline.joblib"


def build_pipeline():
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
    )
    return make_pipeline(features, LogisticRegression(C=20, max_iter=3000))


def main() -> None:
    splits = load_splits()
    ids = label_ids()
    y_train = splits["train"].label.map(ids).to_numpy()
    start = time.time()
    model = build_pipeline().fit(splits["train"].text, y_train)
    # every label must be a class, in LABELS order, so logits line up with label ids
    assert list(model.classes_) == list(range(len(LABELS)))
    y_val = splits["val"].label.map(ids).to_numpy()
    acc = float(np.mean(model.predict(splits["val"].text) == y_val))
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PATH)
    print(f"trained in {time.time() - start:.0f}s, val accuracy {acc:.3f}, saved {PATH}")


if __name__ == "__main__":
    main()
