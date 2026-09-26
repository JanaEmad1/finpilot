"""One interface for both intent models, with temperature-scaled confidence."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from finpilot import config
from finpilot.intent.data import LABELS


@dataclass
class Prediction:
    intent: str
    confidence: float


class IntentModel:
    name = "base"
    temperature = 1.0

    def logits(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def predict(self, texts: list[str]) -> list[Prediction]:
        from eval.stats import softmax
        probs = softmax(self.logits(texts), self.temperature)
        return [Prediction(LABELS[i], float(p[i])) for p, i in zip(probs, probs.argmax(1))]


class BaselineIntentModel(IntentModel):
    name = "tfidf-logreg"

    def __init__(self):
        import joblib
        from finpilot.intent.baseline import PATH
        self.model = joblib.load(PATH)

    def logits(self, texts):
        # log-probabilities work as logits: softmax(log p) == p
        return np.log(self.model.predict_proba(list(texts)) + 1e-12)


class TransformerIntentModel(IntentModel):
    name = "distilbert-finetuned"

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from finpilot.intent.train import OUT_DIR
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(OUT_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(OUT_DIR).eval()
        assert [self.model.config.id2label[i] for i in range(len(LABELS))] == LABELS

    def logits(self, texts, batch_size: int = 128):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(list(texts[i:i + batch_size]), padding=True, truncation=True,
                                     max_length=64, return_tensors="pt")
                out.append(self.model(**enc).logits.numpy())
        return np.concatenate(out)


CALIBRATION_FILE = config.MODELS_DIR / "calibration.json"


@lru_cache(maxsize=1)
def load_intent_model() -> IntentModel:
    """Serve the "champion" chosen by eval/eval_intent.py.

    A model is NOT used just because it exists: the first version preferred DistilBERT
    whenever its folder was present, which would have shipped the statistically WORSE
    model (see JOURNEY.md step 6).
    """
    saved = json.loads(CALIBRATION_FILE.read_text()) if CALIBRATION_FILE.exists() else {}
    champion = saved.get("champion", BaselineIntentModel.name)
    model: IntentModel = (TransformerIntentModel() if champion == TransformerIntentModel.name
                          else BaselineIntentModel())
    model.temperature = saved.get(model.name, {}).get("temperature", 1.0)
    return model
