"""Help-centre retrieval.

Two retrievers, compared in eval/ab_test.py:
  A. TfidfRetriever        — classic keyword search over the articles.
  B. intent-routed         — use the intent classifier's prediction to pick the article
                             directly, and fall back to A when the classifier is unsure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from finpilot import config


@dataclass(frozen=True)
class Article:
    slug: str
    title: str
    intents: tuple[str, ...]
    body: str = field(repr=False)


def _parse(path: Path) -> Article:
    text = path.read_text(encoding="utf-8")
    _, header, body = text.split("---", 2)
    meta = dict(line.split(":", 1) for line in header.strip().splitlines())
    intents = tuple(i.strip() for i in meta["intents"].split(",") if i.strip())
    return Article(path.stem, meta["title"].strip(), intents, body.strip())


@lru_cache(maxsize=1)
def load_articles(kb_dir: Path = config.KB_DIR) -> tuple[Article, ...]:
    return tuple(_parse(p) for p in sorted(kb_dir.glob("*.md")))


def intent_to_article(articles: tuple[Article, ...] | None = None) -> dict[str, Article]:
    articles = articles or load_articles()
    mapping: dict[str, Article] = {}
    for article in articles:
        for intent in article.intents:
            if intent in mapping:
                raise ValueError(f"intent {intent!r} is in two articles: {mapping[intent].slug}, {article.slug}")
            mapping[intent] = article
    return mapping


class TfidfRetriever:
    def __init__(self, articles: tuple[Article, ...] | None = None):
        self.articles = articles or load_articles()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(f"{a.title}\n{a.body}" for a in self.articles)

    def search(self, query: str, k: int = 3) -> list[tuple[Article, float]]:
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        top = np.argsort(-scores)[:k]
        return [(self.articles[i], float(scores[i])) for i in top]

    def search_many(self, queries: list[str], k: int = 3) -> list[list[str]]:
        """Top-k article slugs for many queries at once (used by the evaluation)."""
        scores = (self.vectorizer.transform(queries) @ self.matrix.T).toarray()
        top = np.argsort(-scores, axis=1)[:, :k]
        return [[self.articles[i].slug for i in row] for row in top]
