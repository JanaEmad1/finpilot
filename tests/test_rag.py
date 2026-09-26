from finpilot.intent.data import BANKING77_LABELS
from finpilot.rag import TfidfRetriever, intent_to_article, load_articles


def test_every_banking77_intent_has_exactly_one_article():
    mapping = intent_to_article()  # raises if an intent is listed twice
    assert set(mapping) == set(BANKING77_LABELS)


def test_articles_parse():
    articles = load_articles()
    assert len(articles) == 18
    assert all(a.title and a.body for a in articles)


def test_tfidf_finds_obvious_article():
    retriever = TfidfRetriever()
    top_article, _ = retriever.search("my card was stolen, what do I do?", k=1)[0]
    assert top_article.slug == "lost-stolen-or-compromised-card"
