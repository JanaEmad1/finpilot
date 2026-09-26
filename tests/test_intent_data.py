from finpilot.intent.data import ACCOUNT_INTENTS, _normalise, _split_by_template, _synthetic_account_examples


def test_synthetic_split_has_no_leakage():
    # regression test: a random split once put 22 of 120 test sentences in train too
    df = _synthetic_account_examples(per_template=12, seed=42)
    train, test = _split_by_template(df, test_templates_per_intent=3, seed=42)
    assert not set(train.text.map(_normalise)) & set(test.text.map(_normalise))
    assert set(test.label) == set(ACCOUNT_INTENTS)
