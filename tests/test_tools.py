from datetime import date

from sqlalchemy import text

from finpilot import tools
from finpilot.periods import parse_period

AS_OF = date(2026, 9, 1)


def test_parse_period_last_month():
    p = parse_period("how much did I spend last month?", AS_OF)
    assert (p.start, p.end) == (date(2026, 8, 1), date(2026, 9, 1))


def test_parse_period_named_month_uses_past_year_when_needed():
    p = parse_period("spending in november", AS_OF)
    assert p.start == date(2025, 11, 1) and p.end == date(2025, 12, 1)


def test_parse_period_does_not_confuse_may_the_verb():
    p = parse_period("may I see my spending", AS_OF)
    assert p.label == "the last 30 days"


def test_find_category():
    assert tools.find_category("How much on Uber rides?") == "transport"
    assert tools.find_category("what's my balance") is None


def test_balance_matches_sum_of_transactions(engine):
    balances = tools.get_balances(engine, user_id=1)
    with engine.connect() as conn:
        total = conn.execute(text(
            "SELECT ROUND(SUM(amount), 2) FROM transactions WHERE account_id = :a"
        ), {"a": balances[0]["id"]}).scalar_one()
    assert balances[0]["balance"] == total


def test_generated_accounts_are_realistic(engine):
    # regression test: the first generator produced deeply negative and empty accounts
    with engine.connect() as conn:
        negative = conn.execute(text("SELECT COUNT(*) FROM accounts WHERE balance < 0")).scalar_one()
        empty = conn.execute(text("SELECT COUNT(*) FROM accounts WHERE balance = 0")).scalar_one()
    assert negative == 0 and empty == 0


def test_spending_only_counts_this_user(engine):
    period = parse_period("last month", tools.as_of(engine))
    user1 = tools.spending(engine, 1, period)
    user2 = tools.spending(engine, 2, period)
    with engine.connect() as conn:
        everyone = conn.execute(text("""
            SELECT ROUND(-SUM(amount), 2) FROM transactions
            WHERE amount < 0 AND merchant_id IS NOT NULL AND ts >= :s AND ts < :e
        """), {"s": period.start.isoformat(), "e": period.end.isoformat()}).scalar_one()
    assert 0 < user1["total"] < everyone
    assert user1["total"] != user2["total"]


def test_spending_category_filter(engine):
    period = parse_period("last month", tools.as_of(engine))
    result = tools.spending(engine, 1, period, category="groceries")
    assert [r["category"] for r in result["by_category"]] in ([], ["groceries"])


def test_recent_transactions_limit_is_clamped(engine):
    assert len(tools.recent_transactions(engine, 1, limit=1000)) == 20


def test_freeze_needs_confirmation(engine):
    first = tools.freeze_cards(engine, 3, confirmed=False)
    assert first.needs_confirmation and not first.done
    assert all(c["status"] == "active" for c in tools.list_cards(engine, 3))

    second = tools.freeze_cards(engine, 3, confirmed=True)
    assert second.done
    assert all(c["status"] == "frozen" for c in tools.list_cards(engine, 3))
    # other users are untouched
    assert all(c["status"] == "active" for c in tools.list_cards(engine, 4))
