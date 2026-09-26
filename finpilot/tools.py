"""Account tools the assistant can call.

Design rule: the LLM never writes SQL. Every query here is fixed, uses bound
parameters, and is always filtered by the logged-in user's id. That removes
SQL injection and "show me another customer's data" by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sqlalchemy import Engine, text

from finpilot.data.generate import CATEGORIES
from finpilot.periods import Period

CATEGORY_WORDS: dict[str, list[str]] = {
    "groceries": ["grocer", "supermarket", "food shopping"],
    "restaurants": ["restaurant", "eating out", "dining", "takeaway", "cafe", "coffee"],
    "transport": ["transport", "uber", "taxi", "fuel", "petrol", "metro", "bolt"],
    "travel": ["travel", "flight", "hotel", "holiday", "airbnb", "trip"],
    "shopping": ["shopping", "clothes", "amazon", "online shopping"],
    "entertainment": ["entertainment", "cinema", "movies", "games", "concert"],
    "bills": ["bill", "utilities", "electric", "phone bill", "internet"],
    "health": ["health", "pharmacy", "doctor", "medical", "dentist"],
    "subscriptions": ["subscription", "netflix", "spotify", "streaming", "gym"],
}
assert set(CATEGORY_WORDS) == set(CATEGORIES)


def find_category(message: str) -> str | None:
    text_ = message.lower()
    for category, words in CATEGORY_WORDS.items():
        if category in text_ or any(w in text_ for w in words):
            return category
    return None


def as_of(engine: Engine) -> date:
    with engine.connect() as conn:
        value = conn.execute(text("SELECT value FROM meta WHERE key = 'as_of'")).scalar_one()
    return date.fromisoformat(value)


def get_balances(engine: Engine, user_id: int) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, currency, balance FROM accounts
            WHERE user_id = :uid ORDER BY id
        """), {"uid": user_id}).mappings().all()
    return [dict(r) for r in rows]


def spending(engine: Engine, user_id: int, period: Period, category: str | None = None) -> dict:
    """Total card spending in a period, split by category (money out only)."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT m.category, ROUND(-SUM(t.amount), 2) AS total, COUNT(*) AS n
            FROM transactions t
            JOIN accounts a  ON a.id = t.account_id
            JOIN merchants m ON m.id = t.merchant_id
            WHERE a.user_id = :uid
              AND t.amount < 0
              AND t.ts >= :start AND t.ts < :end
              AND (:category IS NULL OR m.category = :category)
            GROUP BY m.category
            ORDER BY total DESC
        """), {"uid": user_id, "start": period.start.isoformat(), "end": period.end.isoformat(),
               "category": category}).mappings().all()
    by_category = [dict(r) for r in rows]
    return {
        "period": period.label,
        "category": category,
        "total": round(sum(r["total"] for r in by_category), 2),
        "by_category": by_category,
    }


def recent_transactions(engine: Engine, user_id: int, limit: int = 5) -> list[dict]:
    limit = max(1, min(int(limit), 20))
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT t.ts, t.amount, a.currency, t.kind,
                   COALESCE(m.name, CASE WHEN t.kind = 'salary' THEN 'Salary' ELSE 'Transfer' END) AS description
            FROM transactions t
            JOIN accounts a ON a.id = t.account_id
            LEFT JOIN merchants m ON m.id = t.merchant_id
            WHERE a.user_id = :uid
            ORDER BY t.ts DESC, t.id DESC
            LIMIT :limit
        """), {"uid": user_id, "limit": limit}).mappings().all()
    return [dict(r) for r in rows]


def list_cards(engine: Engine, user_id: int) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id, last4, kind, status FROM cards WHERE user_id = :uid ORDER BY id"
        ), {"uid": user_id}).mappings().all()
    return [dict(r) for r in rows]


@dataclass
class ActionResult:
    done: bool
    needs_confirmation: bool
    message: str


def freeze_cards(engine: Engine, user_id: int, confirmed: bool) -> ActionResult:
    """Freeze all of the user's active cards. Changes data, so it needs explicit confirmation."""
    active = [c for c in list_cards(engine, user_id) if c["status"] == "active"]
    if not active:
        return ActionResult(False, False, "All your cards are already frozen.")
    names = ", ".join(f"{c['kind']} card ending {c['last4']}" for c in active)
    if not confirmed:
        return ActionResult(False, True, f"I can freeze your {names}. Reply 'yes' to confirm.")
    with engine.begin() as conn:
        conn.execute(text("UPDATE cards SET status = 'frozen' WHERE user_id = :uid AND status = 'active'"),
                     {"uid": user_id})
    return ActionResult(True, False, f"Done — your {names} {'is' if len(active) == 1 else 'are'} now frozen. "
                                     "You can unfreeze in the app at any time.")
