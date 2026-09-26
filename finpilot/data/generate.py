"""Generate a synthetic bank database.

Everything is random but *seeded*, so the same command always builds the same data.
Run:  python -m finpilot.data.generate  [--users 500] [--days 365] [--seed 7]
"""
from __future__ import annotations

import argparse
import random
from datetime import date, datetime, timedelta

import numpy as np
from faker import Faker
from sqlalchemy import Engine, text

from finpilot import config
from finpilot.db import create_schema, make_engine

# The dataset's "today". Fixed so that "last month" means the same thing every run.
AS_OF = date(2026, 9, 1)

# category -> (example merchants, visits per month, median amount in EUR)
CATEGORIES: dict[str, tuple[list[str], float, float]] = {
    "groceries":     (["Lidl", "Carrefour", "Tesco", "Aldi", "Spinneys"], 8.0, 35.0),
    "restaurants":   (["Pizza Hut", "Nando's", "Local Cafe", "Sushi Bar", "Burger King"], 5.0, 22.0),
    "transport":     (["Uber", "Bolt", "Metro Card", "Shell", "TotalEnergies"], 6.0, 15.0),
    "travel":        (["Ryanair", "Booking.com", "Airbnb", "EasyJet"], 0.4, 180.0),
    "shopping":      (["Amazon", "Zara", "IKEA", "H&M", "Apple Store"], 2.5, 60.0),
    "entertainment": (["Cinema City", "Steam", "Ticketmaster"], 1.5, 25.0),
    "bills":         (["Vodafone", "Electric Co", "Water Utility", "Home Internet"], 3.0, 45.0),
    "health":        (["Pharmacy", "City Clinic", "Dentist"], 0.8, 40.0),
    "subscriptions": (["Netflix", "Spotify", "iCloud", "Gym Membership"], 3.0, 11.0),
}
CURRENCIES = ["EUR", "EUR", "EUR", "GBP", "USD"]
AMOUNT_SIGMA = 0.6
# Expected monthly spend of a "spend_level = 1" customer. The mean of a lognormal
# is median * exp(sigma^2 / 2), which is why the median alone under-counts.
BASE_MONTHLY_SPEND = sum(per_month * median * np.exp(AMOUNT_SIGMA**2 / 2)
                         for _, per_month, median in CATEGORIES.values())


def build(engine: Engine, n_users: int = 500, days: int = 365, seed: int = 7) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    fake = Faker()
    Faker.seed(seed)

    create_schema(engine)
    with engine.begin() as conn:
        for table in ["transactions", "cards", "accounts", "merchants", "users", "meta"]:
            conn.execute(text(f"DELETE FROM {table}"))

        conn.execute(text("INSERT INTO meta(key, value) VALUES ('as_of', :d)"), {"d": AS_OF.isoformat()})

        merchants, merchant_ids = [], {}
        for category, (names, _, _) in CATEGORIES.items():
            for name in names:
                merchant_ids.setdefault(category, []).append(len(merchants) + 1)
                merchants.append({"id": len(merchants) + 1, "name": name, "category": category})
        conn.execute(text("INSERT INTO merchants(id, name, category) VALUES (:id, :name, :category)"), merchants)

        users, accounts, cards, txs = [], [], [], []
        start = AS_OF - timedelta(days=days)
        for uid in range(1, n_users + 1):
            users.append({
                "id": uid, "name": fake.name(), "email": fake.email(),
                "country": fake.country_code(),
                "created_at": (start - timedelta(days=int(rng.integers(0, 900)))).isoformat(),
            })
            # every user has a main account; some have a second currency account
            user_accounts = [len(accounts) + 1]
            accounts.append({"id": len(accounts) + 1, "user_id": uid, "currency": "EUR", "balance": 0.0})
            if rng.random() < 0.3:
                accounts.append({"id": len(accounts) + 1, "user_id": uid,
                                 "currency": random.choice(CURRENCIES[3:]), "balance": 0.0})
                user_accounts.append(len(accounts))
                deposit = float(np.round(rng.uniform(50, 1500), 2))
                txs.append(_tx(len(accounts), None, None, deposit,
                               _at(start + timedelta(days=int(rng.integers(0, days))), 12, rng), "transfer_in"))

            card_id = len(cards) + 1
            cards.append({"id": card_id, "user_id": uid, "last4": f"{rng.integers(0, 10000):04d}",
                          "kind": "physical", "status": "active"})
            if rng.random() < 0.5:
                cards.append({"id": len(cards) + 1, "user_id": uid, "last4": f"{rng.integers(0, 10000):04d}",
                              "kind": "virtual", "status": "active"})

            main = user_accounts[0]
            salary = float(np.round(rng.lognormal(np.log(2200), 0.35), 2))
            # People spend 40-90% of what they earn. (First version picked spending at random,
            # independent of salary, and some accounts drifted to -15,000 — see JOURNEY.md step 2.)
            spend_level = salary * rng.uniform(0.4, 0.9) / BASE_MONTHLY_SPEND
            opening = float(np.round(rng.uniform(200, 3000), 2))
            txs.append(_tx(main, None, None, opening, datetime.combine(start, datetime.min.time()), "transfer_in"))

            for day_offset in range(days):
                day = start + timedelta(days=day_offset)
                if day.day == 25:
                    txs.append(_tx(main, None, None, salary, _at(day, 9, rng), "salary"))
                for category, (_, per_month, median) in CATEGORIES.items():
                    for _ in range(rng.poisson(per_month * spend_level / 30)):
                        amount = -float(np.round(rng.lognormal(np.log(median), AMOUNT_SIGMA), 2))
                        txs.append(_tx(main, card_id, random.choice(merchant_ids[category]),
                                       amount, _at(day, 8 + int(rng.integers(0, 14)), rng), "card_payment"))

        conn.execute(text("INSERT INTO users(id, name, email, country, created_at) "
                          "VALUES (:id, :name, :email, :country, :created_at)"), users)
        conn.execute(text("INSERT INTO accounts(id, user_id, currency, balance) "
                          "VALUES (:id, :user_id, :currency, :balance)"), accounts)
        conn.execute(text("INSERT INTO cards(id, user_id, last4, kind, status) "
                          "VALUES (:id, :user_id, :last4, :kind, :status)"), cards)
        conn.execute(text("INSERT INTO transactions(account_id, card_id, merchant_id, amount, ts, kind) "
                          "VALUES (:account_id, :card_id, :merchant_id, :amount, :ts, :kind)"), txs)
        # balance = sum of all money in and out
        conn.execute(text("""
            UPDATE accounts SET balance = ROUND(COALESCE(
                (SELECT SUM(amount) FROM transactions t WHERE t.account_id = accounts.id), 0), 2)
        """))

    return {"users": len(users), "accounts": len(accounts), "cards": len(cards), "transactions": len(txs)}


def _at(day: date, hour: int, rng: np.random.Generator) -> datetime:
    return datetime(day.year, day.month, day.day, hour, int(rng.integers(0, 60)))


def _tx(account_id, card_id, merchant_id, amount, ts: datetime, kind: str) -> dict:
    return {"account_id": account_id, "card_id": card_id, "merchant_id": merchant_id,
            "amount": amount, "ts": ts.isoformat(timespec="minutes"), "kind": kind}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts = build(make_engine(), args.users, args.days, args.seed)
    print(f"Built {config.DB_PATH}: {counts}")


if __name__ == "__main__":
    main()
