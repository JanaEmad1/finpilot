"""Database connection and schema."""
from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine, text

from finpilot import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    email      TEXT NOT NULL,
    country    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS accounts (
    id       INTEGER PRIMARY KEY,
    user_id  INTEGER NOT NULL REFERENCES users(id),
    currency TEXT NOT NULL,
    balance  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS cards (
    id      INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    last4   TEXT NOT NULL,
    kind    TEXT NOT NULL,              -- physical / virtual
    status  TEXT NOT NULL DEFAULT 'active'  -- active / frozen
);
CREATE TABLE IF NOT EXISTS merchants (
    id       INTEGER PRIMARY KEY,
    name     TEXT NOT NULL,
    category TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transactions (
    id          INTEGER PRIMARY KEY,
    account_id  INTEGER NOT NULL REFERENCES accounts(id),
    card_id     INTEGER REFERENCES cards(id),
    merchant_id INTEGER REFERENCES merchants(id),
    amount      REAL NOT NULL,          -- negative = money out, positive = money in
    ts          TEXT NOT NULL,          -- ISO timestamp
    kind        TEXT NOT NULL           -- card_payment / salary / transfer_in / transfer_out
);
CREATE INDEX IF NOT EXISTS ix_tx_account_ts ON transactions(account_id, ts);
CREATE INDEX IF NOT EXISTS ix_accounts_user ON accounts(user_id);
CREATE INDEX IF NOT EXISTS ix_cards_user ON cards(user_id);
"""


def make_engine(url: str | None = None) -> Engine:
    return create_engine(url or config.DB_URL, future=True)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return make_engine()


def create_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        for statement in SCHEMA.split(";"):
            if statement.strip():
                conn.execute(text(statement))
