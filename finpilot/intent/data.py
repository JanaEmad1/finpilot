"""Load the Banking77 intent dataset (+ our 3 account-data intents).

We download the ORIGINAL CSVs from PolyAI's GitHub instead of a Hugging Face mirror:
the official loader script no longer runs with `datasets>=4`, and the popular mirror
we checked (mteb/banking77) has 9,993/3,076 rows instead of the official 10,003/3,080.
See JOURNEY.md step 3.
"""
from __future__ import annotations

import random
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from finpilot import config

BASE_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
DATA_DIR = config.ROOT / "data" / "banking77"

BANKING77_LABELS = [
    "Refund_not_showing_up", "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed", "cancel_transfer",
    "card_about_to_expire", "card_acceptance", "card_arrival", "card_delivery_estimate", "card_linking",
    "card_not_working", "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card", "contactless_not_working",
    "country_support", "declined_card_payment", "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits", "edit_personal_details",
    "exchange_charge", "exchange_rate", "exchange_via_app", "extra_charge_on_statement", "failed_transfer",
    "fiat_currency_support", "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
    "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal", "pending_top_up",
    "pending_transfer", "pin_blocked", "receiving_money", "request_refund", "reverted_card_payment?",
    "supported_cards_and_currencies", "terminate_account", "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits",
    "top_up_reverted", "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds", "verify_top_up",
    "virtual_card_not_working", "visa_or_mastercard", "why_verify_identity",
    "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal",
]

# Questions about the customer's own data, answered by SQL tools instead of articles.
ACCOUNT_INTENTS = ["balance_query", "spending_query", "recent_transactions"]
LABELS = BANKING77_LABELS + ACCOUNT_INTENTS

_TEMPLATES = {
    "balance_query": [
        "what's my balance", "how much money do I have", "how much is in my {acc} account",
        "show me my balance", "what is my current balance", "can you tell me my {acc} balance",
        "how much money is left in my account", "balance please", "do I have enough money in my account",
        "check my balance", "what do I have in my {acc} account right now", "how much cash is in my account",
    ],
    "spending_query": [
        "how much did I spend {period}", "how much did I spend on {cat} {period}",
        "what did I spend on {cat}", "show my spending {period}", "total spending on {cat} {period}",
        "how much money went on {cat} {period}", "break down my spending {period}",
        "what are my biggest expenses {period}", "how much have I spent on {cat}",
        "am I spending too much on {cat}", "summary of my {cat} spending {period}",
        "where did my money go {period}",
    ],
    "recent_transactions": [
        "show my recent transactions", "what were my last {n} payments", "list my latest transactions",
        "what did I buy recently", "show me my last {n} transactions", "recent activity on my account",
        "what was my last payment", "show my transaction history", "my latest card payments",
        "what came out of my account recently", "show the last {n} things I paid for",
    ],
}
_FILL = {
    "acc": ["", "main", "euro", "EUR", "GBP", "dollar"],
    "period": ["last month", "this month", "last week", "this year", "in march", "in the last 30 days", ""],
    "cat": ["groceries", "restaurants", "eating out", "uber", "travel", "shopping", "bills",
            "subscriptions", "netflix", "coffee", "flights", "the gym"],
    "n": ["3", "5", "10", "few", "two"],
}
# Openers/endings people add to any request. Without them, templates with no {slots}
# produced only one unique sentence each (see JOURNEY.md step 3).
_OPENERS = ["", "hi ", "hey ", "hello, ", "please ", "can you tell me ", "quick question: ",
            "i want to know ", "could you check ", "i need to know "]
_ENDINGS = ["", " please", " thanks", " thank you", " now", " right now", " asap"]


def _normalise(text: str) -> str:
    return " ".join("".join(c for c in text.lower() if c.isalnum() or c.isspace()).split())


def _synthetic_account_examples(per_template: int, seed: int) -> pd.DataFrame:
    """Template-generated examples, with a `template` column so we can split by template.

    Honest caveat: these are easier than real user text, so accuracy on these 3 classes
    is optimistic (reported separately in the eval).
    """
    rng = random.Random(seed)
    rows = []
    seen: set[str] = set()  # de-duplicate on *normalised* text (see JOURNEY.md step 3)
    for intent, templates in _TEMPLATES.items():
        for t_id, template in enumerate(templates):
            made = 0
            for _ in range(per_template * 30):
                if made >= per_template:
                    break
                core = template.format(**{k: rng.choice(v) for k, v in _FILL.items()})
                text = " ".join((rng.choice(_OPENERS) + core + rng.choice(_ENDINGS)).split())
                if rng.random() < 0.4:
                    text = text.capitalize() + rng.choice(["?", "", "!", "."])
                if _normalise(text) in seen:
                    continue
                seen.add(_normalise(text))
                rows.append({"text": text, "label": intent, "template": f"{intent}:{t_id}"})
                made += 1
    return pd.DataFrame(rows)


def _split_by_template(df: pd.DataFrame, test_templates_per_intent: int, seed: int):
    """Hold out whole templates for the test set, so test sentences use phrasings the
    model never saw in training (a random split leaked near-copies into the test set)."""
    rng = random.Random(seed)
    test_templates = set()
    for intent in _TEMPLATES:
        templates = sorted(df.loc[df.label == intent, "template"].unique())
        test_templates.update(rng.sample(templates, test_templates_per_intent))
    is_test = df.template.isin(test_templates)
    return df[~is_test].drop(columns="template"), df[is_test].drop(columns="template")


def download(force: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        path = DATA_DIR / f"{split}.csv"
        if force or not path.exists():
            urllib.request.urlretrieve(f"{BASE_URL}/{split}.csv", path)


def load_splits(seed: int = 42, val_size: float = 0.1) -> dict[str, pd.DataFrame]:
    """Return train / val / test DataFrames with columns text, label.

    Banking77 has no validation split, so we hold out 10% of train (stratified).
    The validation set is used for calibration and picking the hand-off threshold —
    never the test set, otherwise the test numbers would be optimistic.
    """
    download()
    train = pd.read_csv(DATA_DIR / "train.csv").rename(columns={"category": "label"})
    test = pd.read_csv(DATA_DIR / "test.csv").rename(columns={"category": "label"})
    assert sorted(train.label.unique()) == sorted(BANKING77_LABELS)

    synthetic = _synthetic_account_examples(per_template=12, seed=seed)
    syn_train, syn_test = _split_by_template(synthetic, test_templates_per_intent=3, seed=seed)
    train = pd.concat([train, syn_train], ignore_index=True)
    test = pd.concat([test, syn_test], ignore_index=True)

    train, val = train_test_split(train, test_size=val_size, stratify=train.label, random_state=seed)
    return {name: df.reset_index(drop=True) for name, df in
            {"train": train, "val": val, "test": test}.items()}


def label_ids() -> dict[str, int]:
    return {label: i for i, label in enumerate(LABELS)}


if __name__ == "__main__":
    splits = load_splits()
    for name, df in splits.items():
        print(name, len(df), "rows,", df.label.nunique(), "labels")
        print("   account intents:", df[df.label.isin(ACCOUNT_INTENTS)].label.value_counts().to_dict())
