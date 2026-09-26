"""The assistant's brain: decides what to do with each message.

    message
      → guardrails (redact personal data, flag prompt injection)
      → pending "freeze my card?" confirmation?
      → intent classifier (calibrated confidence)
      → confidence too low?          → hand off to a human agent
      → account question?            → SQL tool (always scoped to this user)
      → help-centre question?        → the intent's article
      → lost / compromised card?     → article + offer to freeze (needs "yes")
      → LLM phrases the answer from the facts (number check), or a template if no LLM
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from sqlalchemy import Engine

from finpilot import config, guardrails, llm, tools
from finpilot.intent.predict import CALIBRATION_FILE, IntentModel
from finpilot.periods import parse_period
from finpilot.rag import intent_to_article

FREEZE_INTENTS = {"lost_or_stolen_card", "compromised_card", "card_swallowed"}
YES = re.compile(r"^\s*(yes|yep|yeah|y|confirm|do it|please do|ok|okay)\b", re.I)


@dataclass
class Reply:
    answer: str
    intent: str | None = None
    confidence: float | None = None
    route: str = ""               # tool / article / handoff / refusal / action
    sources: list[str] = field(default_factory=list)
    redacted: list[str] = field(default_factory=list)
    used_llm: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class Agent:
    def __init__(self, engine: Engine, intent_model: IntentModel, handoff_threshold: float | None = None):
        self.engine = engine
        self.intent_model = intent_model
        self.articles = intent_to_article()
        self.threshold = handoff_threshold if handoff_threshold is not None else self._threshold_from_eval()
        self.pending_freeze: set[str] = set()  # session ids waiting for a "yes"

    def _threshold_from_eval(self) -> float:
        if CALIBRATION_FILE.exists():
            saved = json.loads(CALIBRATION_FILE.read_text()).get(self.intent_model.name)
            if saved:
                return float(saved["threshold"])
        return config.HANDOFF_THRESHOLD

    # ------------------------------------------------------------------
    def chat(self, user_id: int, message: str, session_id: str = "default") -> Reply:
        guard = guardrails.check(message)
        if guard.injection:
            return Reply("I can only help with your own account and our help-centre topics. "
                         "What would you like to do?", route="refusal", redacted=guard.redacted)

        if session_id in self.pending_freeze:
            self.pending_freeze.discard(session_id)
            if YES.match(guard.text):
                result = tools.freeze_cards(self.engine, user_id, confirmed=True)
                return Reply(result.message, route="action", redacted=guard.redacted)

        prediction = self.intent_model.predict([guard.text])[0]
        base = dict(intent=prediction.intent, confidence=round(prediction.confidence, 3), redacted=guard.redacted)

        if prediction.confidence < self.threshold:
            return Reply("I'm not completely sure I understood, so I'm passing you to a member of our "
                         "support team — they'll reply here shortly.", route="handoff", **base)

        if prediction.intent in ("balance_query", "spending_query", "recent_transactions"):
            facts, template = self._account_facts(user_id, prediction.intent, guard.text)
            return self._phrase(guard.text, facts, template, route="tool",
                                sources=[f"tool:{prediction.intent}"], **base)

        article = self.articles[prediction.intent]
        facts = f"Help-centre article: {article.title}\n{article.body}"
        template = f"{_first_paragraph(article.body)}\n\n(From our help centre: \"{article.title}\")"
        if prediction.intent in FREEZE_INTENTS:
            offer = tools.freeze_cards(self.engine, user_id, confirmed=False)
            if offer.needs_confirmation:
                self.pending_freeze.add(session_id)
            reply = self._phrase(guard.text, facts, template, route="article", sources=[article.slug], **base)
            reply.answer += f"\n\n{offer.message}"
            return reply
        return self._phrase(guard.text, facts, template, route="article", sources=[article.slug], **base)

    # ------------------------------------------------------------------
    def _account_facts(self, user_id: int, intent: str, text: str) -> tuple[str, str]:
        if intent == "balance_query":
            balances = tools.get_balances(self.engine, user_id)
            lines = [f"{b['currency']} account: {b['balance']:,.2f} {b['currency']}" for b in balances]
            return "Balances:\n" + "\n".join(lines), "Your balance:\n" + "\n".join(lines)

        if intent == "spending_query":
            period = parse_period(text, tools.as_of(self.engine))
            category = tools.find_category(text)
            result = tools.spending(self.engine, user_id, period, category)
            what = f"on {category}" if category else "in total"
            head = f"You spent {result['total']:,.2f} EUR {what} in {period.label}."
            rows = [f"- {r['category']}: {r['total']:,.2f} EUR ({r['n']} payments)" for r in result["by_category"][:5]]
            body = "\n".join(rows) if (rows and not category) else ""
            return (f"Card spending {what}, period {period.label}: {result['total']:,.2f} EUR\n{body}",
                    f"{head}\n{body}".strip())

        txs = tools.recent_transactions(self.engine, user_id, limit=5)
        rows = [f"- {t['ts'][:10]}  {t['description']}: {t['amount']:,.2f} {t['currency']}" for t in txs]
        return "Latest transactions:\n" + "\n".join(rows), "Your latest transactions:\n" + "\n".join(rows)

    def _phrase(self, question: str, facts: str, template: str, **reply_fields) -> Reply:
        answer = llm.generate(question, facts)
        if answer and _numbers_are_grounded(answer, facts):
            return Reply(answer, used_llm=True, **reply_fields)
        return Reply(template, **reply_fields)


def _first_paragraph(body: str) -> str:
    return body.split("\n\n")[0].strip()


def _numbers_are_grounded(answer: str, facts: str) -> bool:
    """Every number in the LLM's answer must appear in the facts. A made-up amount is the
    worst possible mistake for a bank assistant, so we fall back to the template instead."""
    def numbers(s: str) -> set[str]:
        # "1,234.50" and "1234.5" are the same number -> compare as rounded floats
        return {f"{float(n.replace(',', '')):.2f}" for n in re.findall(r"\d[\d,]*(?:\.\d+)?", s)}
    return numbers(answer) <= numbers(facts)
