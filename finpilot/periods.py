"""Turn phrases like "last month" or "in March" into a date range [start, end)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]


@dataclass(frozen=True)
class Period:
    start: date  # inclusive
    end: date    # exclusive
    label: str


def _month_start(d: date) -> date:
    return d.replace(day=1)


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    return date(d.year + y, m + 1, 1)


def parse_period(message: str, as_of: date) -> Period:
    """Find a time period in the message. Default: the last 30 days."""
    text = message.lower()

    if "last month" in text or "previous month" in text:
        start = _add_months(_month_start(as_of), -1)
        return Period(start, _month_start(as_of), start.strftime("%B %Y"))
    if "this month" in text:
        start = _month_start(as_of)
        return Period(start, as_of + timedelta(days=1), "this month so far")
    if "last week" in text:
        this_monday = as_of - timedelta(days=as_of.weekday())
        return Period(this_monday - timedelta(days=7), this_monday, "last week")
    if "this week" in text:
        this_monday = as_of - timedelta(days=as_of.weekday())
        return Period(this_monday, as_of + timedelta(days=1), "this week so far")
    if "this year" in text:
        return Period(date(as_of.year, 1, 1), as_of + timedelta(days=1), f"{as_of.year} so far")
    if "last year" in text:
        return Period(date(as_of.year - 1, 1, 1), date(as_of.year, 1, 1), str(as_of.year - 1))
    if text_match := re.search(r"last (\d{1,3}) days", text):
        n = int(text_match.group(1))
        return Period(as_of - timedelta(days=n - 1), as_of + timedelta(days=1), f"the last {n} days")
    if "today" in text:
        return Period(as_of, as_of + timedelta(days=1), "today")
    if "yesterday" in text:
        return Period(as_of - timedelta(days=1), as_of, "yesterday")

    for i, name in enumerate(MONTHS, start=1):
        # word boundary so "may" in "may I..." is only used when it looks like a month
        if re.search(rf"\b(in|during|for|of) {name}\b", text) or re.search(rf"\b{name} \d{{4}}\b", text):
            year = as_of.year if i <= as_of.month else as_of.year - 1
            if year_match := re.search(rf"{name} (\d{{4}})", text):
                year = int(year_match.group(1))
            start = date(year, i, 1)
            return Period(start, _add_months(start, 1), start.strftime("%B %Y"))

    return Period(as_of - timedelta(days=29), as_of + timedelta(days=1), "the last 30 days")
