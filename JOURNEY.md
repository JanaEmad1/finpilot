# FinPilot — Build Journey

This file is the honest diary of how FinPilot was built: every step, every error,
how we fixed it, and what we learned. It is written **while** we build, not after.
Nothing here is made up — if something broke, it is written down.

Each step uses the same shape:

- **What we did** — the step in plain words
- **Why** — the reason behind it
- **Problems we hit** — the error, copied as it appeared (or "none")
- **How we fixed it**
- **What we learned**

---

## Step 0 — Choosing the project

**What we did:** Read the Revolut *Deep Learning Engineer (LLM/NLP)* job description and
compared it with my existing projects (hospital chatbot, Distill, multimodal speaker ID,
EquiMonitor thesis).

**Why:** No single project covered everything the job asks for: a fine-tuned NLP model,
an LLM assistant, SQL, production tools, *and* probability & statistics.

**Decision:** Build **FinPilot** — a banking chat assistant that:
- understands what the customer wants (a fine-tuned intent classifier),
- answers questions about their own account (safe SQL tools),
- answers help-centre questions (retrieval / RAG),
- writes friendly replies with an LLM (Gemini),
- and is **evaluated with statistics** (confidence intervals, calibration, A/B tests)
  before anything is "released".

**What we learned:** The statistics part is what most portfolio LLM projects skip, so
it is the part that makes this one stand out.

---

## Step 1 — Setting up the computer and the repo

**What we did:** Checked which tools the laptop has, then created the project folder,
a Python virtual environment, and a git repository.

**Problems we hit:**

1. `python` was not found at all:
   ```
   Python was not found; run without arguments to install from the Microsoft Store...
   ```
   Windows had a fake "python" shortcut that opens the Microsoft Store. The real Python
   lives inside Anaconda (`C:\Users\janae\anaconda3\python.exe`) but is not on PATH.

2. Importing PyTorch in the Anaconda base environment crashed:
   ```
   OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
   ```
   Two different packages in the base environment each shipped their own copy of the
   OpenMP library, and they collided.

3. The laptop has no Docker, no Ollama, and no GitHub CLI, and the GPU (MX570) has only
   2 GB of memory.

**How we fixed it:**
- Used the full path to Anaconda's Python to create a **fresh virtual environment**
  (`.venv`) just for this project. A clean environment has only the packages we need,
  so there is no OpenMP clash. (We did *not* use the "unsafe workaround"
  `KMP_DUPLICATE_LIB_OK=TRUE` the error suggests — it hides the problem instead of fixing it.)
- Installed the **CPU version** of PyTorch (much smaller download; the 2 GB GPU would
  not help much anyway).
- The install took longer than 10 minutes, so it was moved to run in the background
  while we kept working.
- Docker and CI files are still written — GitHub Actions will run them for us even
  though Docker is not installed locally.

**What we learned:**
- One virtual environment per project avoids a whole class of "it breaks on my machine" bugs.
- Read the error message's hint, but think before applying an "unsafe workaround".

---

## Step 2 — A fake (synthetic) bank database and safe SQL tools

**What we did:**
- Wrote `finpilot/data/generate.py`, which builds a SQLite bank database: 500 customers,
  accounts in EUR/GBP/USD, physical and virtual cards, merchants in 9 categories
  (groceries, travel, bills...), and about 250,000 card payments over one year.
- Everything is random but **seeded**, so the same command always makes the same data.
  The dataset has a fixed "today" (1 Sept 2026) so "last month" always means August 2026.
- Wrote `finpilot/tools.py`: the only ways the assistant can touch the database —
  check balances, total spending by category and time period, list recent transactions,
  and freeze cards.
- Wrote `finpilot/periods.py`, which turns words like "last month" or "in March" into dates.
- Wrote 10 tests in `tests/test_tools.py`.

**Why:** Real bank data is private, so we generate realistic fake data. And we decided the
LLM should **never write SQL itself**. Each query is fixed, uses bound parameters, and is
always filtered by the logged-in customer's id. That blocks SQL injection and blocks
"show me someone else's account" by design, not by hoping the model behaves.
Freezing cards changes data, so it always asks "are you sure?" first.

**Problems we hit:**

1. The first database *looked* fine and all tests passed, but a quick sanity query showed:
   ```
   balances min/avg/max (EUR main): (-15843, 15404, 54428)
   negative balances: 29
   zero-balance accounts: 156 (USD, GBP)
   ```
   - 29 customers were deep in debt (one at -15,843 EUR). The reason: how much a person
     spent was chosen at random, **not linked to their salary**, so big spenders with
     small salaries drifted further negative every month.
   - Every second-currency account (USD/GBP) had exactly 0 — nobody ever put money in them.

2. A small maths trap while fixing it: spending amounts follow a *lognormal* distribution.
   Its **average is bigger than its median** (average = median × e^(σ²/2)). Using the
   median to estimate monthly spending would under-count it by about 20%.

(A design detail, not a bug we hit: "may" is both a month and a verb — "may I see my
spending?". We planned for it from the start: the parser only treats it as a month in
phrases like "in May" or "May 2026", and a test checks this.)

**How we fixed it:**
- Spending is now tied to income: each customer spends 40–90% of their salary.
- Each USD/GBP account gets an opening deposit.
- Added a **regression test** (`test_generated_accounts_are_realistic`) so this bug can
  never quietly come back.
- Result after the fix: 0 negative accounts, 0 empty accounts, balances between about
  2,000 and 37,500 EUR.

**What we learned:**
- **Passing tests does not mean the data makes sense.** Always look at the data itself
  (min / max / counts) before building on top of it.
- When you fix a bug, add a test that would have caught it.
- Know your distributions: mean ≠ median for skewed data like money.
