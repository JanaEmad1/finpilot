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

---

## Step 3 — The help centre and the intent dataset

**What we did:**
- Wrote **18 help-centre articles** (`finpilot/kb/*.md`) for a made-up bank called FinPilot:
  lost cards, refunds, transfers, top-ups, exchange rates, fees, identity checks, and so on.
  The policies are invented for the demo — they are not any real bank's rules.
- Each article lists which customer **intents** it answers. Together they cover all 77
  intents of **Banking77**, a public dataset of 13,083 real-style banking questions labelled
  with 77 intents (e.g. `card_arrival`, `lost_or_stolen_card`). A test checks that every
  intent belongs to exactly one article.
  Bonus: this gives us **free ground truth** for testing search later — if the question is
  labelled `card_arrival`, we know which article is the right answer.
- Wrote `finpilot/rag.py` — a TF-IDF keyword search over the articles.
- Added **3 intents of our own** for questions about the customer's own data:
  `balance_query`, `spending_query`, `recent_transactions` (these go to the SQL tools, not
  to an article). They are generated from sentence templates.
- Split the data into **train / validation / test**. Banking77 has no validation set, so we
  hold out 10% of train. Validation is for tuning (calibration, thresholds); the test set is
  touched only for the final numbers.

**Problems we hit:**

1. The normal way to load the dataset failed:
   ```
   RuntimeError: Dataset scripts are no longer supported, but found banking77.py
   ```
   New versions of the Hugging Face `datasets` library no longer run the loading scripts
   that older datasets use.

2. The "converted to Parquet" version didn't exist either:
   ```
   DatasetNotFoundError: Revision 'refs/convert/parquet' doesn't exist for dataset 'PolyAI/banking77'
   ```

3. A popular copy on Hugging Face (`mteb/banking77`) loaded fine — but it had
   **9,993 / 3,076** rows, while the official dataset has **10,003 / 3,080**. Someone had
   changed it. Using it would make our numbers impossible to compare with published results.

4. Our own synthetic examples had two bugs, found by checking the data, not by tests:
   - **Data leakage:** 22 of the 120 synthetic test sentences were *also* in the training set.
     We removed duplicates, but "Check my balance" and "check my balance" counted as different.
     A model tested on sentences it already saw looks better than it really is.
   - **Too few examples:** templates with no blanks to fill (like "check my balance") can only
     make one sentence, so one intent ended up with just 22 training examples.

**How we fixed it:**
- Read the official loading script to find where the data really comes from, and downloaded
  the **original CSV files from PolyAI's GitHub**. Checked them: 10,003 / 3,080 rows, 77 intents,
  no duplicates, and **no test sentence appears in train**.
- Leakage: compare sentences after **normalising** them (lowercase, no punctuation), and split
  the synthetic data **by template** — whole templates are kept only for the test set, so test
  sentences use wording the model has never seen. Added a regression test for it.
- Too few examples: added natural openers and endings ("hey", "could you check", "thanks") to
  every template. Now each account intent has 86–97 training examples (Banking77 classes have
  32–168), and 36 test examples each.

**What we learned:**
- Popular dataset copies can be silently modified. Go back to the original source and
  **check the row counts against the paper**.
- **Data leakage is easy to create and hard to see.** Always check whether test examples also
  appear in train — after normalising the text.
- For template data, a *random* split is not enough; split by template.
- Honest caveat we keep: template sentences are easier than real customer messages, so we will
  report the account-intent accuracy **separately** from Banking77.
