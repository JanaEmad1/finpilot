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

(Small slip at the end of step 3: the downloaded dataset CSVs were accidentally committed
with the code. They are downloaded by the script, so they don't belong in the repo. The commit
had not been pushed yet, so we amended it and added `data/banking77/` to `.gitignore`.)

---

## Step 4 — Statistics toolkit and a simple baseline model

**What we did:**
- Wrote `eval/stats.py`, the statistics used to decide if a model is "good enough to release":
  - **Bootstrap confidence intervals** — resample the test set 2,000 times to see how much a
    score could move just by luck. We report `0.915 [0.905, 0.924]` instead of just `91.5%`.
  - **McNemar test** — compares two models on the *same* questions, looking only at the
    questions where they disagree.
  - **Power analysis** — how many test questions we need to detect a difference of a given size.
  - **Calibration**: *ECE* (does "90% confident" really mean right 90% of the time?) and
    **temperature scaling**, a one-number fix for over- or under-confident models.
  - **Coverage vs. accuracy** — if the bot only answers when it's confident enough, how many
    messages does it answer, and how accurate is it on those?
- 7 tests for the statistics, using simulated data where we *know* the right answer
  (e.g. we generate data with temperature 2.5 and check that we recover about 2.5).
- Trained a **baseline**: TF-IDF features + logistic regression (`finpilot/intent/baseline.py`).
  It trains in about 76 seconds. Any fancy model has to beat this to be worth it.
- Wrote `eval/eval_intent.py`, which writes `reports/intent_eval.md`.

**Baseline results (test set):**
- Banking77 accuracy **0.915 [0.905, 0.924]** — surprisingly strong for such a simple model.
- Our account intents: **0.917 [0.861, 0.963]** — note the much wider interval: only 108 test
  examples, so we know this number much less precisely.
- Calibration surprise: the best temperature was **0.72**, i.e. below 1. The model was
  *under*-confident (neural networks are usually the opposite). Calibration brought the error
  (ECE) from 0.072 down to **0.010**.
- **Hand-off rule:** we picked the confidence threshold on the *validation* set to reach 97%
  accuracy. On the test set it held: the bot answers **84.5%** of messages at **97.3%**
  accuracy and passes the rest to a human.

**Problems we hit:**
1. Printing the report crashed on Windows:
   ```
   UnicodeEncodeError: 'charmap' codec can't encode character '→'
   ```
   The Windows console uses an old text encoding (cp1252) that has no "→" arrow.
   The report *file* was fine (we write it as UTF-8); only printing failed.
   **Fix:** `sys.stdout.reconfigure(encoding="utf-8")` at the start of the script.
2. The A/B test printed **"p-value: 0"**. A probability is never exactly zero — the real
   value was just smaller than the smallest number a computer float can store.
   **Fix:** print `< 1e-300` in that case.

**What we learned:**
- Always start with a simple baseline. Here it set a high bar (91.5%).
- A number without an uncertainty range can mislead — 108 examples vs. 3,080 examples give
  very different certainty even at the same accuracy.
- Tune thresholds and calibration on validation data, then check them **once** on test.

---

## Step 5 — Fine-tuning DistilBERT (and making training fast enough on a laptop)

**What we did:**
- Wrote `finpilot/intent/train.py`: fine-tunes **DistilBERT** (a smaller, faster version of BERT)
  on our 80 intents with a plain PyTorch training loop — AdamW optimiser, learning-rate warm-up
  then linear decay, gradient clipping, and we keep the epoch with the best **validation**
  accuracy (never choosing by test score).
- The laptop has no usable GPU, so everything trains on the CPU.

**Problems we hit:**
1. **Training was far too slow.** After 6 minutes it had not finished 50 steps — it was heading
   for 2+ hours. The cause: we padded *every* sentence to 64 tokens, but the sentences are short:
   ```
   tokens per sentence: mean 16.1, p95 36, max 98, share >64: 0.0029
   ```
   About **75% of the computation was spent on padding** (empty filler tokens).
   **Fix:** *dynamic padding* — each batch is only padded to its own longest sentence.
2. It was still slow (6.8 s/step). Instead of guessing, we **measured** (`scripts/bench_training.py`):

   | Setup | seconds per step |
   |---|---|
   | pad to 64, 12 threads (original) | 3.71 |
   | pad to 64, 10 threads | 3.32 |
   | dynamic padding, 12 threads | 2.85 |
   | **dynamic padding, 10 threads** | **2.51** |

   Two surprises:
   - The CPU (Intel i7-1255U) has 10 physical cores but 12 "logical" ones. Using all 12 threads
     was *slower* than 10 — the extra hyper-threads just add overhead.
   - The benchmark (3.71 s) was much faster than the 6.8 s we saw during real training. The reason
     was us: we ran evaluations and a demo **at the same time** as training, and they fought over the CPU.
   **Fix:** dynamic padding + 10 threads (a new `--threads` option), and leave the CPU alone while training.
3. Small one: running the benchmark as `python scripts/bench_training.py` failed with
   `ModuleNotFoundError: No module named 'finpilot'`, because running a file directly doesn't add the
   project folder to Python's import path. **Fix:** run it as a module, `python -m scripts.bench_training`.

**What we learned:**
- **Measure before optimising.** A two-minute benchmark beat guessing and gave a 1.5× speed-up.
- Padding is not free — look at your real sequence lengths.
- "Use all the cores" is not always fastest, especially on laptop chips with mixed core types.
- Don't benchmark (or train) while other heavy jobs are running.

**Result of training:** 3 epochs took **68 minutes**. Validation accuracy went 76.4% → 87.7% →
**89.7%** — about the same as the baseline (89.6%). Real speed during training was 4–6 s/step,
slower than the 2.5 s/step in the short benchmark; a laptop chip usually slows down under a long,
heavy load, so short benchmarks are optimistic.

---

## Step 6 — The results, and a release decision we did not expect

**What we did:** Evaluated both models on the test set (3,188 messages) with the statistics from
step 4, and re-ran the retrieval A/B test.

**Intent results:**

| Model | Banking77 accuracy | Account intents | Bot answers (at 97% target) |
|---|---|---|---|
| TF-IDF + logistic regression | **0.915** [0.905, 0.924] | 0.917 | **84.5%** of messages |
| DistilBERT (3 CPU epochs) | 0.903 [0.893, 0.913] | 0.852 | 79.0% of messages |

- Is DistilBERT really better? **No — it is significantly *worse*.** Paired difference
  **−1.3 points [−2.4, −0.3]**, exact McNemar test **p = 0.011**. The whole confidence interval is
  below zero, so this is not bad luck.
- Why? Most likely it is **under-trained**: only 3 epochs (published Banking77 results usually train
  longer, on GPUs). A clue: its best temperature was 0.73 (below 1), meaning it is still
  *under-confident* — typical of a network that hasn't finished learning.

**Problem we hit — the big one:** our code would have **shipped the worse model anyway**.
`load_intent_model()` used DistilBERT automatically whenever its folder existed, because "fine-tuned
transformer" *sounds* better. Nothing checked the numbers.

**How we fixed it:** a **champion / challenger** rule. After evaluation, a new model replaces the
current one ("champion") only if it is **significantly better** (whole CI above zero and p < 0.05).
The decision is saved, and the app serves whatever the evaluation chose. Today that is the baseline.
There is also a **release gate** (`eval/gate.py`, run in CI) that fails the build if a model's
accuracy — judged by the *lower end* of its confidence interval — drops below a minimum bar.

**Retrieval A/B test:** routing a question to an article by its predicted intent (B) beat plain
TF-IDF search (A): **0.909 vs 0.526** top-1 accuracy, +38 points [+36.6, +40.0], p < 1e-300.
We state the caveat openly: it's not a fair fight (B learned from ~9,000 labelled questions, A from
none), so the size of the gap is the interesting part, not that B wins.

**What we learned:**
- **Bigger / newer is not automatically better.** A simple model with good features can beat a
  transformer that is trained on a small budget.
- The value of a release process is that it can say **"no"** — here it stopped us from shipping a
  model that was worse, which we would otherwise have done without noticing.
- A negative result, measured properly, is still a result worth publishing.
- Next thing to try: train DistilBERT for more epochs (ideally on a GPU, e.g. free Google Colab) and
  let the same statistical test decide again.

---

## Step 7 — Guardrails, the agent, the LLM and the API

**What we did:**
- `finpilot/guardrails.py` — hides personal data (card numbers, IBANs, emails, phone numbers,
  CVV codes) **before** the message reaches the model, the LLM or the logs, and flags obvious
  prompt-injection attempts ("ignore previous instructions...", "show another customer's balance").
  Card numbers are confirmed with the **Luhn checksum** (the check digit every real card number
  has), so random long numbers like order ids are left alone.
- `finpilot/llm.py` — calls **Gemini** over plain HTTPS to write the final reply. The LLM gets
  only the facts (tool results or the article) and strict rules. If there is no API key, or
  Gemini is down, we use a template answer instead, so the chat never breaks.
- `finpilot/agent.py` — the "brain": guardrails → intent → (not confident? hand off to a human)
  → SQL tool or help article → reply. For a lost or stolen card it also offers to freeze the
  cards, and only does it after the customer says "yes".
- A **number check**: every number in the LLM's reply must appear in the facts; otherwise the
  reply is thrown away and the template is used. A made-up amount is the worst mistake a bank
  assistant can make.
- `finpilot/api.py` — FastAPI with `POST /chat` and `GET /health`. It logs the decision,
  confidence and speed of every chat, but **never the raw message** (it could contain personal data).
- 12 new tests. The agent tests use a tiny keyword "model" so they run offline in milliseconds.

**Problems we hit:**
1. A test failed: an order number (`1234567890123`) was hidden as a **phone number**.
   The card rule correctly skipped it (it fails the Luhn check), but the phone rule treated
   *any* long run of digits as a phone number.
   **Fix:** phones must look like phones — start with `+`, or with `0`, or be written in groups
   like `555-123-4567`. Added tests for real phone formats.
2. A bug caught while reading my own code, before any test ran: the number check first
   used `"100".rstrip(".0")`, which turns **100 into 1** (it strips *every* trailing 0 and dot).
   An LLM saying "1 payment" would have "matched" a fact of "100 payments".
   **Fix:** compare numbers as real numbers (`1,234.50` → `1234.50`), and a test for exactly this case.
3. The end-to-end demo showed a real weakness: *"how long does an international transfer take?"*
   was handed to a human, because the baseline model was only 53% sure — it could not decide
   between `pending_transfer` and `transfer_timing`, two Banking77 intents that overlap.
   This is the hand-off rule doing its job (better to ask a human than guess), but it is also a
   sign the model needs to be better on similar intents — one reason to try a transformer.
4. Running the demo *changed the demo database*: saying "yes" really froze customer 7's cards.
   That is correct behaviour, but it means the database should be rebuilt before a clean demo
   (`python -m finpilot.data.generate`).

**What we learned:**
- Safety comes mainly from **architecture** (the LLM can't run SQL or take actions by itself),
  and only secondarily from filters. Filters like regexes are easy to get too broad or too narrow.
- Re-read small "clever" string tricks — `rstrip` does not do what it looks like it does.
- An end-to-end demo finds different problems than unit tests do.

---

## Step 8 — Packaging: Docker, CI and the README

**What we did:**
- `Dockerfile` + `docker-compose.yml`: the image installs CPU-only PyTorch (much smaller), builds
  the synthetic database and the baseline model, and runs the API.
- `.github/workflows/ci.yml`: on every push GitHub runs the tests, retrains the baseline, runs the
  evaluation and the **release gate**, then builds the Docker image and sends a real chat request to it.
- `README.md`: what the project does, the design decisions, the real results (copied from
  `reports/`), how to run it, and its limitations.
- `.env.example` shows where the Gemini key goes; the real `.env` is git-ignored.

**Honest note:** Docker is not installed on this laptop, so the Dockerfile and the CI workflow have
**not been run yet** at the time of writing. The first push to GitHub will be their first real test —
if something fails there, it will be recorded here as the next step.

**What we learned:** CI lets a free cloud machine test things (like Docker) that the laptop can't.

---

## Step 9 — The first CI run on GitHub failed

**What happened:** After pushing to GitHub, the very first CI step failed:
```
Unit tests (offline, no API keys needed)
Process completed with exit code 4.
```
Exit code 4 from pytest means it crashed *before running any test*. We reproduced it on the
laptop by running `pytest` exactly like CI does:
```
ImportError while loading conftest 'tests/conftest.py'.
E   ModuleNotFoundError: No module named 'finpilot'
```

**Why:** On the laptop we always ran `python -m pytest`, and that form quietly adds the current
folder to Python's import path. CI runs plain `pytest`, which does **not** — so Python could not
find our own `finpilot` package. The tests "passed on my machine" only because of *how* we started them.

**How we fixed it:** Added a small `pytest.ini` with `pythonpath = .`, so the project folder is
always on the import path, no matter how pytest is started. Ran plain `pytest` locally: 33 passed.

**What we learned:**
- "Works on my machine" often hides a difference in *how* the command is run. Reproduce CI's exact
  command locally.
- Know your tool's exit codes: pytest's exit code 1 = tests failed, 4 = pytest itself couldn't start.
- This is exactly why CI exists — it caught a problem that our local setup was hiding.

**Result after pushing the fix:** the whole CI pipeline passed on GitHub — unit tests, baseline
training, evaluation, the release gate, **and** the Docker build with a real chat request sent to the
running container. That closes the "not run yet" note from step 8: the Dockerfile worked on its
first real run, even though it was written on a laptop without Docker.
