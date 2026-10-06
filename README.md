# FinPilot

[![CI](https://github.com/JanaEmad1/finpilot/actions/workflows/ci.yml/badge.svg)](https://github.com/JanaEmad1/finpilot/actions/workflows/ci.yml)

**An LLM banking assistant with a statistically evaluated release process.**

<p align="center"><img src="docs/demo.png" width="640" alt="FinPilot answering spending, balance and lost-card questions, refusing a prompt injection and handing off when unsure"></p>

FinPilot answers customer questions for a (fictional) digital bank: *"How much did I spend
on groceries last month?"*, *"My card was stolen"*, *"Why was I charged an exchange fee?"*.
It uses an intent classifier (a TF-IDF baseline vs. a fine-tuned DistilBERT, with the
winner chosen by a significance test), safe SQL tools, help-centre retrieval and an LLM
(Gemini). Each model decision is backed by statistics: confidence intervals,
calibration, paired significance tests and a release gate in CI.

> 📓 **[JOURNEY.md](JOURNEY.md)** is the honest build diary. It records each step, each
> error we hit, how we fixed it and what we learned.

---

## How it works

```
customer message
   │
   ▼
guardrails ── redact card numbers / IBAN / email / phone, block prompt injection
   │
   ▼
intent classifier (80 intents, calibrated confidence)
   │
   ├── confidence < threshold ──────────────► hand off to a human agent
   ├── account question ────► SQL tool (fixed query, scoped to this user) ─┐
   ├── help question ───────► help-centre article for that intent ─────────┤
   └── lost / stolen card ──► article + "freeze your cards?" (needs "yes") ┤
                                                                           ▼
                                  LLM writes the reply from these facts only
                                  (every number must appear in the facts,
                                   otherwise a safe template is used)
```

### Design decisions

| Decision | Why |
|---|---|
| **The LLM never writes SQL.** It only phrases answers. | Fixed, parameterised queries filtered by the logged-in user's id rule out SQL injection and cross-customer data leaks by design. |
| **Actions need explicit confirmation.** | Freezing cards changes data, so the assistant asks first and acts only after "yes". |
| **Hand-off threshold is chosen from data.** | It is picked on the validation set to reach 97% accuracy on the messages the bot answers itself. |
| **Number grounding check.** | If the LLM's reply contains a number that is not in the facts, the reply is discarded. A made-up amount is the worst error a bank assistant can make. |
| **Personal data is redacted before the model and the logs.** | The classifier, the LLM and the logs never see card numbers or IBANs. Logs record decisions, never raw messages. |
| **Works without an API key.** | Tests, CI and the demo fall back to template answers, so nothing depends on a paid service. |

## Results

### Intent classifier (test set, 3,188 messages)

| Model | Banking77 accuracy | Account-intent accuracy* | Calibration error (ECE) after temp. scaling | Bot answers (at ≥97% target) |
|---|---|---|---|---|
| **TF-IDF + logistic regression** (served) | **0.915** [0.905, 0.924] | 0.917 [0.861, 0.963] | 0.010 | **84.5%** at 0.973 accuracy |
| DistilBERT, fine-tuned 3 epochs on CPU | 0.903 [0.893, 0.913] | 0.852 [0.787, 0.917] | 0.014 | 79.0% at 0.972 accuracy |

\* template-generated intents from held-out templates, reported separately.

**Release decision: the baseline stays in production.** The fine-tuned DistilBERT was
**significantly worse**: paired difference −1.3 points [−2.4, −0.3], exact McNemar p = 0.011.
It is most likely under-trained (3 CPU epochs, and a fitted temperature below 1 shows it is still
under-confident). The champion/challenger rule in `eval/eval_intent.py` promotes a new model
only if it is significantly better, so a model is never shipped just because it is newer or bigger.

**Hand-off to humans:** the confidence threshold is chosen on validation data to reach 97% accuracy
on messages the bot answers itself. On the test set it held (97.3%) while the bot handled 84.5% of messages.

<p align="center"><img src="docs/automation_tradeoff.png" width="680" alt="Accuracy on answered messages falls as the share answered by the bot rises; the chosen point is 84.5% answered at 97.3% accuracy"></p>
<p align="center"><img src="docs/model_comparison.png" width="680" alt="TF-IDF baseline vs DistilBERT with 95% confidence intervals"></p>

### Help-centre retrieval: offline A/B test (3,080 paired questions)

| System | Top-1 correct article |
|---|---|
| A: TF-IDF search over articles | 0.526 [0.509, 0.544] |
| **B: route by predicted intent, TF-IDF fallback** | **0.909** [0.899, 0.919] |

<p align="center"><img src="docs/retrieval_ab.png" width="640" alt="Top-1 correct article: A 52.6%, B 90.9%"></p>

Difference +0.383 [0.366, 0.400], McNemar p < 1e-300 → **ship B**. This is not a like-for-like
comparison, because B uses a supervised classifier. See [reports/retrieval_ab_test.md](reports/retrieval_ab_test.md).

Full reports: [reports/intent_eval.md](reports/intent_eval.md) · [reports/retrieval_ab_test.md](reports/retrieval_ab_test.md)

All numbers come from `reports/` and are regenerated by the commands below.
Intervals are 95% bootstrap confidence intervals. The charts are drawn from the same reports by
`python scripts/make_charts.py` (needs `matplotlib`).

## Business impact

**What a bank gets:** 84.5% of customer messages answered instantly and around the clock, at a measured 97.3%
accuracy. The other 15.5% go to a person *with the intent already identified*, and no answer ever contains a
made-up amount.

An illustrative estimate. The rates come from the test set; the volume and costs are assumptions to replace with the
bank's own figures:

| Assumption | Value |
|---|---|
| Customer messages per month | 10,000 |
| Agent handling time per message | 4 minutes |
| Fully loaded agent cost | $30 / hour |

| Outcome | Calculation | Per month |
|---|---|---|
| Messages answered by the bot | 10,000 × 84.5% | **8,450** |
| Agent hours freed | 8,450 × 4 min | **≈ 563 hours** (≈ 3.5 full-time agents) |
| Cost avoided | 563 h × $30 | **≈ $16,900** |
| Wrong automated answers | 8,450 × (1 − 97.3%) | ≈ 228 |

**The threshold is a business lever, not a technical detail.** Moving along the trade-off curve to 88.6% automation
(threshold 0.70) automates ~410 more messages a month but raises wrong answers from ≈ 228 to ≈ 337 (8,860 × 3.8%).
Whether that is worth it depends on what a wrong answer costs the bank, so the threshold is set from a stated accuracy
target and not left at a default.

**Risk controls that make it deployable:** card numbers and IBANs are redacted before any model or log sees them, the LLM
cannot query data (fixed SQL only), actions like freezing a card require a "yes", and the CI release gate blocks
any model that is not significantly better than the current one.

## Data

- **Intents:** [Banking77](https://arxiv.org/abs/2003.04807) (13,083 customer questions, 77 intents,
  CC BY 4.0), downloaded from the [original source](https://github.com/PolyAI-LDN/task-specific-datasets)
  plus 3 template-generated account intents (`balance_query`, `spending_query`, `recent_transactions`).
  The synthetic test sentences come from **held-out templates** to avoid leakage.
- **Bank data:** a synthetic, seeded SQLite database (500 customers, ~250k card payments).
  No real customer data.
- **Help centre:** 18 short articles written for this project. The policies are invented
  for the demo and are not any real bank's rules.

## Run it

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m finpilot.data.generate        # build the synthetic bank database
python -m finpilot.intent.baseline      # train the baseline (~1-2 min)
python -m finpilot.intent.train         # fine-tune DistilBERT (optional, slow on CPU)
python -m eval.eval_intent              # -> reports/intent_eval.md
python -m eval.ab_test                  # -> reports/retrieval_ab_test.md
python -m eval.gate                     # release gate (exit code 1 = do not ship)

cp .env.example .env                    # optional: add GEMINI_API_KEY for LLM-written replies
uvicorn finpilot.api:app --reload       # http://localhost:8000/docs
pytest -q
```

Or with Docker: `docker compose up --build`.

Example request:

```bash
curl -X POST localhost:8000/chat -H "content-type: application/json" \
     -d '{"user_id": 7, "message": "How much did I spend on groceries last month?"}'
```

## Project layout

```
finpilot/
  agent.py        decides what to do with each message
  api.py          FastAPI app (POST /chat, GET /health)
  guardrails.py   PII redaction and prompt-injection check
  llm.py          Gemini client (optional)
  tools.py        safe SQL tools
  periods.py      "last month" -> dates
  rag.py          help-centre retrieval
  kb/             help-centre articles
  intent/         data loading, baseline, DistilBERT fine-tuning, prediction
  data/           synthetic database generator
eval/
  stats.py        bootstrap CIs, McNemar, power analysis, calibration
  eval_intent.py  intent evaluation report
  ab_test.py      retrieval A/B test report
  gate.py         release gate used in CI
tests/            unit tests (run offline, no API key)
reports/          generated evaluation reports
```

## Limitations and next steps

- The account intents are template-generated and easier than real messages; they are
  reported separately.
- The retrieval A/B test compares an unsupervised retriever with a supervised router,
  so the gap is expected. The next step is embedding-based retrieval for the fallback path.
- `user_id` comes from the request body for the demo. A real system would take it from an
  authenticated session.
- SQLite fits a demo; production would use Postgres.
- Not done yet: an online A/B test with real traffic, multi-turn memory beyond the freeze
  confirmation, and a C++/Java component.
