# Agentic Purchase Dispute Evidence Engine

**Track:** AI Risk Manager — Razorpay Buildathon 2026

## The problem, in one example

Priya sets up an AI assistant: *"order dinner when I'm working late, under
Rs.1,500, food only."* She sets this once and forgets about it.

Weeks later a Rs.4,200 charge appears. She doesn't recognise it and tells her
bank she never made this payment. The bank opens a dispute. The merchant now
has to prove the purchase was legitimate, or they lose the money automatically.

Normally that proof is easy: IP address, device fingerprint, click path, time
on site. All of it assumes a *human* browsed and clicked. Here, an agent
decided. That evidence doesn't exist, and disputes involving AI buyers have no
way to be defended.

## The gap this fills

Razorpay's dispute API already accepts evidence types for exactly this
situation: `shipping_proof`, `billing_proof`, `proof_of_service`,
`customer_communication`, `access_activity_log`, `refund_cancellation_policy`,
and so on. Every one of them answers a single question — **did the merchant
deliver what was ordered?**

None of them answer the question an agent-made purchase actually raises:
**was the buyer authorised to order it?** When an agent overspends its budget
or gets manipulated by planted content, the merchant delivered exactly what
was asked for. Every existing evidence field says "yes, delivered." The
dispute isn't about delivery, it's about authority, and there's no field for
that.

This project is that missing evidence type: given a mandate (what the user
authorised), a transaction, and the record of what the agent read and decided,
produce a verdict — who is actually at fault — with evidence a merchant could
submit.

## Four possible verdicts

| Verdict | Meaning | Who's liable |
|---|---|---|
| `WITHIN_MANDATE` | Agent stayed inside its authority | Customer (regret, not fraud) |
| `MANDATE_BREACH` | Agent exceeded cap, category, or expiry | Agent operator |
| `INJECTED` | Merchant content instructed the agent and it complied | Merchant / third party |
| `AMBIGUOUS` | Genuinely undecidable from the record | Flagged for human review |

## How it works

- **`generator.py`** — builds labelled synthetic disputes. Ten scenario types,
  including deliberately hard ones: a charge that's fine alone but breaks the
  weekly cap, injected text disguised as ordinary menu copy, injected text the
  agent *saw but ignored*, and cases with no clean answer at all.
- **`classify.py`** — the engine. Mandate checks (expiry, per-transaction cap,
  cumulative window cap, category) are pure deterministic rules — a cap breach
  is arithmetic, not a judgment call. Content analysis checks two things
  separately: did merchant content address an automated buyer with a
  directive, and did the agent's own reasoning show it acted on that
  directive. Only both together produce `INJECTED`. Output is a full evidence
  pack: findings, spend breakdown, content analysis, confidence, and an
  auto-drafted explanation letter.
- **`score.py`** — scores any classifier against the labelled set. Reports
  accuracy, a confusion matrix, and two costed error types: **false blame**
  (an innocent merchant blamed for manipulation) and **missed injection** (a
  real manipulation not caught).

## Run it

```bash
python generator.py    # builds scenarios.json, prints label distribution
python score.py         # scores the naive keyword baseline
python classify.py      # prints sample evidence packs
```

The regex-vs-baseline comparison, scored on a **held-out set using injection
phrasing neither classifier was written against**:

```bash
python -c "from generator import generate; from score import evaluate, naive_baseline; from classify import classify; evaluate(generate(200, held_out=True), naive_baseline, 'baseline'); evaluate(generate(200, held_out=True), classify, 'rule engine')"
```

The LLM-based injection detector needs a Gemini API key (free tier, no card
-- see the setup note at the top of `gemini_classify.py`):

```bash
set GEMINI_API_KEY=your-key-here      # Windows cmd
python run_llm_eval.py --n 60         # runs all three approaches side by side
```

## Results (n=60, held-out phrasing)

Three approaches, same held-out scenarios, same scoring:

| | Accuracy | False blame | Notes |
|---|---|---|---|
| Naive baseline (keywords + cap check) | 46.0% | 0 | fails every hard case type |
| Rule engine, regex injection detection | 73.0% | 0 | mandate rules perfect, injection detection collapses to 0% on unseen phrasing |
| Rule engine, LLM injection detection | **91.7%** | **0** | mandate rules perfect, injection detection generalises to phrasing it was never tuned on |

Every mandate-rule case type — cap, window, category, expiry — scores **100%**
on held-out data in all three versions, because those are deterministic
arithmetic and don't need a model. The difference between the rows is
entirely in the one genuinely linguistic judgment: whether a piece of
merchant content is instructing an automated buyer, and whether the agent
complied.

The regex version was tuned against specific phrasing and scored 100% on
that phrasing, then collapsed to 0% the moment the same attacks were
reworded (see `generate(n, held_out=True)` in `generator.py`). That gap is
the actual finding of this project, not a footnote: pattern-matching on
vocabulary doesn't generalise, and the LLM version, given the same two
questions, does.

The one honest weak spot in the best version: `injection_followed` (blatant,
obvious manipulation attempts) scores 56% (5/9), lower than `subtle_injection`
(86%) or `injection_ignored` (100%). **Crucially, every one of those 5 misses
fails safe** — they're classified as `MANDATE_BREACH` rather than
`WITHIN_MANDATE`, so the agent operator is still held liable and no innocent
merchant is ever blamed. False blame is 0 across all 60 scenarios and all
three approaches. That's the metric that actually matters for a dispute
system: an evidence engine that's occasionally too cautious about *why* a
charge was wrong is far safer than one that's ever wrong about *who* to
blame.

**Both the regex and LLM numbers are reported, not just the better one**,
because a classifier that's only ever checked against the phrasing it was
tuned on is measuring nothing.

## What's deliberately not here

No dashboard, no live Razorpay API integration, no database, no auth. The
brief asks for a measured match rate and an honest exception list on
synthetic data — that's the entire scope, and every hour spent past that is
an hour not spent making the numbers real.

## Files

- `generator.py` — builds labelled synthetic disputes, including a held-out
  mode with injection phrasing no classifier here was tuned against
- `classify.py` — the rule-based engine: deterministic mandate checks plus
  a regex content analyser, full evidence packs, explanation letters
- `score.py` — scoring harness: accuracy, confusion matrix, false blame,
  missed injection, per-case-type breakdown
- `gemini_classify.py` — swaps the regex content analyser for a real LLM
  call (Google Gemini, free tier), same interface, mandate rules untouched
- `run_llm_eval.py` — runs baseline, regex, and LLM versions side by side
  on the same held-out set
