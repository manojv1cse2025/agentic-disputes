# ADC OSCE Simulator — Starter (Frontend + Backend, no AI yet)

This is a fully working local app: signup/login, a case bank, session start/end,
Stripe subscription checkout, and a session page — with the AI layer stubbed out
in exactly one file so you can plug it in later without touching anything else.

## Folder structure

```
adc-osce-app/
  backend/
    .env.example
    package.json
    src/
      server.js              # Express entrypoint
      store.js                # in-memory DB (swap for Postgres later)
      middleware/
        auth.js               # JWT auth check
      routes/
        auth.js               # signup/login
        cases.js               # list/get cases
        sessions.js            # start/end session, get feedback
        payments.js            # Stripe checkout + webhook
      services/
        aiService.js           # <-- THE ONLY FILE YOU REWRITE FOR PHASE 2
      data/
        cases.json             # your case bank (structured, matches earlier schema)
  frontend/
    .env.local.example
    package.json
    next.config.js
    lib/
      api.js                  # fetch helper (adds JWT, handles errors)
    pages/
      _app.js
      index.js                # login/signup
      dashboard.js            # case list, start case, upgrade subscription
      session/
        [caseId].js           # session page (AI mounts here in Phase 2)
    styles/
      globals.css
```

## 1. Run the backend

```bash
cd backend
cp .env.example .env
# open .env and set JWT_SECRET to any random string
# (Stripe keys can stay as placeholders until you want to test payments)
npm install
npm run dev
```

Backend runs at `http://localhost:4000`. Check it's alive:
```bash
curl http://localhost:4000/api/health
```

## 2. Run the frontend

In a second terminal:

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

Frontend runs at `http://localhost:3000`.

## 3. Try the full flow

1. Open `http://localhost:3000`
2. Sign up with any email/password
3. You'll land on the dashboard and see the one seeded case (`case-001`)
4. Click "Start case" → goes to the session page (AI area shows a placeholder box)
5. Click "End case & get feedback" → returns a stub feedback report

Everything — auth, routing, case gating, session lifecycle, Stripe checkout —
is real and working. Only the AI conversation itself is stubbed.

## 4. Setting up real Stripe payments (optional right now)

1. Create a Stripe account, switch to test mode
2. Create a Product + recurring Price, copy its ID into `STRIPE_PRICE_ID_MONTHLY`
3. Copy your test secret key into `STRIPE_SECRET_KEY`
4. For webhooks locally, install the Stripe CLI and run:
   ```bash
   stripe listen --forward-to localhost:4000/api/payments/webhook
   ```
   Copy the printed webhook signing secret into `STRIPE_WEBHOOK_SECRET`
5. Click "Upgrade subscription" on the dashboard — it'll redirect to a real Stripe Checkout page

## 5. Adding your own cases

Edit `backend/src/data/cases.json`. Each case needs:
- `patientProfile` — what the student sees before/during the case
- `historyFacts` — facts the AI patient can reveal when asked (Phase 2 will gate these behind tool calls)
- `examFindings` — same idea for clinical findings
- `rubricItems` — what a "good" answer covers, used for scoring

This structure is intentionally the same one discussed earlier for the LLM
tool-calling design — you're already building your content in the right shape.

## 6. Where the AI layer plugs in later (Phase 2)

**Backend:** `backend/src/services/aiService.js` has three stub functions:
- `startAiSession()` — currently returns a mock message. Later: spins up a
  LiveKit room + starts your agent worker (STT → Claude with tool-calling → TTS → avatar),
  returns real connection details (room name, LiveKit token/URL).
- `recordRubricHit()` — currently just logs. Later: called by the agent worker
  whenever the student triggers a rubric item during conversation.
- `generateFeedbackReport()` — currently returns a stub report. Later: sends
  the transcript + rubric to Claude for narrative feedback, combined with the
  deterministic rubric-hit log.

Nothing else in the app needs to change — `routes/sessions.js` already calls
these functions and passes their output straight to the frontend.

**Frontend:** `frontend/pages/session/[caseId].js` has a clearly marked black
box — that's where you'll mount the LiveKit React SDK video/audio room component
once Phase 2 is ready. The rest of the page (case info, end-case button,
feedback display) is already wired to real backend data.

## 7. Suggested next step

Get comfortable with this flow, add a few more cases to `cases.json`, then
when ready for Phase 2: set up a LiveKit Cloud account, build the Python
agent worker (STT/LLM/TTS/avatar pipeline with tool-calling), and fill in
the three functions in `aiService.js`.
