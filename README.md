# Manifest — Return-Risk Desk

**Track 02: AI Risk Manager.** A return-risk scorer for e-commerce orders: flags orders likely
to be returned so a merchant can intervene (e.g. nudge toward prepaid, add sizing info, hold for
manual review) before shipping. Strictly defense-only — the model scores orders for human review
and never autonomously blocks, cancels, or charges anything.

## What's in here

```
data/generate_data.py    synthetic e-commerce order dataset (25,000 orders)
model/train.py           train/val/test split, model selection, cost-based threshold, honest eval
backend/main.py          FastAPI service serving the trained model
frontend/                React app (also ships as a standalone artifact with the model embedded)
```

## The honest part

- **Real train/val/test split** (70/15/15), stratified. The test set is touched exactly once,
  after the model and threshold are already locked in.
- **Two models compared** (logistic regression, gradient boosting) on validation PR-AUC; logistic
  regression won and was carried to test.
- **Threshold chosen by expected cost, not accuracy or F1.** We priced a missed return (false
  negative) at ~₹250 (reverse logistics + restocking) against a false alarm (false positive) at
  ~₹40 (review friction), then swept thresholds on the validation set to minimize total ₹ cost.
- **A naive baseline is reported alongside the model, and it isn't flattered.** "Flag every
  apparel/footwear order" beats the model on precision (0.307 vs 0.265) and F1 (0.413 vs 0.402).
  The model wins on recall (0.832 vs 0.633) and on total cost (46% cost reduction vs. no
  intervention) because its threshold is deliberately tuned for the cost asymmetry. We're
  reporting where the naive rule wins rather than picking a threshold that hides it.

Full metrics live in `model/metrics.json` after running `train.py`, and are surfaced verbatim in
the `/metrics` API endpoint and the frontend's evaluation panel — no numbers are hand-edited for
the demo.

### Held-out test set (n=3,750)

| Metric | Value |
|---|---|
| ROC-AUC | 0.739 |
| PR-AUC | 0.391 |
| Precision | 0.265 |
| Recall | 0.832 |
| F1 | 0.402 |
| Confusion matrix | TN 1527 · FP 1550 · FN 113 · TP 560 |
| Expected cost | ₹90,250 (vs ₹168,250 with no intervention) |

## Running it

### 1. Regenerate data + retrain (optional — model artifacts are already committed)
```bash
cd data && python3 generate_data.py
cd ../model && python3 train.py
```

### 2. Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Endpoints: `GET /metrics`, `POST /score`, `POST /batch`.

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```
Opens at `http://localhost:5173`.

**Note on the frontend/model coupling:** the shipped frontend scores orders client-side using the
exact coefficients extracted from the trained logistic regression (same math the backend runs),
so the demo works standalone without CORS/deployment wiring. For production, point the "Inspect
order" button at `POST /score` on the FastAPI backend instead — the request/response shapes
already match.

## Known limitations (said out loud, not buried)

- Data is synthetic. Category base rates, payment-method multipliers, and the size-variant effect
  were set from typical Indian e-commerce return-rate ranges reported in industry writeups, but
  they are illustrative, not fitted to real merchant data. Before production use this needs to be
  retrained on the merchant's actual order/return history.
- `customer_return_rate_hist` is the single strongest feature (coef 0.41 standardized) and is only
  available for repeat customers — first-time-customer orders lean more on category and payment
  method, which is exactly where the naive baseline is competitive.
- Cost assumptions (₹250 missed return, ₹40 false alarm) are illustrative placeholders — swap in
  the merchant's real reverse-logistics and review-friction costs before shipping a threshold.
