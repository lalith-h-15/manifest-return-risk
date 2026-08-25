"""
Return-Risk Scorer API
-----------------------
Serves the trained logistic regression model for scoring individual orders,
batches, and exposing the honest held-out evaluation metrics computed in
model/train.py.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
"""
import json
from pathlib import Path
from typing import Optional, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "return_risk_model.joblib"
THRESHOLD_PATH = BASE_DIR / "model" / "threshold.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

app = FastAPI(title="Return-Risk Scorer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)
with open(METRICS_PATH) as f:
    metrics_blob = json.load(f)

CATEGORIES = ["accessories", "apparel", "beauty", "electronics", "footwear", "grocery", "home"]
PAYMENT_METHODS = ["COD", "UPI", "prepaid_card", "wallet"]


class OrderInput(BaseModel):
    category: str = Field(..., description=f"One of {CATEGORIES}")
    price: float = Field(..., gt=0)
    discount_pct: float = Field(0, ge=0, le=100)
    payment_method: str = Field(..., description=f"One of {PAYMENT_METHODS}")
    customer_order_count: int = Field(1, ge=1)
    customer_return_rate_hist: Optional[float] = Field(None, ge=0, le=1, description="null if first-time customer")
    days_since_signup: int = Field(30, ge=0)
    delivery_days: int = Field(4, ge=0, le=60)
    is_weekend_order: bool = False
    size_variant: bool = False
    coupon_used: bool = False
    review_rating_at_purchase: float = Field(4.0, ge=1, le=5)


class ScoreResponse(BaseModel):
    risk_score: float
    flagged: bool
    threshold_used: float
    top_factors: List[dict]


def _validate_categoricals(order: OrderInput):
    if order.category not in CATEGORIES:
        raise HTTPException(400, f"category must be one of {CATEGORIES}")
    if order.payment_method not in PAYMENT_METHODS:
        raise HTTPException(400, f"payment_method must be one of {PAYMENT_METHODS}")


def _to_frame(order: OrderInput) -> pd.DataFrame:
    return pd.DataFrame([{
        "price": order.price,
        "discount_pct": order.discount_pct,
        "customer_order_count": order.customer_order_count,
        "customer_return_rate_hist": order.customer_return_rate_hist,
        "days_since_signup": order.days_since_signup,
        "delivery_days": order.delivery_days,
        "review_rating_at_purchase": order.review_rating_at_purchase,
        "category": order.category,
        "payment_method": order.payment_method,
        "is_weekend_order": int(order.is_weekend_order),
        "size_variant": int(order.size_variant),
        "coupon_used": int(order.coupon_used),
    }])


@app.get("/")
def root():
    return {"service": "return-risk-scorer", "status": "ok", "model": metrics_blob["metrics"]["model_selected"]}


@app.get("/metrics")
def get_metrics():
    """Held-out test set metrics, exactly as computed at training time. No live recomputation."""
    return metrics_blob


@app.post("/score", response_model=ScoreResponse)
def score_order(order: OrderInput):
    _validate_categoricals(order)
    df = _to_frame(order)
    proba = float(model.predict_proba(df)[0, 1])
    flagged = proba >= threshold

    # simple factor explanation: which categorical bucket + a couple of numeric flags
    # dominate, based on known coefficient signs from training
    factors = []
    if order.payment_method == "COD":
        factors.append({"factor": "Cash on Delivery", "effect": "increases risk"})
    if order.category in ("apparel", "footwear"):
        factors.append({"factor": f"Category: {order.category}", "effect": "increases risk"})
    if order.customer_return_rate_hist and order.customer_return_rate_hist > 0.25:
        factors.append({"factor": "High personal return history", "effect": "increases risk"})
    if order.discount_pct > 40:
        factors.append({"factor": "Deep discount", "effect": "increases risk"})
    if order.review_rating_at_purchase >= 4.5:
        factors.append({"factor": "High product rating", "effect": "decreases risk"})
    if not factors:
        factors.append({"factor": "No strong risk drivers detected", "effect": "neutral"})

    return ScoreResponse(
        risk_score=round(proba, 4),
        flagged=bool(flagged),
        threshold_used=round(float(threshold), 3),
        top_factors=factors[:4],
    )


@app.post("/batch")
def score_batch(orders: List[OrderInput]):
    results = []
    for o in orders:
        results.append(score_order(o))
    return {"count": len(results), "results": results}
