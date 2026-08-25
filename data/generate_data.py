"""
Synthetic e-commerce order dataset generator for a return-risk scorer.

We simulate orders with features that are known (from real e-commerce/BFSI
fraud & returns literature) to correlate with return probability:
  - category (apparel/footwear have high size-related returns)
  - price & discount depth (heavy discounts -> more impulse buys -> more returns)
  - payment method (COD orders return far more than prepaid, since there's no
    sunk cost at purchase time)
  - customer's historical return rate (past behavior predicts future behavior)
  - customer tenure / order count (newer accounts, less predictable)
  - delivery delay (late delivery slightly raises return odds)
  - size-variant flag (sizing-sensitive items return more)
  - coupon usage

Noise is added throughout so no single feature is deterministic - the model
has to actually learn a combination, and a ceiling on achievable AUC exists
(as in real data).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 25000

categories = ["apparel", "footwear", "electronics", "home", "beauty", "accessories", "grocery"]
category_base_return_rate = {
    "apparel": 0.22, "footwear": 0.20, "electronics": 0.09,
    "home": 0.07, "beauty": 0.05, "accessories": 0.08, "grocery": 0.02,
}
category_probs = [0.22, 0.15, 0.18, 0.15, 0.12, 0.10, 0.08]

payment_methods = ["COD", "prepaid_card", "UPI", "wallet"]
payment_probs = [0.42, 0.28, 0.22, 0.08]
payment_return_multiplier = {"COD": 1.9, "prepaid_card": 0.7, "UPI": 0.85, "wallet": 0.9}

rows = []
for i in range(N):
    category = rng.choice(categories, p=category_probs)
    base_rate = category_base_return_rate[category]

    price = float(np.round(np.exp(rng.normal(6.5, 1.0)), 2))  # skewed, ~ INR 300-15000
    price = min(price, 50000)

    discount_pct = float(np.clip(rng.beta(2, 5) * 100, 0, 90))

    payment_method = rng.choice(payment_methods, p=payment_probs)

    customer_order_count = int(rng.negative_binomial(3, 0.4)) + 1
    # historical return rate: newer customers get a noisy prior; repeat
    # customers have a persistent personal tendency
    personal_tendency = np.clip(rng.beta(2, 8), 0, 1)  # most people rarely return
    if customer_order_count <= 1:
        customer_return_rate_hist = np.nan  # no history yet
    else:
        n_hist = customer_order_count - 1
        hist_returns = rng.binomial(n_hist, personal_tendency)
        customer_return_rate_hist = hist_returns / n_hist

    days_since_signup = int(rng.exponential(180)) + 1
    delivery_days = int(np.clip(rng.normal(4, 2), 1, 15))
    is_weekend_order = int(rng.random() < 0.3)
    size_variant = int(category in ("apparel", "footwear") and rng.random() < 0.85)
    coupon_used = int(rng.random() < 0.35)
    review_rating_at_purchase = float(np.round(np.clip(rng.normal(4.1, 0.6), 1, 5), 1))

    # ---- latent return probability (ground truth generative process) ----
    logit = np.log(base_rate / (1 - base_rate))
    logit += 0.55 * (payment_return_multiplier[payment_method] - 1)
    logit += 0.006 * (discount_pct - 20)
    logit += 0.35 * size_variant
    logit += -0.15 * np.log1p(price / 1000)  # very cheap items returned slightly more per-unit
    logit += 0.10 * max(delivery_days - 5, 0)
    logit += -0.35 * (review_rating_at_purchase - 4.0)
    logit += 0.20 * is_weekend_order * 0
    if not np.isnan(customer_return_rate_hist):
        logit += 1.8 * (customer_return_rate_hist - base_rate)
    else:
        logit += 0.10  # slight new-customer uncertainty penalty

    logit += rng.normal(0, 0.6)  # irreducible noise
    p_return = 1 / (1 + np.exp(-logit))
    returned = int(rng.random() < p_return)

    rows.append({
        "order_id": f"ORD{100000+i}",
        "category": category,
        "price": price,
        "discount_pct": round(discount_pct, 1),
        "payment_method": payment_method,
        "customer_order_count": customer_order_count,
        "customer_return_rate_hist": customer_return_rate_hist,
        "days_since_signup": days_since_signup,
        "delivery_days": delivery_days,
        "is_weekend_order": is_weekend_order,
        "size_variant": size_variant,
        "coupon_used": coupon_used,
        "review_rating_at_purchase": review_rating_at_purchase,
        "returned": returned,
    })

df = pd.DataFrame(rows)
df.to_csv("/home/claude/return-risk-manager/data/orders.csv", index=False)
print(df.shape)
print(df["returned"].mean())
print(df.groupby("category")["returned"].mean())
print(df.groupby("payment_method")["returned"].mean())
