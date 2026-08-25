"""
Train + evaluate the return-risk scorer.

Honesty requirements (per hackathon bar):
  - Held-out test set, never touched during training/tuning.
  - Report precision, recall, F1, ROC-AUC, PR-AUC on the held-out set.
  - Report a confusion matrix at the operating threshold we actually ship.
  - Cost-based threshold selection: false positives (flagging a good order
    as high-risk -> friction, e.g. forcing prepaid / manual review on a
    customer who would not have returned) are cheaper than false negatives
    (missing an order that gets returned -> shipping + reverse-logistics +
    restocking loss). We pick the threshold that minimizes expected cost,
    not the one that maximizes accuracy.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, confusion_matrix,
                              precision_score, recall_score, f1_score)
import joblib

df = pd.read_csv("/home/claude/return-risk-manager/data/orders.csv")

FEATURES_NUM = ["price", "discount_pct", "customer_order_count",
                 "customer_return_rate_hist", "days_since_signup",
                 "delivery_days", "review_rating_at_purchase"]
FEATURES_CAT = ["category", "payment_method"]
FEATURES_BIN = ["is_weekend_order", "size_variant", "coupon_used"]
TARGET = "returned"

X = df[FEATURES_NUM + FEATURES_CAT + FEATURES_BIN]
y = df[TARGET]

# 70/15/15 train/val/test - val used only for threshold selection, test never touched until final eval
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)}")

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ("bin", "passthrough", FEATURES_BIN),
])

models = {
    "logistic_regression": Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
    "gradient_boosting": Pipeline([("prep", preprocess), ("clf", GradientBoostingClassifier(random_state=42, n_estimators=200, max_depth=3, learning_rate=0.05))]),
}

results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    val_proba = pipe.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_proba)
    pr_auc = average_precision_score(y_val, val_proba)
    results[name] = {"pipe": pipe, "val_auc": auc, "val_pr_auc": pr_auc}
    print(f"{name}: val ROC-AUC={auc:.4f} val PR-AUC={pr_auc:.4f}")

best_name = max(results, key=lambda k: results[k]["val_pr_auc"])
best_pipe = results[best_name]["pipe"]
print(f"\nSelected model: {best_name}")

# ---- Cost-based threshold selection on validation set ----
# Assumed unit economics (typical mid-size Indian e-comm order, INR):
#   Cost of a MISSED return (false negative): reverse logistics + restock + platform penalty ~ INR 250
#   Cost of a FALSE ALARM (false positive): friction cost of flagging a good order
#     (e.g. forcing COD->prepaid conversion attempt, manual review, possible cart abandonment) ~ INR 40
FN_COST = 250
FP_COST = 40

val_proba = best_pipe.predict_proba(X_val)[:, 1]
thresholds = np.linspace(0.01, 0.99, 99)
costs = []
for t in thresholds:
    preds = (val_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
    cost = fp * FP_COST + fn * FN_COST
    costs.append(cost)
best_idx = int(np.argmin(costs))
best_threshold = float(thresholds[best_idx])
print(f"Cost-optimal threshold (val set): {best_threshold:.3f} (expected cost={costs[best_idx]:.0f} INR on {len(X_val)} val orders)")

# Compare to naive 0.5 threshold cost, for the write-up
preds_50 = (val_proba >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_val, preds_50).ravel()
cost_50 = fp * FP_COST + fn * FN_COST
print(f"Cost at naive 0.5 threshold: {cost_50:.0f} INR")

# ---- FINAL held-out test evaluation (only touched once, here) ----
test_proba = best_pipe.predict_proba(X_test)[:, 1]
test_preds = (test_proba >= best_threshold).astype(int)

test_auc = roc_auc_score(y_test, test_proba)
test_pr_auc = average_precision_score(y_test, test_proba)
test_precision = precision_score(y_test, test_preds)
test_recall = recall_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds)
tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()

# baseline: naive "flag everyone in top-risk category" heuristic, for comparison
baseline_preds = X_test["category"].isin(["apparel", "footwear"]).astype(int).values
base_precision = precision_score(y_test, baseline_preds)
base_recall = recall_score(y_test, baseline_preds)
base_f1 = f1_score(y_test, baseline_preds)

test_expected_cost = int(fp) * FP_COST + int(fn) * FN_COST
never_flag_cost = int(y_test.sum()) * FN_COST  # cost of flagging nobody at all
cost_reduction_pct = 100 * (1 - test_expected_cost / never_flag_cost)

metrics = {
    "model_selected": best_name,
    "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
    "operating_threshold": round(best_threshold, 3),
    "cost_assumptions_inr": {"false_negative_missed_return": FN_COST, "false_positive_false_alarm": FP_COST},
    "test_set_metrics": {
        "roc_auc": round(test_auc, 4),
        "pr_auc": round(test_pr_auc, 4),
        "precision": round(test_precision, 4),
        "recall": round(test_recall, 4),
        "f1": round(test_f1, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "expected_cost_inr": round(test_expected_cost, 0),
        "cost_reduction_vs_no_intervention_pct": round(cost_reduction_pct, 1),
    },
    "naive_category_baseline": {
        "description": "flag every apparel/footwear order (no ML)",
        "precision": round(base_precision, 4),
        "recall": round(base_recall, 4),
        "f1": round(base_f1, 4),
    },
    "base_return_rate_test_set": round(float(y_test.mean()), 4),
}

print(json.dumps(metrics, indent=2))

# PR curve data for the frontend chart
precisions, recalls, pr_thresholds = precision_recall_curve(y_test, test_proba)
pr_curve = [{"threshold": float(t), "precision": float(p), "recall": float(r)}
            for p, r, t in zip(precisions[:-1], recalls[:-1], pr_thresholds)]
pr_curve_sampled = pr_curve[::max(1, len(pr_curve)//60)]  # thin for frontend

with open("/home/claude/return-risk-manager/model/metrics.json", "w") as f:
    json.dump({"metrics": metrics, "pr_curve": pr_curve_sampled}, f, indent=2)

joblib.dump(best_pipe, "/home/claude/return-risk-manager/model/return_risk_model.joblib")
joblib.dump(best_threshold, "/home/claude/return-risk-manager/model/threshold.joblib")
print("\nSaved model, threshold, and metrics.")
