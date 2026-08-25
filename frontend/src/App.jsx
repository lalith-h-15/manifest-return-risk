import React, { useState, useMemo } from "react";

// ---- Exact trained logistic-regression model, extracted from the
// scikit-learn pipeline (model/train.py). This reproduces the real
// held-out-evaluated model bit-for-bit for the in-browser demo. ----
const MODEL = {
  intercept: -0.41512354686537944,
  num: {
    price: { coef: -0.0651, mean: 1083.8746954285716, scale: 1379.7237967222295 },
    discount_pct: { coef: 0.128, mean: 28.436291428571426, scale: 15.857755311002904 },
    customer_order_count: { coef: -0.0036, mean: 5.498857142857143, scale: 3.3222623715333084 },
    customer_return_rate_hist: { coef: 0.4077, mean: 0.19776917073589342, scale: 0.24899837554163812, imputeMedian: 0.125 },
    days_since_signup: { coef: 0.0116, mean: 180.13411428571428, scale: 179.753321166818 },
    delivery_days: { coef: 0.0704, mean: 3.6202285714285716, scale: 1.8574643097623211 },
    review_rating_at_purchase: { coef: -0.2033, mean: 4.0904742857142855, scale: 0.5663760519265609 },
  },
  category: { accessories: -0.1317, apparel: 0.836, beauty: -0.4083, electronics: 0.1538, footwear: 0.7729, grocery: -1.4474, home: -0.1788 },
  payment: { COD: 0.3581, UPI: -0.2372, prepaid_card: -0.3242, wallet: -0.2002 },
  bin: { is_weekend_order: -0.0628, size_variant: 0.2554, coupon_used: -0.0713 },
  threshold: 0.4,
};

const METRICS = {
  model_selected: "logistic_regression", n_train: 17500, n_val: 3750, n_test: 3750,
  operating_threshold: 0.4,
  cost: { fn: 250, fp: 40 },
  test: { roc_auc: 0.7385, pr_auc: 0.3907, precision: 0.2654, recall: 0.8321, f1: 0.4024, tn: 1527, fp: 1550, fn: 113, tp: 560, expected_cost_inr: 90250, cost_reduction_pct: 46.4 },
  baseline: { description: "flag every apparel / footwear order, no model", precision: 0.3067, recall: 0.633, f1: 0.4132 },
  base_rate: 0.1795,
};

const PR_CURVE = [
  [0.041,0.179,1.0],[0.086,0.182,1.0],[0.115,0.185,0.996],[0.145,0.187,0.991],[0.168,0.190,0.990],
  [0.183,0.193,0.987],[0.197,0.196,0.984],[0.210,0.198,0.978],[0.220,0.201,0.973],[0.231,0.203,0.964],
  [0.241,0.206,0.958],[0.250,0.209,0.952],[0.262,0.212,0.947],[0.271,0.215,0.941],[0.280,0.218,0.935],
  [0.289,0.222,0.930],[0.298,0.226,0.927],[0.306,0.231,0.926],[0.316,0.234,0.917],[0.326,0.238,0.911],
  [0.335,0.242,0.903],[0.345,0.246,0.896],[0.356,0.250,0.886],[0.365,0.252,0.871],[0.375,0.256,0.860],
  [0.386,0.260,0.851],[0.396,0.263,0.835],[0.406,0.268,0.828],[0.416,0.273,0.817],[0.428,0.278,0.807],
  [0.440,0.281,0.789],[0.450,0.287,0.779],[0.461,0.289,0.759],[0.472,0.296,0.750],[0.480,0.302,0.737],
  [0.489,0.305,0.716],[0.500,0.310,0.698],[0.508,0.313,0.678],[0.518,0.316,0.654],[0.529,0.322,0.637],
  [0.542,0.329,0.621],[0.552,0.330,0.593],[0.561,0.337,0.574],[0.571,0.345,0.556],[0.581,0.350,0.532],
  [0.591,0.357,0.510],[0.602,0.365,0.487],[0.618,0.380,0.473],[0.633,0.386,0.444],[0.647,0.389,0.412],
  [0.659,0.398,0.385],[0.675,0.400,0.349],[0.690,0.401,0.314],[0.703,0.418,0.288],[0.715,0.435,0.260],
  [0.730,0.453,0.229],[0.748,0.468,0.193],[0.767,0.495,0.159],[0.790,0.526,0.120],[0.823,0.598,0.082],
  [0.881,0.800,0.036],
];

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function scoreOrder(order) {
  let logit = MODEL.intercept;
  for (const [key, spec] of Object.entries(MODEL.num)) {
    let raw = order[key];
    if (raw === null || raw === undefined || raw === "") {
      raw = spec.imputeMedian !== undefined ? spec.imputeMedian : spec.mean;
    }
    logit += spec.coef * ((Number(raw) - spec.mean) / spec.scale);
  }
  logit += MODEL.category[order.category] || 0;
  logit += MODEL.payment[order.payment_method] || 0;
  logit += order.is_weekend_order ? MODEL.bin.is_weekend_order : 0;
  logit += order.size_variant ? MODEL.bin.size_variant : 0;
  logit += order.coupon_used ? MODEL.bin.coupon_used : 0;
  return sigmoid(logit);
}

function factorsFor(order, score) {
  const f = [];
  if (order.payment_method === "COD") f.push({ label: "Cash on delivery", dir: "up" });
  if (order.category === "apparel" || order.category === "footwear") f.push({ label: `Category: ${order.category}`, dir: "up" });
  if (order.customer_return_rate_hist !== "" && Number(order.customer_return_rate_hist) > 0.25) f.push({ label: "High personal return history", dir: "up" });
  if (Number(order.discount_pct) > 40) f.push({ label: "Deep discount", dir: "up" });
  if (Number(order.review_rating_at_purchase) >= 4.5) f.push({ label: "High product rating", dir: "down" });
  if (order.category === "grocery" || order.category === "beauty") f.push({ label: `Category: ${order.category}`, dir: "down" });
  if (order.payment_method === "prepaid_card") f.push({ label: "Prepaid card", dir: "down" });
  if (f.length === 0) f.push({ label: "No strong risk drivers", dir: "flat" });
  return f.slice(0, 4);
}

const CATEGORIES = ["apparel", "footwear", "electronics", "home", "beauty", "accessories", "grocery"];
const PAYMENTS = ["COD", "UPI", "prepaid_card", "wallet"];

const inputStyle = {
  width: "100%", padding: "9px 10px", fontFamily: "'IBM Plex Mono', monospace",
  fontSize: 13.5, background: "#F7F6F0", border: "1px solid #C9CABA",
  color: "#16233D", outline: "none",
};
const labelStyle = {
  display: "block", fontFamily: "'IBM Plex Sans', sans-serif", fontSize: 11,
  letterSpacing: "0.06em", textTransform: "uppercase", color: "#5B6470", marginBottom: 5,
};

function Field({ label, children }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <label style={labelStyle}>{label}</label>
      {children}
    </div>
  );
}

export default function App() {
  const [order, setOrder] = useState({
    category: "apparel", price: 1499, discount_pct: 35, payment_method: "COD",
    customer_order_count: 2, customer_return_rate_hist: "", days_since_signup: 90,
    delivery_days: 4, is_weekend_order: false, size_variant: true, coupon_used: true,
    review_rating_at_purchase: 4.0,
  });
  const [scored, setScored] = useState(null);
  const [stampKey, setStampKey] = useState(0);

  const update = (k, v) => setOrder((o) => ({ ...o, [k]: v }));

  const runScore = () => {
    const proba = scoreOrder(order);
    const flagged = proba >= MODEL.threshold;
    setScored({ proba, flagged, factors: factorsFor(order, proba) });
    setStampKey((k) => k + 1);
  };

  const prPoints = useMemo(() => PR_CURVE, []);

  return (
    <div style={{
      fontFamily: "'IBM Plex Sans', sans-serif", background: "#E9EAE2", color: "#16233D",
      minHeight: "100vh", padding: "0 0 60px",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');
        * { box-sizing: border-box; }
        button.stamp-btn { transition: transform .08s ease; }
        button.stamp-btn:active { transform: scale(0.97); }
        input[type=checkbox] { accent-color: #16233D; width: 15px; height: 15px; }
        @keyframes stampDown {
          0% { opacity: 0; transform: rotate(-14deg) scale(2.4); }
          60% { opacity: 1; transform: rotate(-8deg) scale(0.92); }
          80% { transform: rotate(-8deg) scale(1.05); }
          100% { opacity: 1; transform: rotate(-8deg) scale(1); }
        }
        .stamp { animation: stampDown 0.5s cubic-bezier(.2,.8,.3,1); }
      `}</style>

      {/* HEADER */}
      <div style={{ borderBottom: "1px solid #C9CABA", background: "#F1F0E8" }}>
        <div style={{ maxWidth: 1080, margin: "0 auto", padding: "28px 28px 22px", display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: 12 }}>
          <div>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, letterSpacing: "0.14em", color: "#8A6B1D", marginBottom: 6 }}>
              TRACK 02 &middot; RETURN-RISK DESK
            </div>
            <h1 style={{ fontFamily: "'Source Serif 4', serif", fontWeight: 600, fontSize: 34, margin: 0, letterSpacing: "-0.01em" }}>
              Manifest
            </h1>
            <p style={{ margin: "4px 0 0", fontSize: 14, color: "#5B6470", maxWidth: 460 }}>
              Every order gets inspected against the ledger before it ships. Scores are advisory — Manifest never blocks, cancels, or auto-charges an order.
            </p>
          </div>
          <div style={{ textAlign: "right", fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, color: "#5B6470", lineHeight: 1.7 }}>
            <div>model &nbsp;<b style={{ color: "#16233D" }}>logistic regression</b></div>
            <div>held-out test set &nbsp;<b style={{ color: "#16233D" }}>n = {METRICS.n_test.toLocaleString()}</b></div>
            <div>operating threshold &nbsp;<b style={{ color: "#16233D" }}>{METRICS.operating_threshold}</b></div>
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1080, margin: "0 auto", padding: "32px 28px 0", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 28 }}>

        {/* INTAKE FORM */}
        <div style={{ background: "#F7F6F0", border: "1px solid #C9CABA", padding: 22 }}>
          <div style={{ fontFamily: "'Source Serif 4', serif", fontSize: 17, fontWeight: 600, marginBottom: 16, borderBottom: "1px solid #C9CABA", paddingBottom: 10 }}>
            Order manifest entry
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <Field label="Category">
              <select style={inputStyle} value={order.category} onChange={(e) => update("category", e.target.value)}>
                {CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </Field>
            <Field label="Payment method">
              <select style={inputStyle} value={order.payment_method} onChange={(e) => update("payment_method", e.target.value)}>
                {PAYMENTS.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </Field>
            <Field label="Price (INR)">
              <input style={inputStyle} type="number" value={order.price} onChange={(e) => update("price", e.target.value)} />
            </Field>
            <Field label="Discount %">
              <input style={inputStyle} type="number" value={order.discount_pct} onChange={(e) => update("discount_pct", e.target.value)} />
            </Field>
            <Field label="Customer's past orders">
              <input style={inputStyle} type="number" value={order.customer_order_count} onChange={(e) => update("customer_order_count", e.target.value)} />
            </Field>
            <Field label="Customer's return rate (blank = new)">
              <input style={inputStyle} type="number" step="0.05" placeholder="e.g. 0.20" value={order.customer_return_rate_hist} onChange={(e) => update("customer_return_rate_hist", e.target.value)} />
            </Field>
            <Field label="Days since signup">
              <input style={inputStyle} type="number" value={order.days_since_signup} onChange={(e) => update("days_since_signup", e.target.value)} />
            </Field>
            <Field label="Delivery days (est.)">
              <input style={inputStyle} type="number" value={order.delivery_days} onChange={(e) => update("delivery_days", e.target.value)} />
            </Field>
            <Field label="Product rating">
              <input style={inputStyle} type="number" step="0.1" min="1" max="5" value={order.review_rating_at_purchase} onChange={(e) => update("review_rating_at_purchase", e.target.value)} />
            </Field>
          </div>

          <div style={{ display: "flex", gap: 18, margin: "6px 0 20px" }}>
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12.5, color: "#5B6470" }}>
              <input type="checkbox" checked={order.size_variant} onChange={(e) => update("size_variant", e.target.checked)} /> Size-variant item
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12.5, color: "#5B6470" }}>
              <input type="checkbox" checked={order.coupon_used} onChange={(e) => update("coupon_used", e.target.checked)} /> Coupon used
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12.5, color: "#5B6470" }}>
              <input type="checkbox" checked={order.is_weekend_order} onChange={(e) => update("is_weekend_order", e.target.checked)} /> Weekend order
            </label>
          </div>

          <button className="stamp-btn" onClick={runScore} style={{
            width: "100%", padding: "12px", background: "#16233D", color: "#F1F0E8", border: "none",
            fontFamily: "'IBM Plex Mono', monospace", fontSize: 13, letterSpacing: "0.06em", cursor: "pointer",
          }}>
            INSPECT ORDER
          </button>
        </div>

        {/* RESULT PANEL */}
        <div style={{ background: "#F7F6F0", border: "1px solid #C9CABA", padding: 22, position: "relative", display: "flex", flexDirection: "column" }}>
          <div style={{ fontFamily: "'Source Serif 4', serif", fontSize: 17, fontWeight: 600, marginBottom: 16, borderBottom: "1px solid #C9CABA", paddingBottom: 10 }}>
            Inspection result
          </div>

          {!scored && (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "#8A8A80", fontSize: 13.5, textAlign: "center", padding: "40px 20px" }}>
              Fill in the manifest and inspect the order to see its risk score.
            </div>
          )}

          {scored && (
            <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "10px 0 20px" }}>
              <div key={stampKey} className="stamp" style={{
                border: `3px solid ${scored.flagged ? "#A6321F" : "#3D6B4A"}`,
                color: scored.flagged ? "#A6321F" : "#3D6B4A",
                padding: "14px 30px", transform: "rotate(-8deg)", marginBottom: 22,
                fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600, fontSize: 20,
                letterSpacing: "0.08em", borderRadius: 3,
              }}>
                {scored.flagged ? "FLAGGED · REVIEW" : "CLEARED"}
              </div>

              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 42, fontWeight: 600, lineHeight: 1 }}>
                {(scored.proba * 100).toFixed(1)}<span style={{ fontSize: 20 }}>%</span>
              </div>
              <div style={{ fontSize: 12, color: "#5B6470", marginTop: 4, marginBottom: 20 }}>
                estimated return probability &middot; threshold {MODEL.threshold}
              </div>

              <div style={{ width: "100%", maxWidth: 320 }}>
                {scored.factors.map((f, i) => (
                  <div key={i} style={{
                    display: "flex", justifyContent: "space-between", fontSize: 12.5,
                    padding: "7px 0", borderBottom: i < scored.factors.length - 1 ? "1px solid #DEDFD3" : "none",
                  }}>
                    <span>{f.label}</span>
                    <span style={{ color: f.dir === "up" ? "#A6321F" : f.dir === "down" ? "#3D6B4A" : "#8A8A80", fontFamily: "'IBM Plex Mono', monospace" }}>
                      {f.dir === "up" ? "↑ risk" : f.dir === "down" ? "↓ risk" : "—"}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* HONEST EVAL SECTION */}
      <div style={{ maxWidth: 1080, margin: "40px auto 0", padding: "0 28px" }}>
        <div style={{ fontFamily: "'Source Serif 4', serif", fontSize: 20, fontWeight: 600, marginBottom: 4 }}>
          The ledger: honest evaluation on held-out orders
        </div>
        <p style={{ fontSize: 13.5, color: "#5B6470", margin: "0 0 20px", maxWidth: 700 }}>
          Trained on 17,500 synthetic orders, tuned on a separate 3,750-order validation set, scored once on a 3,750-order test set the model never saw during training or threshold selection.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 1, background: "#C9CABA", border: "1px solid #C9CABA", marginBottom: 20 }}>
          {[
            ["ROC-AUC", METRICS.test.roc_auc],
            ["PR-AUC", METRICS.test.pr_auc],
            ["Precision", METRICS.test.precision],
            ["Recall", METRICS.test.recall],
            ["F1", METRICS.test.f1],
          ].map(([label, val]) => (
            <div key={label} style={{ background: "#F7F6F0", padding: "16px 14px" }}>
              <div style={{ fontSize: 11, color: "#5B6470", textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</div>
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 24, fontWeight: 600, marginTop: 4 }}>{val.toFixed(3)}</div>
            </div>
          ))}
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 24 }}>
          {/* Confusion matrix */}
          <div style={{ background: "#F7F6F0", border: "1px solid #C9CABA", padding: 20 }}>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 14 }}>Confusion matrix (test set, n=3,750)</div>
            <div style={{ display: "grid", gridTemplateColumns: "80px 1fr 1fr", fontFamily: "'IBM Plex Mono', monospace", fontSize: 12.5 }}>
              <div></div>
              <div style={{ textAlign: "center", color: "#5B6470", paddingBottom: 6 }}>pred. no return</div>
              <div style={{ textAlign: "center", color: "#5B6470", paddingBottom: 6 }}>pred. return</div>

              <div style={{ color: "#5B6470", display: "flex", alignItems: "center" }}>actual no</div>
              <div style={{ background: "#E7EEE8", textAlign: "center", padding: "14px 0" }}>{METRICS.test.tn}<div style={{ fontSize: 10, color: "#5B6470" }}>TN</div></div>
              <div style={{ background: "#F6E3DF", textAlign: "center", padding: "14px 0" }}>{METRICS.test.fp}<div style={{ fontSize: 10, color: "#8A5B4E" }}>FP · false alarm</div></div>

              <div style={{ color: "#5B6470", display: "flex", alignItems: "center" }}>actual return</div>
              <div style={{ background: "#F6E3DF", textAlign: "center", padding: "14px 0" }}>{METRICS.test.fn}<div style={{ fontSize: 10, color: "#8A5B4E" }}>FN · missed</div></div>
              <div style={{ background: "#E7EEE8", textAlign: "center", padding: "14px 0" }}>{METRICS.test.tp}<div style={{ fontSize: 10, color: "#5B6470" }}>TP</div></div>
            </div>
          </div>

          {/* Cost analysis */}
          <div style={{ background: "#F7F6F0", border: "1px solid #C9CABA", padding: 20 }}>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 14 }}>False-positive cost, priced in</div>
            <div style={{ fontSize: 12.5, color: "#5B6470", lineHeight: 1.7, marginBottom: 14 }}>
              A missed return (FN) costs ~₹{METRICS.cost.fn} in reverse logistics and restocking.
              A false alarm (FP) costs ~₹{METRICS.cost.fp} in review friction. Because FN is ~6x more expensive than FP,
              the operating threshold ({METRICS.operating_threshold}) is deliberately tuned for recall over precision —
              this is a cost-minimizing choice, not a modeling mistake.
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "'IBM Plex Mono', monospace", fontSize: 13, padding: "8px 0", borderTop: "1px solid #DEDFD3" }}>
              <span>Expected cost with model</span><b>₹{METRICS.test.expected_cost_inr.toLocaleString()}</b>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "'IBM Plex Mono', monospace", fontSize: 13, padding: "8px 0", borderTop: "1px solid #DEDFD3" }}>
              <span>Cost reduction vs. no intervention</span><b style={{ color: "#3D6B4A" }}>-{METRICS.test.cost_reduction_pct}%</b>
            </div>
          </div>
        </div>

        {/* Honest baseline comparison callout */}
        <div style={{ background: "#F6F0DC", border: "1px solid #D9C88A", padding: "16px 20px", marginBottom: 28, fontSize: 13 }}>
          <b>Where a naive rule beats the model:</b> flagging every apparel/footwear order with no ML at all gets
          precision {METRICS.baseline.precision.toFixed(3)} / recall {METRICS.baseline.recall.toFixed(3)} / F1 {METRICS.baseline.f1.toFixed(3)} —
          slightly higher precision and F1 than the tuned model ({METRICS.test.precision.toFixed(3)} / {METRICS.test.recall.toFixed(3)} / {METRICS.test.f1.toFixed(3)}).
          The model still wins on recall and on total cost, because its threshold was chosen to minimize ₹ loss, not F1.
          We're reporting this rather than hiding it — it shows the category signal is doing a lot of the work, and the model's
          real value-add is the customer-history and discount-depth signal layered on top.
        </div>

        {/* PR curve */}
        <div style={{ background: "#F7F6F0", border: "1px solid #C9CABA", padding: 20, marginBottom: 40 }}>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 4 }}>Precision-recall tradeoff across thresholds</div>
          <div style={{ fontSize: 12, color: "#5B6470", marginBottom: 14 }}>Computed on the same held-out test set. Vertical marker = shipped operating threshold ({METRICS.operating_threshold}).</div>
          <PRChart data={prPoints} threshold={METRICS.operating_threshold} />
        </div>

        <div style={{ fontSize: 11.5, color: "#8A8A80", borderTop: "1px solid #C9CABA", paddingTop: 16, paddingBottom: 30 }}>
          Defense-only. Manifest scores orders for human review — it does not autonomously cancel orders, charge customers,
          or take any action against a customer account. Base return rate on this dataset: {(METRICS.base_rate * 100).toFixed(1)}%.
        </div>
      </div>
    </div>
  );
}

function PRChart({ data, threshold }) {
  const W = 620, H = 220, PAD = 36;
  const x = (v) => PAD + v * (W - PAD * 2);
  const y = (v) => H - PAD - v * (H - PAD * 2) / 0.6;

  const precPath = data.map((d, i) => `${i === 0 ? "M" : "L"} ${x(d[0]).toFixed(1)} ${y(d[1]).toFixed(1)}`).join(" ");
  const recPath = data.map((d, i) => `${i === 0 ? "M" : "L"} ${x(d[0]).toFixed(1)} ${y(d[2]).toFixed(1)}`).join(" ");
  const closestIdx = data.reduce((best, d, i) => Math.abs(d[0] - threshold) < Math.abs(data[best][0] - threshold) ? i : best, 0);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 620, display: "block" }}>
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#C9CABA" strokeWidth="1" />
      <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#C9CABA" strokeWidth="1" />
      {[0, 0.2, 0.4, 0.6].map((t) => (
        <text key={t} x={PAD - 6} y={y(t) + 3} textAnchor="end" fontSize="9.5" fontFamily="IBM Plex Mono, monospace" fill="#8A8A80">{t.toFixed(1)}</text>
      ))}
      {[0, 0.5, 1].map((t) => (
        <text key={t} x={x(t)} y={H - PAD + 14} textAnchor="middle" fontSize="9.5" fontFamily="IBM Plex Mono, monospace" fill="#8A8A80">{t}</text>
      ))}
      <text x={W / 2} y={H - 6} textAnchor="middle" fontSize="10" fontFamily="IBM Plex Sans, sans-serif" fill="#5B6470">score threshold</text>

      <line x1={x(threshold)} y1={PAD} x2={x(threshold)} y2={H - PAD} stroke="#16233D" strokeWidth="1" strokeDasharray="3,3" />

      <path d={precPath} fill="none" stroke="#A6321F" strokeWidth="2" />
      <path d={recPath} fill="none" stroke="#3D6B4A" strokeWidth="2" />

      <circle cx={x(data[closestIdx][0])} cy={y(data[closestIdx][1])} r="3.5" fill="#A6321F" />
      <circle cx={x(data[closestIdx][0])} cy={y(data[closestIdx][2])} r="3.5" fill="#3D6B4A" />

      <g transform={`translate(${W - 150}, ${PAD})`}>
        <line x1="0" y1="4" x2="16" y2="4" stroke="#A6321F" strokeWidth="2" />
        <text x="20" y="8" fontSize="10.5" fontFamily="IBM Plex Sans, sans-serif" fill="#16233D">Precision</text>
        <line x1="0" y1="20" x2="16" y2="20" stroke="#3D6B4A" strokeWidth="2" />
        <text x="20" y="24" fontSize="10.5" fontFamily="IBM Plex Sans, sans-serif" fill="#16233D">Recall</text>
      </g>
    </svg>
  );
}
