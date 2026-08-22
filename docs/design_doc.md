# Return-Risk Scorer — Project Design Doc
**Track:** AI Risk Manager (Razorpay Buildathon)
**Builder:** Solo
**Submission deadline:** Sep 5, 2026

---

## 1. Problem Statement

Merchants lose margin to returns silently — no single "loss event" like fraud, just steady erosion. Most merchants treat every order the same at fulfillment time, with no signal on which orders are likely to be returned.

**We build:** a model that scores each order with a probability of return at the time of purchase (or shortly after), so a merchant can intervene — flag for manual review, adjust packaging/QC, hold off on express shipping, or trigger a confirmation nudge — before the cost is locked in.

---

## 2. Project Goals (in priority order)

1. **G1 — Working scorer.** Given order features, output a calibrated return-risk probability (0–1) and a risk tier (Low/Medium/High).
2. **G2 — Honest, measured accuracy.** Precision, recall, F1, and AUC reported on a held-out test set — not the training set. Include false-positive cost framing (a wrongly-flagged order has a real merchant cost — friction, delay).
3. **G3 — Explainability.** For any flagged order, show *why* — top contributing features (e.g. "high-value item + first-time buyer + COD payment").
4. **G4 — Audit trail.** Every scored order logs: input features, score, tier, timestamp, model version. Demonstrates the "bounded and gated" spirit the judges want even outside the commerce track.
5. **G5 — Demoable end-to-end.** A simple UI where a judge can input/select an order and see score + explanation + audit entry live, including an interactive cost-tradeoff view (see Section 7a).

**Explicit non-goals** (to prevent scope creep):
- No LLM agent orchestration — this is a scoring system, not an agentic workflow.
- No real Razorpay transaction data — synthetic only.
- No deployment/hosting — local demo is sufficient.
- No multi-model ensembling or deep learning — a single well-validated model beats a complex stack you can't fully explain on demo day.

---

## 3. Why This Problem (Judging Fit)

| Track requirement | How this project meets it |
|---|---|
| "Detector, verifier, or auto-responder for one class of loss" | Detector — return risk |
| "Measured precision and recall on a held-out test set" | Core deliverable, built early (Day 6-7), not bolted on at the end |
| "Honest metrics including false-positive cost" | Explicit cost-of-false-positive discussion in the write-up and demo |
| "Strictly defense-only" | Pure scoring/flagging — no autonomous action taken |

---

## 4. Data Design (Synthetic)

Since there's no real data, the dataset *is* part of the deliverable — it needs to be believable and have real signal, not random noise, or the model will look meaningless.

**Approach:** generate synthetic orders with features that have a *plausible causal link* to returns, then generate the return outcome using a rule-based/probabilistic function of those features (with noise). This way the "ground truth" is defined by you and the model's job is to recover the pattern — which is exactly what you're measuring in the metrics.

**Explicit label-generation formula (locked now, not left vague):**

```
logit = (
    w1 * is_apparel_or_footwear
  + w2 * standardize(order_value)
  + w3 * is_cod
  + w4 * standardize(discount_pct)
  + w5 * standardize(past_return_rate)
  + w6 * is_first_time_customer
  + w7 * standardize(qty_or_size_variants_ordered)
  + w8 * standardize(pincode_return_rate)
  + w9 * standardize(delivery_days)
  + w10 * is_cod * standardize(order_value)   # interaction: COD removes commitment, worse specifically at high value
  + noise ~ Normal(0, sigma)
)
p_return = sigmoid(logit)
label = Bernoulli(p_return)
```

- Weights (w1–w10) chosen by hand, documented in the generator script's docstring — not tuned against the model's ability to learn them.
- **w10 is a deliberate interaction term** (COD × order value), not just an additive one. A purely additive label function gives the gradient-boosted model nothing to find beyond what logistic regression already captures — the interaction term means the baseline-vs-primary-model comparison in the demo has real teeth: "XGBoost beats logistic regression by X points of AUC because it captures the COD×value interaction" is a concrete, verifiable claim, not just "XGBoost is marginally better."
- `sigma` (noise) is a deliberate knob: tuned so the **best model's held-out test AUC lands in the 0.75–0.85 range**, not higher. A near-perfect AUC (0.95+) on synthetic data is a red flag for label leakage, not a win — this is called out explicitly in the README.
- **Correction protocol if AUC misses the 0.75–0.85 band:** adjust sigma (increase to lower AUC, decrease to raise it) and regenerate the dataset. Capped at **3 regeneration cycles**, all completed before Day 4 ends. If still out of band after 3 cycles, widen the target band to 0.70–0.88 and document the deviation honestly in the README rather than burning further days chasing an exact number.
- **Target base rate: ~15–20% positive class** (real-world return rates), not 50/50. This makes precision/recall meaningful and means raw accuracy is never reported as a headline metric (it would be trivially high and misleading on an imbalanced set).

**Candidate features:**
| Feature | Type | Return-risk logic |
|---|---|---|
| Product category | categorical | apparel/footwear (size issues) > electronics > groceries |
| Order value | numeric | high value → more likely to be reconsidered/returned |
| Payment method | categorical | COD > prepaid (lower commitment at purchase) |
| Discount % applied | numeric | heavy discount → impulse buy → higher return risk |
| Customer order history | numeric (past orders, past returns) | past returner → higher risk |
| First-time vs repeat customer | boolean | first-time → higher uncertainty |
| Size/quantity ordered | numeric | multiple sizes/qty of same item → likely a "try before decide" order |
| Delivery pin code return-rate (synthetic regional signal) | numeric | some regions have logistics-driven return issues |
| Time to delivery (days) | numeric | longer wait → higher cancel/return odds |

Target row count: **2,500–3,000 synthetic orders**, committing to the top of the original range. This matters because the validation slice used for threshold tuning is small: 70% train → 80/20 fit/validation leaves roughly 420–480 validation rows at a 15–20% base rate, i.e. ~65–95 positive examples to tune three tier thresholds on. At the low end of the original 1,000-row target that would have been only 20–85 positives — thin enough for noisy, non-principled thresholds. Even at 2,500–3,000 rows the validation-slice sample size is modest; this is stated as an explicit limitation in the README rather than left for a judge to discover.

**Split:** 70% train / 30% held-out test, stratified by return label. The test split is frozen immediately after generation and touched exactly once — for final metric reporting. No feature selection, threshold tuning, or early stopping is ever decided by looking at test performance (see Section 5 for where tuning actually happens).

---

## 5. Model Design

- **Baseline:** Logistic Regression (fast, interpretable coefficients — good for explainability and a sanity-check baseline).
- **Primary model:** Gradient Boosted Trees (XGBoost or LightGBM) — better accuracy on tabular data, still explainable via feature importance / SHAP values.
- **Explainability layer:** SHAP is the primary choice. **Fallback (if SHAP proves too slow to wire up under deadline):** per-prediction contribution from the logistic regression baseline — i.e. `weight_i * feature_value_i` for each feature, which is a genuine per-order explanation, not a downgrade to global importance. This fallback is never silently substituted and called "the same thing" as SHAP in the pitch — if used, it's named explicitly as the logistic-coefficient method.
- **Risk tiers:** thresholds on the probability output (e.g. Low <0.3, Medium 0.3–0.65, High >0.65) are tuned on a **validation slice carved out of the training set** (e.g. train further split 80/20 into fit/validation), never on the held-out test set. Test-set metrics are computed once, after thresholds are locked, and never fed back into tuning.
- **Model version tracking:** `model_version` in the audit log is a hash of the trained model artifact (e.g. SHA256 of the serialized model file) plus a training timestamp — not a hardcoded string like `"v1"` — so the audit trail is actually verifiable against a specific trained artifact.

---

## 6. Metrics & Evaluation Plan

Report on the held-out test set:
- **Precision, Recall, F1** at the chosen threshold
- **AUC-ROC** (threshold-independent view)
- **Confusion matrix** (raw counts — judges can sanity-check this fastest)
- **False-positive cost framing:** e.g. "X% of flagged orders were not actually returned — at an assumed review cost of ₹Y per flagged order, that's ₹Z in unnecessary friction, versus ₹W in returns prevented." This turns raw metrics into a business number, which is what "honest metrics including false-positive cost" is really asking for.
- **Exception/uncertain cases:** orders where the model's confidence is near the decision boundary — explicitly called out, not hidden.
- **Reproducibility:** the raw output of the metrics script (confusion matrix, classification report, AUC) is committed to the repo as a log file — not just numbers pasted into a slide. Anyone can re-run the script against the frozen test set and get the same file.
- **Threshold sweep:** metrics (precision, recall, FP-rate, FN-rate) are computed across the full range of thresholds on the frozen test set and saved as a table/array — this same sweep powers the interactive cost-tradeoff slider in the demo UI (Section 7a), so the "live" recomputation is a fast lookup, not a retrain.

---

## 7. Explainability & Audit Trail

- Each scored order produces a record: `{order_id, features, score, tier, top_3_contributing_factors, model_version, timestamp}`.
- **`top_3_contributing_factors` are signed, not just largest-magnitude.** Only features that *increased* the risk score are eligible to appear as "why this was flagged" — a feature that strongly lowered risk (e.g. long repeat-customer history) must never surface in that list just because its magnitude was large. This is covered by a specific unit test (feed a synthetic order with one large negative contributor and assert it never appears in the top-3-risk-increasing list), since getting this wrong would look like a bug live in front of judges.
- Stored as a simple local log (JSON lines or SQLite) — enough to demonstrate the "bounded and gated" / audit-trail expectation without building real infrastructure.
- Demo UI surfaces this record when a judge clicks into any scored order.

---

## 7a. Interactive Cost Trade-off (Demo Differentiator)

The Streamlit dashboard includes a slider-driven view, not just a static false-positive-cost writeup:

- Two sliders: **C_FP** (cost of a false alarm — reviewing/flagging an order that wasn't actually going to be returned) and **C_FN** (cost of an unmitigated return — letting a high-risk order ship without intervention).
- As the judge moves either slider, the dashboard recomputes the **optimal decision threshold** (minimizing expected cost = C_FP × false-positive-rate + C_FN × false-negative-rate, evaluated across thresholds on the frozen test set) and redraws the confusion matrix and expected-cost number live.
- This makes the abstract point "thresholds should reflect merchant economics, not be arbitrary" into something a judge can *touch* — a low-margin merchant (high C_FP sensitivity, can't afford review friction) wants a different threshold than a high-value/high-return-cost merchant (high C_FN sensitivity). Recomputation is a lookup over a precomputed threshold sweep — no retraining, so it's instant during the demo.

**LLM guardrails (if the optional LLM explanation step is used):**
- The LLM call (turning SHAP/coefficient values into a natural-language sentence) is never in the scoring critical path — the score, tier, and structured top-3-factors are computed and displayed independent of whether the LLM call succeeds.
- LLM output is validated against a strict JSON/pydantic schema before being shown; a malformed or failed response never reaches the UI.
- **Deterministic local fallback** if the LLM call fails, times out, or fails validation: a template sentence built directly from the structured factors, e.g. `"High risk driven by {feature_1} and {feature_2}."` — no natural-language step is a single point of failure for a live demo.

---

## 8. Tech Stack (Zero Budget)

| Layer | Tool | Cost |
|---|---|---|
| Data generation | Python + pandas + numpy | Free |
| Modeling | scikit-learn + XGBoost/LightGBM | Free |
| Explainability | SHAP | Free |
| Audit log | SQLite or JSON lines | Free |
| Demo UI | Streamlit | Free |
| Explanation text (optional polish) | Free-tier LLM API, schema-validated with deterministic fallback (see Section 7a) | Free tier |
| Submission | GitHub repo + README + demo video | Free |

---

## 9. Two-Week Build Plan (Aug 22 → Sep 5)

| Days | Milestone |
|---|---|
| 1–2 | Finalize feature list + return-generation logic; write synthetic data generator |
| 3–4 | Generate dataset (1-3k rows), EDA, sanity-check class balance and feature-signal strength |
| 5–6 | Train baseline (logistic regression) + primary model (XGBoost); tune threshold |
| 7 | Compute held-out metrics (precision/recall/F1/AUC/confusion matrix) + false-positive cost writeup |
| 8–9 | Add SHAP explainability; build audit log |
| 10–11 | Build Streamlit demo UI (score an order, see explanation, see audit entry) |
| 12 | Polish: README, architecture diagram, write-up connecting to "the bar" |
| 13 | Buffer — bug fixes, re-run metrics, record demo video |
| 14 (Sep 5) | Submit |

**Commit discipline:** the git log is itself evidence of the rigor this doc argues for — a README can claim "metrics before polish" after the fact; a commit history that shows the label formula committed before model code, and the metrics script committed before Streamlit UI work, can't be faked without it being visible. Practically: one focused commit per milestone above (not squashed at the end), with messages that name the milestone (e.g. `"lock label-generation formula + weights"`, `"held-out metrics: precision/recall/AUC/confusion matrix"`, `"streamlit demo UI"`). No commit that mixes a metrics change with a UI change — keep the boundary in the log, not just in the plan. **Tag checkpoints** at each major milestone (e.g. `git tag v0.1-synthetic-data`, `v0.2-baseline-model`, `v0.3-heldout-metrics`, `v0.4-explainability`, `v1.0-demo-ready`) and reference the tag log directly in the README as verifiable proof of sequence — not just a claim, a diffable one.

---

## 10. README Sections to Write (Beyond the Demo)

**"How the ground truth was constructed"** — the label-generation formula (Section 4) written out in the README with reasoning, not just "features with plausible causal links." Shows explicit reasoning about ground-truth construction and its limits — a data-maturity signal beyond what the hackathon strictly requires.

**"What I'd do differently with real data"** (one paragraph) — naming specific things that break in production:
- Concept drift as return policies change over time
- Feature leakage from signals only available *after* purchase (e.g. actual delivery experience) that wouldn't be available at scoring time in a real deployment
- Cold-start problem for new SKUs/regions with no history

**"How I'd validate this against real data before trusting it"** (one short section) — a synthetic-data model's real test starts after the hackathon, not before. Covers, at a sketch level:
- A shadow-mode deployment period: score real orders without acting on them, compare predicted risk against actual return outcomes once known
- A drift-detection metric to monitor post-launch (e.g. tracking the model's predicted positive rate vs. the actual observed return rate over rolling windows, flagging when they diverge)
- What would trigger a retrain (sustained metric drift, not a fixed calendar schedule)

---

## 11. Demo Script (for judges)

1. Show the problem in one line: "Merchants lose margin to returns silently — we score risk before dispatch."
2. Show the dataset + generation logic briefly (proves it's not random).
3. Live: pick/enter an order → show score, tier, top contributing factors.
4. Show the metrics dashboard: precision/recall/AUC/confusion matrix on held-out data.
5. Show the false-positive cost framing — this is the line that separates you from teams with only a demo, no numbers.
6. Show one audit log entry — proves traceability.
7. Close with limitations honestly stated (synthetic data, single loss-class, thresholds not merchant-calibrated yet) — judges respect honesty about scope over false confidence.

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Synthetic data too clean/unrealistic → trivial accuracy | Noise term (sigma) explicitly tuned for 0.75–0.85 test AUC, not maximized; documented as a deliberate choice |
| AUC misses the 0.75–0.85 band after generation | Adjust sigma and regenerate, capped at 3 cycles before Day 4 ends; if still out of band, widen target to 0.70–0.88 and disclose in README |
| Validation slice too small for stable threshold tuning | Row count committed to top of range (2,500–3,000); residual small-sample risk stated explicitly as a README limitation |
| Test-set leakage via threshold tuning | Thresholds tuned on a validation slice from train only; test set touched exactly once, after thresholds are locked |
| Running out of time before metrics are done | Metrics are scheduled Day 7, deliberately before UI polish — never let polish precede honesty |
| Solo scope creep (adding agent features, extra tracks) | Non-goals section above — revisit if tempted to add scope |
| SHAP too slow/complex to wire up in time | Named fallback: per-prediction logistic-coefficient contribution (genuinely per-order, not global importance) — disclosed as the fallback method if used, not disguised as SHAP |
| Class imbalance makes accuracy misleading | Base rate fixed at ~15–20% positive class by design; accuracy never reported as a headline metric |