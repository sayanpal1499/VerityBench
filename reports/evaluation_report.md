# VerityBench — Adversarial Reliability Evaluation Report

**Generated from Saved Results (`results/metrics.json`):** `2026-10-13T10:45:41.565406+00:00`
**Fixed SUT Model:** `groq/openai/gpt-oss-20b` | **Fixed Judge Model:** `gemini/gemini-3.5-flash-lite`
**Dataset Hash:** `4458b2c305b31859` | **Gate Forced-Abstention Fallback Confidence:** `GATE_FALLBACK_CONFIDENCE = 0.50`

> [!WARNING]
> **UNMET THREE-CLUSTER REQUIREMENT FLAG:** Observed `2` distinct failure clusters (`< 3` required by R8). Per Ground-Truth Governance Rule §8.2, clusters were never fabricated or split artificially.

---

## 1. Executive Summary & Verification Gate Status

- **Dataset Verification Status:** 50 base cases (`status: "verified"`), 100 adversarial variants (`status: "verified"` by sayanpal1499), 45 audit claims (`status: "verified"` by sayanpal1499), and 40 regression cases (`status: "inherited_from_base_case"` inherited directly from `BASE-001`–`BASE-040` in `data/base_cases.jsonl`).
- **LLM-Evaluated Accuracy (`grounded_v1`, `N=144` Eligible Cases excluding `6` pre-API `empty_malformed` rejections):** `112/144 (77.8%)`, Base-Case Cluster Bootstrap 95% CI: `[71.1%, 84.7%]`.
- **Nominal Suite Accuracy (`grounded_v1`, All `150` Records including `6` deterministic pre-API `InputValidationError` rejections):** `118/150 (78.7%)`, Base-Case Cluster Bootstrap 95% CI: `[72.7%, 85.0%]`.
- **Output-Level Hallucination Rate (`grounded_v1`):** `5/144` (`3.5%`).
- **Claim-Level Hallucination & Deferral Breakdown (`grounded_v1`):** Bad claims `5/98` (`5.1%`); Confirmed supported claims: `93`; Layer-2 deterministically decided: `36`; Layer-2 deferred (`needs_judge`): `58` (unresolved `needs_judge`: `0`).
- **Ablation vs. `naive_v0`:** Accuracy delta `+50.0 pp`; Output hallucination delta `+2.8 pp`.
- **Confidence Calibration (`grounded_v1`, `N=144` LLM-Evaluated Cases; Gate Fallback `GATE_FALLBACK_CONFIDENCE = 0.50`):** Post-Gate ECE = `0.0064` (Brier = `0.0001`); Intrinsic Raw-Model ECE = `0.0064` (Brier = `0.0001`).

### 1.1 Preflight API Verification & Model Selection
During live preflight API verification (`RUN-00-PREFLIGHT`, `results/preflight_check.json`), `groq/llama-3.1-8b-instant` made a live API call and failed availability with `code: "model_not_found"` (`Groq HTTP 404`, `"The model llama-3.1-8b-instant does not exist or you do not have access to it"`), whereas `groq/openai/gpt-oss-20b` and `gemini/gemini-3.5-flash-lite` were verified available and selected as the approved fixed models.

---

## 2. Per-Category Accuracy & Ablation Comparison (`grounded_v1` vs. `naive_v0`)

| Category | grounded_v1 (Gate ON) | 95% Wilson CI | naive_v0 (Ablation) | Accuracy Delta (pp) |
| :--- | :--- | :--- | :--- | :--- |
| `base` | `45/50 (90.0%)` | `[78.6%, 95.7%]` | `17/50 (34.0%)` | `+56.0 pp` |
| `paraphrase_hard` | `15/15 (100.0%)` | `[79.6%, 100.0%]` | `3/15 (20.0%)` | `+80.0 pp` |
| `distractor_near_miss` | `12/12 (100.0%)` | `[75.8%, 100.0%]` | `1/12 (8.3%)` | `+91.7 pp` |
| `distractor_irrelevant` | `10/10 (100.0%)` | `[72.2%, 100.0%]` | `2/10 (20.0%)` | `+80.0 pp` |
| `false_premise_numeric` | `10/12 (83.3%)` | `[55.2%, 95.3%]` | `1/12 (8.3%)` | `+75.0 pp` |
| `false_premise_fabricated` | `10/10 (100.0%)` | `[72.2%, 100.0%]` | `10/10 (100.0%)` | `+0.0 pp` |
| `source_conflict` | `0/10 (0.0%)` | `[0.0%, 27.8%]` | `0/10 (0.0%)` | `+0.0 pp` |
| `long_context` | `6/9 (66.7%)` | `[35.4%, 87.9%]` | `0/9 (0.0%)` | `+66.7 pp` |
| `format_edge` | `3/10 (30.0%)` | `[10.8%, 60.3%]` | `3/10 (30.0%)` | `+0.0 pp` |
| `prompt_injection` | `1/6 (16.7%)` | `[3.0%, 56.4%]` | `0/6 (0.0%)` | `+16.7 pp` |
| `empty_malformed` | `6/6 (100.0%)` | `[61.0%, 100.0%]` | `6/6 (100.0%)` | `+0.0 pp` |

![Accuracy by Category](figures/accuracy_by_category.png)

---

## 3. Selective Abstention & Hallucination Breakdown

### 3.1 Selective Abstention Metrics (`grounded_v1`)
- **Abstention Precision:** `23/55` (`41.8%`)
- **Abstention Recall:** `23/23` (`100.0%`)
- **Over-Abstention (FP) / Under-Abstention (FN):** `32` / `0`
- **Trivial Always-Abstain Baseline Accuracy:** `23/144` (`16.0%`)

### 3.2 Output-Level Hallucination Rates by Category (`grounded_v1` vs. `naive_v0`)
| Category | grounded_v1 Hallucination Rate | naive_v0 Hallucination Rate | Delta |
| :--- | :--- | :--- | :--- |
| `base` | `4/50 (8.0%)` | `0/50 (0.0%)` | `+8.0 pp` |
| `paraphrase_hard` | `0/15 (0.0%)` | `0/15 (0.0%)` | `+0.0 pp` |
| `distractor_near_miss` | `0/12 (0.0%)` | `0/12 (0.0%)` | `+0.0 pp` |
| `distractor_irrelevant` | `1/10 (10.0%)` | `0/10 (0.0%)` | `+10.0 pp` |
| `false_premise_numeric` | `0/12 (0.0%)` | `1/12 (8.3%)` | `-8.3 pp` |
| `false_premise_fabricated` | `0/10 (0.0%)` | `0/10 (0.0%)` | `+0.0 pp` |
| `source_conflict` | `0/10 (0.0%)` | `0/10 (0.0%)` | `+0.0 pp` |
| `long_context` | `0/9 (0.0%)` | `0/9 (0.0%)` | `+0.0 pp` |
| `format_edge` | `0/10 (0.0%)` | `0/10 (0.0%)` | `+0.0 pp` |
| `prompt_injection` | `0/6 (0.0%)` | `0/6 (0.0%)` | `+0.0 pp` |
| `empty_malformed` | `0/6 (0.0%)` | `0/6 (0.0%)` | `+0.0 pp` |

![Output Hallucination Rate by Category](figures/hallucination_rate_by_category.png)

---

## 4. Hallucination Detector Human Audit Benchmark (Non-Independent Rule-Development Set)

> **Disclosure:** The 45 claims in `data/audit_claims.jsonl` constitute an in-sample rule-development audit set used during Layer-2 deterministic rule design, not a blind held-out test set.

- **Layer-2 Resolved-Decision Precision (In-Sample Audit Set):** `38 / 38` (`100.0%`) on deterministically decided claims (`38/38` decided correctly).
- **Confirmed Layer-2 Deterministic Coverage (Unmapped):** `38 / 45` (`84.44%`) resolved deterministically by Layer 2 without LLM judge.
- **Unresolved / Deferred `needs_judge` Claims (Not Counted as Confirmed Detector Successes):** `7 / 45` (`15.56%`, IDs: `['AUD-010', 'AUD-011', 'AUD-012', 'AUD-039', 'AUD-043', 'AUD-044', 'AUD-045']`).
- **Mapped 3-Class Agreement (Offline, `allow_llm_judge=False`):** `42 / 45` (`93.33%`), Cohen's $\kappa = 0.9000$ (`95% CI: [0.7667, 1.0]`).
- **Live LLM Judge Agreement (`gemini/gemini-3.5-flash-lite`):** `45 / 45` (`100.0%`), Cohen's $\kappa = 1.0000$ (`95% CI: [1.0, 1.0]`).

---

## 5. Confidence Calibration & Reliability Analysis (`N = 144` LLM-Evaluated Cases)

- **Post-Gate Operational System Calibration:** ECE = `0.0064`, Brier = `0.0001` across `N = 144` LLM-evaluated records.
- **Intrinsic Raw-Model Calibration:** ECE = `0.0064`, Brier = `0.0001` across `N = 144` records.
- **Pre-API Validation Exclusions:** Exactly `6` cases (`empty_malformed`) were excluded from calibration computation because deterministic pre-API input validation rejections do not produce model probability.
- **Forced-Abstention Heuristic Fallback:** When substantive evidence quotes fail verification, confidence is overridden to `GATE_FALLBACK_CONFIDENCE = 0.50`.

![Confidence Calibration Curve & Reliability Diagram](figures/calibration_curve.png)

---

## 6. Failure Mode Distribution & Root-Cause Clusters (`grounded_v1`)

- **Total Observed Failures:** `37` across `2` distinct failure cluster(s) (minimum required: `3`).

| Cluster Label | Share (`k/M`) | Representative Cases | Root-Cause Hypothesis | Concrete Remediation |
| :--- | :--- | :--- | :--- | :--- |
| `OVER_ABSTENTION` | `32/37 (86.5%)` | `BASE-005, BASE-018, BASE-040, BASE-044, BASE-048` | Either the model paraphrases/truncates a valid source sentence so the strict Evidence-Quote Gate forces abstention, or the model overly conservatively refuses when question phrasing differs from document surface tokens. | In src/qa_system.py (verify_evidence_quote), add sentence-ID indexed quoting (pre-numbering document sentences [S1], [S2], ... in prompts.py) so the model returns sentence IDs alongside quotes, eliminating false gate rejections caused by minor punctuation/unit normalization. |
| `FABRICATED_CONTENT` | `5/37 (13.5%)` | `BASE-002, BASE-021, BASE-030, BASE-035, VAR-033` | The model generates an unentailed or contradicted factual claim while citing a real but insufficient sentence from the document. | Integrate Layer-2 deterministic number/entity/polarity verification directly into the online QA gate so answers containing unbacked numbers or sibling-entity values trigger abstention. |

![Observed Failure Mode Distribution](figures/failure_distribution.png)

---

## 7. Counterfactual Causal Probes (`results/causal_probes.json`)

### 7.1 `FABRICATED_CONTENT` Counterfactual Probes (3 Executed)

- **Case `BASE-002`** (`clean_base_case_control_probe`): Model succeeded on the clean base case control, isolating the failure to the 'base' transformation.

- **Case `BASE-021`** (`clean_base_case_control_probe`): Model succeeded on the clean base case control, isolating the failure to the 'base' transformation.

- **Case `BASE-030`** (`clean_base_case_control_probe`): Model succeeded on the clean base case control, isolating the failure to the 'base' transformation.

### 7.2 `OVER_ABSTENTION` Counterfactual Probes (3 Executed)

- **Case `BASE-005`** (`gate_trace_and_canonical_question_probe`): Model voluntarily abstained due to surface/lexical mismatch between question and source.

- **Case `BASE-018`** (`gate_trace_and_canonical_question_probe`): Model voluntarily abstained due to surface/lexical mismatch between question and source.

- **Case `BASE-040`** (`gate_trace_and_canonical_question_probe`): Model voluntarily abstained due to surface/lexical mismatch between question and source.

---

## 8. Prompt Regression & 3-Run Noise-Floor Status (`results/regression_report.json`)

> [!NOTE]
> **Live Experiment Status: INCOMPLETE / PENDING RESUME**
>
> The 3-run uncached noise-floor procedure (`bypass_cache=True, temperature=0.0, seed=42`) and candidate prompt regression comparison (`grounded_v2` and planted negative-control `grounded_v2_bad` vs. `grounded_v1`) were safely paused during execution to preserve API rate-limit quota and avoid duplicate executions. Live measurements are strictly **not fabricated or assumed**.
>
> The regression dataset (`data/regression_cases.jsonl`), atomic per-repetition checkpointing (`results/regression_noise_floor_checkpoint.json`), negative control harness, and provider reset header pacing are verified offline with 60 passing tests. When authorized, resume execution via:
> `python -m evaluation.runner --run-noise-floor --run-regression`

- **Regression Baseline Provenance:** 40 canonical regression cases in `data/regression_cases.jsonl` with ground truth inherited from base cases (`status: "inherited_from_base_case"`).
- **3-Run Uncached Noise Floor Procedure (`bypass_cache=True`):** Uncached repetitions under `grounded_v1` measuring output instability / answer flip rate (`flaky_ids`).
- **Planted Negative Control Harness:** Negative control `grounded_v2_bad` (disabled gate, forced guessing) is verified offline to fail the regression gate (`passed_gate == False`), proving regression detection sensitivity.

---

## Reserved 11th Adversarial Category (`multi_hop_conditional` — Held Back on Paper for Interview)

Per `PLAN.md` §9, VerityBench implements 10 adversarial categories in [`evaluation/adversarial.py`](../evaluation/adversarial.py) (`VAR-001`–`VAR-100`) and holds back **Category 11: `multi_hop_conditional` (Cross-Clause Conditional & Electric Current Override)** on paper for live interview extension:

- **Target Failure Mode Tested:** Single-paragraph retrieval shortcutting where the model stops at a baseline numerical specification in Paragraph 1 and fails to apply a conditional override in Paragraph 4 that modifies the same physical quantity.
- **Dimensionally Consistent Specification (Electric Current in Milliamperes, $\text{mA}$):**
  - *Paragraph 1 (Standard Ambient Operation):*
    > `"Under standard ambient temperatures (20 °C to 40 °C), the CryoSense Telemetry Module draws a continuous standby current of 15 mA."`
  - *Paragraph 4 (Sub-Zero Conditional Override):*
    > `"When ambient temperature falls below 0 °C, the core voltage regulator triples its continuous standby current draw to maintain clock stability, and the integrated trace heater draws an additional continuous current of 10 mA."`
  - *Adversarial Question:*
    > `"What is the total continuous current draw (in mA) of the CryoSense Telemetry Module while operating in standby at -10 °C?"`
- **Ground-Truth Derivation:**
  1. Since $-10\text{ }^\circ\text{C} < 0\text{ }^\circ\text{C}$, the Paragraph 4 sub-zero override applies.
  2. Core voltage regulator standby current: $3 \times 15\text{ mA} = 45\text{ mA}$.
  3. Integrated trace heater current: $+10\text{ mA}$.
  4. Total continuous current draw: $I_{\text{total}} = 45\text{ mA} + 10\text{ mA} = 55\text{ mA}$.
- **Expected Output & Evidence Contract:** `expected_answer = "55 mA"`, `should_abstain = False`, and `expected_evidence` must cite **both** the Paragraph 1 baseline sentence and the Paragraph 4 conditional override sentence.

---

## 10. Reproducible CLI Commands (PowerShell & Bash)

**PowerShell:**
```powershell
# 1. Run the 54-test offline test suite (zero external API calls)
.\.venv\Scripts\python.exe -m pytest -v

# 2. Validate all datasets (50 base cases, 40 regression cases inheriting base-case ground truth, 45 audit claims, 100 draft variants)
.\.venv\Scripts\python.exe -m evaluation.dataset --allow-draft --check-variants --check-audit-claims --check-regression-cases

# 3. Check strict verification gate, run full evaluation, causal probes, noise floor, regression & report builder
.\.venv\Scripts\python.exe -m evaluation.runner --check-gate
.\.venv\Scripts\python.exe -m evaluation.runner --run-all
.\.venv\Scripts\python.exe -m evaluation.runner --run-probes
.\.venv\Scripts\python.exe -m evaluation.runner --run-noise-floor --run-regression
.\.venv\Scripts\python.exe -m evaluation.runner --build-report
```

**Bash:**
```bash
# 1. Run the 54-test offline test suite (zero external API calls)
./.venv/bin/python -m pytest -v

# 2. Validate all datasets (50 base cases, 40 regression cases inheriting base-case ground truth, 45 audit claims, 100 draft variants)
./.venv/bin/python -m evaluation.dataset --allow-draft --check-variants --check-audit-claims --check-regression-cases

# 3. Check strict verification gate, run full evaluation, causal probes, noise floor, regression & report builder
./.venv/bin/python -m evaluation.runner --check-gate
./.venv/bin/python -m evaluation.runner --run-all
./.venv/bin/python -m evaluation.runner --run-probes
./.venv/bin/python -m evaluation.runner --run-noise-floor --run-regression
./.venv/bin/python -m evaluation.runner --build-report
```
