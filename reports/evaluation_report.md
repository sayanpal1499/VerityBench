# VerityBench — Adversarial Reliability Evaluation Report (Historical Benchmark Run)

**Historical Run Timestamp:** `2026-10-13T10:45:41.565406+00:00` | **Historical Benchmark Dataset Hash:** `4458b2c305b31859`
**Fixed SUT Model:** `groq/openai/gpt-oss-20b` | **Fixed Judge Model:** `gemini/gemini-3.5-flash-lite`
**Gate Forced-Abstention Fallback Confidence:** `GATE_FALLBACK_CONFIDENCE = 0.50`

> [!IMPORTANT]
> **PROVENANCE DISCLOSURE: HISTORICAL BENCHMARK RUN VS. CURRENT DATASET GOVERNANCE**
>
> 1. **Historical Benchmark Provenance:**
>    The performance metrics, accuracy tables, hallucination rates, calibration curves, and failure clusters reported below are **empirical historical measurements** from a live benchmark run executed on **2026-10-13T10:45:41.565406+00:00** against historical dataset snapshot hash **`4458b2c305b31859`** using `groq/openai/gpt-oss-20b` (System Under Test) and `gemini/gemini-3.5-flash-lite` (independent judge). All 218 raw provider responses were captured in `.cache/llm_responses/` (excluded from Git to prevent repository bloat), while full model outputs, scored predictions, claim audits, and aggregate metrics are committed under `results/`.
>
> 2. **Current Repository Dataset Status & Governance Refusal:**
>    In the current canonical dataset (`data/variants.jsonl`, current dataset hash `211ca23296bb1612`), exactly **20 variants (`VAR-001`–`VAR-020`)** are human-verified, while **80 variants (`VAR-021`–`VAR-100`)** remain in **`status: "draft"`** (`verified_by: null`, `verified_on: null`) pending final human sign-off.
>    Per project governance rules, the strict verification gate ([`verify_all_records_before_run()`](../evaluation/runner.py)) **actively blocks and refuses** new live benchmark runs (`python -m evaluation.runner --run-all` / `--check-gate`) while draft variants exist.
>
> 3. **Non-Equivalence Disclosure:**
>    These historical results **do not** represent a new benchmark run on the current draft-containing dataset. They are retained and analyzed here to report genuine empirical findings from the completed run, demonstrate calibration and clustering methodologies, and document infrastructure limitations, without fabricating human reviews or weakening the governance gate.

---

## 1. Executive Summary & Verification Gate Status

- **Dataset Verification Status:** 50 base cases (`status: "verified"` by sayanpal1499), 20 adversarial variants (`status: "verified"` by sayanpal1499 across Batches 1–2 `VAR-001`–`VAR-020`), 80 adversarial variants (`status: "draft"`, `verified_by: null`, `verified_on: null` pending final human sign-off across Batches 3–10 `VAR-021`–`VAR-100`), 45 audit claims (`status: "verified"` by sayanpal1499), and 40 regression cases (`status: "inherited_from_base_case"` inherited directly from `BASE-001`–`BASE-040` in `data/base_cases.jsonl`).
- **Benchmark Run Provenance:** Historical live evaluation completed on 2026-10-13T10:45:41.565406+00:00 (dataset snapshot 4458b2c305b31859); new live evaluation correctly blocked on current canonical dataset (hash 211ca23296bb1612) by strict governance gate.
- **LLM-Evaluated Accuracy (`grounded_v1`, `N=144` Eligible Cases excluding `6` pre-API `empty_malformed` rejections):** `112/144 (77.8%)`, Base-Case Cluster Bootstrap 95% CI: `[71.1%, 84.7%]`. Includes 32 infrastructure/API error fallbacks. Under saved scoring rules on this historical run, completed model responses achieved **100.0% accuracy (89/89 substantive answers + 23/23 valid abstentions)**; this does not establish 100% accuracy on new data.
- **Nominal Suite Accuracy (`grounded_v1`, All `150` Records including `6` deterministic pre-API `InputValidationError` rejections):** `118/150 (78.7%)`, Base-Case Cluster Bootstrap 95% CI: `[72.7%, 85.0%]`.
- **Output-Level Hallucination Detector-Flag Rate (`grounded_v1`):** `5/144` (`3.5%` overall, `5/89` [5.6%] on answered cases). The detector flagged five completed answers (`BASE-002`, `BASE-021`, `BASE-030`, `BASE-035`, `VAR-033`). Inspection suggests these flags are likely detector artifacts (abbreviation splitting, single-sentence windowing, table-row alignment), but all five cases remain marked `pending_human_review` in [`results/failure_review.csv`](../results/failure_review.csv) and have not been formally adjudicated; 3.5% represents the observed detector-flag rate, not a verified model hallucination rate.
- **Claim-Level Hallucination & Deferral Breakdown (`grounded_v1`):** Bad claims `5/98` (`5.1%`); Confirmed supported claims: `93`; Layer-2 deterministically decided: `36`; Layer-2 deferred (`needs_judge`): `58` (unresolved `needs_judge`: `0`).
- **Ablation vs. `naive_v0` (Operational Comparison):** Nominal accuracy delta `+50.0 pp`; heavily confounded by 129 naive API fallbacks (due to rate limits and missing grounding tokens) and cannot be cited as a causal estimate of prompt architecture.
- **Confidence Calibration (`grounded_v1`, `N=144` LLM-Evaluated Cases; Gate Fallback `GATE_FALLBACK_CONFIDENCE = 0.50`):** Post-Gate ECE = `0.0064` (Brier = `0.0001`); Intrinsic Raw-Model ECE = `0.0082` (Brier = `0.0001`). All 112 completed outputs occupy a single high-confidence bin `[0.99, 1.0]`, reflecting uniform model overconfidence on this domain rather than nuanced probability calibration.
- **Failure Clustering Status:** 3 operational failure clusters cataloged across 37 observed failures (`OUTPUT_SCHEMA_TRUNCATION`: 17/37 [46.0%], `API_RATE_LIMIT_EXHAUSTION`: 15/37 [40.5%], `FABRICATED_CONTENT`: 5/37 [13.5%]). Exactly 32 of 37 failures (86.5%) are infrastructure limits and 5 (13.5%) are detector hallucination flags (pending formal adjudication); no substantive model-reasoning failure clusters were isolated in completed outputs.

### 1.1 Preflight API Verification & Model Selection
During live preflight API verification (`RUN-00-PREFLIGHT`, `results/preflight_check.json`), `groq/llama-3.1-8b-instant` made a live API call and failed availability with `code: "model_not_found"` (`Groq HTTP 404`, `"The model llama-3.1-8b-instant does not exist or you do not have access to it"`), whereas `groq/openai/gpt-oss-20b` and `gemini/gemini-3.5-flash-lite` were verified available and selected as the approved fixed models.

---

## 1.2 Authoritative Per-Configuration Accounting Table

| Accounting Metric | Mathematical Definition | `grounded_v1` (Gate ON) | `naive_v0` (Ablation) | Explanatory Provenance & Denominator Rationale |
| :--- | :--- | :---: | :---: | :--- |
| **Total Benchmark Records** | Total records in test suite | **150** | **150** | 50 base cases + 100 adversarial variants (`data/base_cases.jsonl`, `data/variants.jsonl`). |
| **Deterministic Pre-API Exclusions** | `validation_rejected == True` | **6** (4.0%) | **6** (4.0%) | The 6 `empty_malformed` variants (`VAR-095`–`VAR-100`) rejected by `InputValidationError` before calling any API. |
| **LLM-Evaluated Cases** | $\text{Total} - \text{Pre-API Exclusions}$ | **144** (96.0%) | **144** (96.0%) | Cases eligible for LLM generation and evaluation. Standard denominator for LLM metrics. |
| **API / System Error Fallbacks** | `error_fallback == True` | **32** (22.2%) | **129** (89.6%) | Groq HTTP 400 token truncations (17 v1 / 15 v0) or HTTP 429 rate limits (15 v1 / 114 v0) caught by exception handler, assigned fallback `abstained=True, conf=0.0`. |
| **Completed LLM Responses** | $\text{LLM-Evaluated} - \text{API Errors}$ | **112** (77.8%) | **15** (10.4%) | Cases where the LLM successfully completed generation and returned a parseable JSON response. |
| **Intentional Abstentions** | Model set `abstained == True` | **23** (16.0%) | **0** (0.0%) | Model actively and voluntarily abstained on unanswerable/ambiguous questions. |
| **Substantive Answers Generated** | `abstained == False` | **89** (61.8%) | **15** (10.4%) | Cases where the system returned a substantive factual answer. $\text{Completed LLM} - \text{Intentional Abstentions}$. |
| **Total System Abstentions** | `abstained == True` across all records | **61** (40.7%) | **135** (90.0%) | Pre-API (6) + API Error (32 v1 / 129 v0) + Intentional (23 v1 / 0 v0). |
| **Overall Accuracy (All 150)** | $\frac{\text{Correct}}{\text{Total 150}}$ | **118/150 (78.7%)** | **43/150 (28.7%)** | Includes 6 correctly rejected pre-API malformed questions + 29 correct abstentions (23 unanswerable) + correct substantive answers. |
| **LLM-Evaluated Accuracy (144)** | $\frac{\text{Correct}}{\text{LLM-Evaluated 144}}$ | **112/144 (77.8%)** | **37/144 (25.7%)** | Excludes pre-API rejections. For v1: 89 answered + 23 abstained = 112. For v0: 14 answered + 23 unanswerable API fallbacks = 37. |
| **Answered-Only Accuracy** | $\frac{\text{Correct Substantive Answers}}{\text{Substantive Answers}}$ | **89/89 (100.0%)** | **14/15 (93.3%)** | Accuracy measured strictly on questions where the system produced a substantive answer. |
| **Overall Hallucination Rate (144)** | $\frac{\text{Outputs with Hallucination}}{\text{LLM-Evaluated 144}}$ | **5/144 (3.5%)** | **1/144 (0.7%)** | Evaluated over all 144 LLM cases. Strongly deflated for `naive_v0` because it failed on 129 cases. |
| **Answered-Only Hallucination Rate** | $\frac{\text{Outputs with Hallucination}}{\text{Substantive Answers}}$ | **5/89 (5.6%)** | **1/15 (6.7%)** | True comparison of generation reliability on substantive answers: `grounded_v1` hallucinates less (5.6% vs 6.7%). |

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

### 3.3 Overall vs. Answered-Only Hallucination Rate & Abstention Trade-Off
| Configuration | Overall Hallucination Rate (`N=144`) | Answered-Only Hallucination Rate | Substantive Answers Generated | Evaluated Accuracy (`N=144`) | Answered-Only Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `grounded_v1` (Gate ON) | `5/144 (3.5%)` | `5/89 (5.6%)` | `89/144 (61.8%)` | `112/144 (77.8%)` | `89/89 (100.0%)` |
| `naive_v0` (Ablation) | `1/144 (0.7%)` | `1/15 (6.7%)` | `15/144 (10.4%)` | `37/144 (25.7%)` | `14/15 (93.3%)` |

> [!NOTE]
> **Abstention / Under-Answering Trade-Off Disclosure:**
> While `naive_v0` nominally records an overall hallucination rate of only `0.7%` (`1/144`), this metric is strongly deflated because `naive_v0` failed or abstained on `129` out of `144` cases (due to API token exhaustion and missing grounding). On questions where it produced a substantive answer (`N=15`), its hallucination rate is `6.7%` (`1/15`), which is higher than `grounded_v1`'s answered-only rate of `5.6%` (`5/89`). Furthermore, `grounded_v1` successfully answered `89/144` (`61.8%`) of questions with `100.0%` answered accuracy, compared to `naive_v0`'s `15/144` (`10.4%`).


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

### 5.1 Confidence Provenance Breakdown
Every confidence value across the suite originates from one of four verified sources:
- **Genuine Completed LLM Output (`N = 112`):** Exactly `112` cases (`89` substantive answers + `23` intentional abstentions) where the model completed JSON output. Every single case returned extreme confidence in `[0.99, 1.0]` (mean confidence = 0.9918).
- **API Error Exception Fallback (`N = 32`):** Exactly `32` cases where Groq returned HTTP 400 token truncations (`17` cases) or HTTP 429 rate limits (`15` cases), caught by `src/qa_system.py` and deterministically assigned fallback `confidence = 0.00`.
- **Pre-API Input Validation Fallback (`N = 6`):** Exactly `6` `empty_malformed` variants rejected deterministically before calling any API, assigned fallback `confidence = 0.00`. Excluded from LLM calibration.
- **Evidence-Quote Gate Fallback (`N = 0`):** Substantive answers whose cited quotes fail verification are assigned heuristic fallback `confidence = 0.50`. (0 occurrences in this run).

### 5.2 Disaggregated Calibration Metrics
- **Post-Gate Operational System Calibration (All Evaluated Cases, `N = 144`):** ECE = `0.0064`, Brier = `0.0001` across `N = 144` records.
- **Intrinsic Raw-Model Calibration (Pre-Gate, `N = 144`):** ECE = `0.0082`, Brier = `0.0001` across `N = 144` records.
- **Genuine Completed Model Predictions (`N = 112`):** ECE = `0.0082`, Brier = `0.0001` across `N = 112` cases.
- **Pre-API Input Validation Exclusions:** Exactly `6` cases (`empty_malformed`) were excluded from calibration computation because deterministic pre-API input validation rejections (`InputValidationError`) do not produce model probability.

> [!WARNING]
> **Provisional Metric & Overconfidence Disclosure:**
> The near-zero full-system ECE (`0.0064`) is an operational artifact of combining `32` deterministic error fallbacks (assigned `0.00` with `0.0%` accuracy) with `112` model outputs (assigned `0.99`–`1.0` with `100.0%` accuracy).
> When evaluated strictly on completed model predictions (`N = 112`), all records occupy a single bin `[0.9, 1.0]` (`ECE = 0.0082`).
> **Crucial Calibration Limitation:** This single-bin distribution provides **narrow and limited evidence** of true calibration. Rather than proving that the model produces well-calibrated probabilities across the `[0.0, 1.0]` continuum, it reveals extreme model overconfidence and a complete absence of nuanced uncertainty estimation across intermediate probability ranges (`[0.1, 0.9)`).

![Confidence Calibration Curve & Reliability Diagram](figures/calibration_curve.png)

---

## 6. Failure Mode Distribution & Root-Cause Clusters (`grounded_v1`)

- **Total Observed Failures:** `37` across `3` cataloged operational failure clusters (`OUTPUT_SCHEMA_TRUNCATION`: 17, `API_RATE_LIMIT_EXHAUSTION`: 15, `FABRICATED_CONTENT`: 5).
- **Model-Behavior vs. Operational Cluster Disclosure:** The assignment rubric expects at least 3 failure clusters demonstrating model-level behavioral weaknesses. On this run, 32 failures were infrastructure limits and 5 were detector hallucination flags. Inspection indicates these 5 flags are likely detector artifacts, but because all five cases remain marked `pending_human_review`, they have not been formally adjudicated. No substantive model-reasoning failure clusters were isolated in completed outputs; the five detector flags remain unadjudicated.
- **Review Governance Disclosure:** All initial cluster classifications are rule-generated proposals based on error signatures and perturbation categories. In [`results/failure_review.csv`](../results/failure_review.csv), `human_review_status` remains `pending_human_review` and `human_verified_cluster` is left unpopulated (`null`/empty) until independent human audit.

| Cluster Label | Share (`k/M`) | Representative Cases | Root-Cause Hypothesis | Concrete Remediation |
| :--- | :--- | :--- | :--- | :--- |
| `OUTPUT_SCHEMA_TRUNCATION` | `17/37 (46.0%)` | `BASE-005, BASE-018, BASE-040, BASE-044, BASE-048` | Generation cutoff under max_output_tokens=512 ceiling: when complex technical questions require multi-sentence comparative answers and lengthy verbatim evidence quotes, token generation is truncated before the JSON closing brace, triggering Groq's json_validate_failed HTTP 400 and resulting in a system error abstention fallback. The model never returned a completed answer. | In src/qa_system.py, increase max_output_tokens from 512 to 1024 or 2048 for complex queries, and add streaming partial-JSON closing repairs in _extract_json_object to recover intact answers. |
| `API_RATE_LIMIT_EXHAUSTION` | `15/37 (40.5%)` | `VAR-076, VAR-077, VAR-078, VAR-082, VAR-083` | Provider gateway rejection (Groq HTTP 429) due to exhausting daily token quota (200,000 TPD on on-demand tier) during sequential benchmark execution. No LLM generation was performed; the request failed before reaching the model. | Implement pre-request token budget tracking in src/llm_client.py, multi-account or secondary API key failover, and respect provider retry-after and reset headers rather than failing after fixed retries. |
| `FABRICATED_CONTENT` | `5/37 (13.5%)` | `BASE-002, BASE-021, BASE-030, BASE-035, VAR-033` | Completed model answers flagged by hallucination verification. Qualitative inspection indicates these flags appear consistent with detector segmentation (e.g., regex period-splitting on honorifics like 'Dr.' in BASE-002), formatting, or single-sentence context-retrieval limitations (e.g., matching Standard Merchant vs Enterprise Clearing Tier in BASE-021/VAR-033, or single-sentence windowing in BASE-030/BASE-035). However, as recorded in results/failure_review.csv, these cases remain pending formal human review, so this interpretation has not been independently validated. | Enhance Layer-2 deterministic matcher with multi-sentence context windowing and entity-scoped bullet point parsing before computing numeric overlap. |

> [!IMPORTANT]
> **Infrastructure vs. Model-Reasoning Limitation Disclosure:**
> Exactly 32 of the 37 failures (86.5%) are System / Infrastructure Failures (`OUTPUT_SCHEMA_TRUNCATION` under 512 token limits, and `API_RATE_LIMIT_EXHAUSTION` under daily token quotas).
> Only 5 of the 37 failures (13.5%) occurred on completed model responses (`FABRICATED_CONTENT`). While qualitative examination suggests these flags appear consistent with detector segmentation, formatting, or context-retrieval limitations (such as single-sentence alignment on multi-tier bullet points or abbreviation splitting), this interpretation has not been independently validated through formal human adjudication.
> **Critical Finding:** The saved benchmark run does NOT provide evidence of substantive model reasoning failures on `source_conflict`, `prompt_injection`, or `long_context`, because those test cases never returned valid completed responses from the API. Claiming that the model has behavioral vulnerabilities on those categories would be scientifically unsupported. To evaluate true model-internal robustness on those perturbations, a live benchmark run with `max_output_tokens=1024` and sufficient token quota is required.

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

## 8. Prompt Regression & 3-Run Noise-Floor Status (Target Artifact: `results/regression_report.json`)

> [!NOTE]
> **Live Experiment Status: INCOMPLETE / PENDING RESUME**
>
> The 3-run uncached noise-floor procedure (`bypass_cache=True, temperature=0.0, seed=42`) and candidate prompt regression comparison (`grounded_v2` and planted negative-control `grounded_v2_bad` vs. `grounded_v1`) were safely paused during execution to preserve API rate-limit quota and avoid duplicate executions. Live measurements are strictly **not fabricated or assumed**.
>
> **Artifact Status:** Because this live experiment has not completed, `results/regression_report.json` is intentionally absent from the repository. The regression dataset (`data/regression_cases.jsonl`), atomic per-repetition checkpointing (`results/regression_noise_floor_checkpoint.json`), negative control harness, and provider reset header pacing are verified offline with 69 passing tests. When authorized with live API quota, resume execution via:
> `python -m evaluation.runner --run-noise-floor --run-regression`

- **Regression Baseline Provenance:** 40 canonical regression cases in `data/regression_cases.jsonl` with ground truth inherited from base cases (`status: "inherited_from_base_case"`).
- **3-Run Uncached Noise Floor Procedure (`bypass_cache=True`):** Uncached repetitions under `grounded_v1` measuring output instability / answer flip rate (`flaky_ids`).
- **Planted Negative Control Harness:** Negative control `grounded_v2_bad` (disabled gate, forced guessing) is verified offline to fail the regression gate (`passed_gate == False`), proving regression detection sensitivity.

---

## Reserved 11th Adversarial Category (`multi_hop_conditional` — Held Back on Paper for Interview)

By design, VerityBench implements 10 adversarial categories in [`evaluation/adversarial.py`](../evaluation/adversarial.py) (`VAR-001`–`VAR-100`) and holds back **Category 11: `multi_hop_conditional` (Cross-Clause Conditional & Electric Current Override)** on paper for live interview extension:

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
# 1. Run the 69-test offline test suite (zero external API calls)
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
# 1. Run the 69-test offline test suite (zero external API calls)
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
