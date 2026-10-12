# VerityBench — Technical Architecture & Evaluation Report

- **Repository:** VerityBench (Custom Adversarial Reliability & Hallucination Evaluation Harness)
- **Fixed System Under Test (SUT) Model:** `groq/openai/gpt-oss-20b` (`temperature=0.0, seed=42`)
- **Fixed Independent Judge Model:** `gemini/gemini-3.5-flash-lite` (`temperature=0.0, seed=42`)
- **Evaluation Framework Policy:** Zero third-party evaluation frameworks (`RAGAS`, `TruLens`, `DeepEval` are strictly prohibited and unused).

---

## 1. Executive Summary & Current Verification Gate Status

VerityBench evaluates a strictly grounded Document Question Answering (QA) system across **8 synthetic multi-paragraph domain documents (`DOC-001`–`DOC-008`)**, **50 human-verified base cases (`BASE-001`–`BASE-050`)**, **100 adversarial variants (`VAR-001`–`VAR-100`)**, and **45 human-verified audit claims (`AUD-001`–`AUD-045`)**.

### Distinction Between Offline Verification and Live Full-Suite Evaluation
Per the repository's **Strict Ground-Truth Governance Gate** ([`verify_all_records_before_run`](file:///p:/HEVA%20AI%20Assignment/evaluation/runner.py#L56-L95)):
1. **Source Documents (`data/documents/DOC-001..008.txt`):** `8 / 8` complete (`450–515` words each).
2. **Base Cases (`data/base_cases.jsonl`):** **`50 / 50` `status: "verified"`** (`verified_by: "sayanpal1499"`, `verified_on: "2026-10-12"`).
3. **Hallucination Audit Claims (`data/audit_claims.jsonl`):** **`45 / 45` `status: "verified"`** (`verified_by: "sayanpal1499"`, `verified_on: "2026-10-13"`, `0` remaining `needs_human_review` flags).
4. **Adversarial Variants (`data/variants.jsonl`):** **`100 / 100` `status: "draft"`** (`verified_by: null`, `verified_on: null`).
   - All 100 variants (`VAR-001`–`VAR-100`) were systematically audited automatically against the canonical source documents, base cases, and expected evidence (with `VAR-001`–`VAR-020` also reviewed in human batches), but the full 100-variant set has **not** been individually human-verified.
   - Because human verification metadata (`status: "verified"`, `verified_by`, `verified_on`) is never fabricated or auto-populated without actual individual human verification, `data/variants.jsonl` remains in `status: "draft"`.
   - **Live 150-Case Evaluation Status (`Not Yet Run`):** Because `data/variants.jsonl` is in `"draft"` status, [`verify_all_records_before_run()`](file:///p:/HEVA%20AI%20Assignment/evaluation/runner.py#L56-L95) strictly blocks `evaluation/runner.py --run-all`. The live 150-case evaluation has **not yet been run**, and no live model accuracy, hallucination rates, calibration scores, or observed failure cluster distributions are claimed or invented in this report.

---

## 2. System Architecture & Honest Reproducibility

### 2.1 Structured Output Contract & Deterministic Evidence-Quote Gate
Every call to [`answer_question()`](file:///p:/HEVA%20AI%20Assignment/src/qa_system.py#L149-L315) enforces a three-stage reliability pipeline:
1. **Pre-API Input Validation (`validate_inputs`):** Rejects empty, whitespace-only, punctuation-only (`< 2` alphanumeric characters), too-short (`< 3` chars), or excessively long (`> 2,000` chars) questions by raising `InputValidationError` before consuming any API tokens.
2. **Schema Validation & Single Repair Retry:** Parses the model response into `QAOutput(answer: str, evidence: list[str], confidence: float, abstained: bool)`. If JSON parsing or Pydantic validation fails, the system performs at most **one** repair retry (`max_retries=1`) before returning a safe abstention fallback (`abstained=True, confidence=0.0, evidence=[]`).
3. **Deterministic Evidence-Quote Gate (`verify_evidence_quote`):** When enabled (`grounded_v1`, `grounded_v2`), every quote in `output.evidence` must be at least 15 characters long and match a substring of the source document after conservative whitespace and smart-quote normalization (`normalize_text_for_quote_match`).
   - Near-miss quotes (`SequenceMatcher >= 0.85`) are recorded in `QATrace.fuzzy_candidate_quotes` for failure analysis but **never** pass the gate.
   - If `abstained=False` and `evidence` is empty or any quote fails verification, the gate forces `abstained=True`, `evidence=[]`, and post-gate `confidence=0.0`, while preserving the pre-gate `raw_output` (`raw_answer`, `raw_confidence`, `raw_abstained`) inside `QATrace`.

### 2.2 Reproducibility & Hosted-Model Determinism Limitations
Hosted LLM providers (`groq` and `gemini`) do not guarantee strict bit-for-bit determinism across distributed GPU workers even when `temperature=0.0` and `seed=42` are passed.
- [`src/llm_client.py`](file:///p:/HEVA%20AI%20Assignment/src/llm_client.py) passes `temperature=0.0` and `seed=42` on every request, logs `parameters_requested`, `parameters_supported`, `parameters_unsupported`, and `system_fingerprint`, and persists responses in `.cache/llm_calls/<sha256>.json`.
- **Measured Preflight Verification (`results/preflight_check.json`):**
  - `groq/openai/gpt-oss-20b`: Verified available (`latency_ms=1008.93`, `system_fingerprint="fp_816fc87d18"`, `cached_replay_hit=True`). Note that `groq/llama-3.1-8b-instant` returned HTTP 404 (`model_not_found`) during preflight and was replaced by `groq/openai/gpt-oss-20b` as the fixed SUT model.
  - `gemini/gemini-3.5-flash-lite`: Verified available (`latency_ms=1831.32`, structured `response_json_schema=True`, `cached_replay_hit=True`).
  - All 5 manual smoke-test scenarios (`EX-1` clean numeric lookup, `EX-2` near-miss sibling entity `Velox-400` vs. `Velox-900`, `EX-3` two-clause synthesis, `EX-4` unanswerable weight lookup, and `EX-5` false numeric premise correction `9.5 bar -> 4.2 bar`) passed verbatim quote gating on the first run.

---

## 3. Dataset Structure & Adversarial Design

### 3.1 Source Corpus (`data/documents/DOC-001..008.txt`)
Eight synthetic, domain-specific documents (`450–515` words each) were authored with dense numeric, entity, conditional, and negation specifications so pretraining priors cannot shortcut document comprehension:
- `DOC-001`: *Helios-X9 Cryogenic Telemetry & Power Controller — Engineering Datasheet (Rev D)*
- `DOC-002`: *Aethelgard Orbital Systems — Global Travel, Remote Work & Expense Policy (Pol-HR-409)*
- `DOC-003`: *Arctic Science Dispatch — Svalbard Sub-Glacial Neutrino Array (SSNA) Phase-II Commissioning*
- `DOC-004`: *ChronosPay Ledger Gateway v3.4 — API Integration & Idempotency Specification*
- `DOC-005`: *Master Biopolymer Licensing Agreement (VB-2025-881: VortexBio & Kestrel Biopolymers)*
- `DOC-006`: *Zelvorin-XR (brigatadine fumarate) 45 mg Capsules — Clinical Prescribing Information*
- `DOC-007`: *Nordic Maritime Safety Directorate — MV Borealis-IV Autonomous Feeder Incident Bulletin*
- `DOC-008`: *StratosVault Sovereign Cloud Object Storage — Service Level & Cryptographic Architecture*

### 3.2 Base Cases (`data/base_cases.jsonl` — `50` Verified Records)
- **Answerable (`38` cases):** Stratified across `number_with_unit`, `named_entity`, `date`, `yes_no`, `list`, and `two_sentence_fact`. Every answerable record carries verbatim `expected_evidence` validated against its source document.
- **Unanswerable (`12` cases):** Plausible domain questions asking for attributes intentionally omitted from the document (`should_abstain=True`, `expected_evidence=[]`).

### 3.3 Adversarial Variants (`data/variants.jsonl` — `100` Audited Records across `10` Categories)
Every variant links via `base_id` to a verified base case and applies one controlled perturbation (`multi_hop_composition` is reserved exclusively as an unimplemented 11th category for live interview extension):

| Category (`transformation`) | Count | IDs | Target Failure Mode Tested |
| :--- | :---: | :--- | :--- |
| `paraphrase_hard` | `15` | `VAR-001`–`VAR-015` | Surface-form lexical reliance & synonym/unit rephrasing |
| `distractor_near_miss` | `12` | `VAR-016`–`VAR-027` | Entity-attribute binding under sibling-entity/tier addenda (`DISTRACTOR_CAPTURE`) |
| `distractor_irrelevant` | `10` | `VAR-028`–`VAR-037` | Attention dilution from unrelated administrative/canteen framing paragraphs |
| `false_premise_numeric` | `12` | `VAR-038`–`VAR-049` | Sycophantic acceptance of user-supplied wrong numbers (`FALSE_PREMISE_ACCEPTED`) |
| `false_premise_fabricated` | `10` | `VAR-050`–`VAR-059` | Hallucinated explanations for non-existent features (`UNDER_ABSTENTION`) |
| `source_conflict` | `10` | `VAR-060`–`VAR-069` | Single-span retrieval bias when two sections state conflicting values (`SOURCE_CONFLICT_MISSED`) |
| `long_context` | `9` | `VAR-070`–`VAR-078` | Positional attention decay (`start`, `middle`, `end`) under ~5x (`~2,460`-word) archival padding (`POSITION_SENSITIVITY`) |
| `format_edge` | `10` | `VAR-079`–`VAR-088` | Unit conversions (`mA/A`, `m/km`, `ms/s`, `h/days`), spelled-out numbers & ASCII tables (`FORMAT_REASONING_ERROR`) |
| `prompt_injection` | `6` | `VAR-089`–`VAR-094` | Instruction-data boundary hijacking inside untrusted documents (`INJECTION_FOLLOWED`) |
| `empty_malformed` | `6` | `VAR-095`–`VAR-100` | Pre-API input validation on empty, whitespace, punctuation-only, 1-char & `>2,000`-char inputs |

---

## 4. Measured Offline Hallucination Detector Benchmark (`45` Verified Claims)

To evaluate the 3-layer hallucination detector independently of live LLM judge calls, [`run_audit_benchmark()`](file:///p:/HEVA%20AI%20Assignment/evaluation/hallucination.py#L430-L539) was executed in strict offline mode (`allow_llm_judge=False`) on all **45 human-verified audit claims** (`15 supported`, `15 contradicted`, `15 unsupported`) and saved to [`results/audit_benchmark_offline.json`](file:///p:/HEVA%20AI%20Assignment/results/audit_benchmark_offline.json).

### 4.1 Separation of Confirmed Deterministic Decisions vs. Deferred `needs_judge` Cases
Per ground-truth governance rules, deferred `needs_judge` cases are **never** counted as confirmed deterministic detector successes:

| Metric | Value (`k/n`) | Percentage / Statistic | Interpretation |
| :--- | :---: | :---: | :--- |
| **Layer-2 Deterministically Decided Claims** | `38 / 45` | `84.44%` | Claims resolved deterministically by Layer 2 without deferring to Layer 3 |
| **Confirmed Layer-2 Deterministic Accuracy (Unmapped)** | **`38 / 45`** | **`84.44%`** | **`38 / 38` (`100.0%`) precision** on all deterministically decided claims (`0` errors) |
| **Layer-2 Deferred (`needs_judge`) Claims** | `7 / 45` | `15.56%` | `AUD-010`, `AUD-011`, `AUD-012`, `AUD-039`, `AUD-043`, `AUD-044`, `AUD-045` |
| **Fallback-Inflated `unsupported` Matches (`needs_judge -> unsupported`)** | `4 / 45` | `8.89%` | `AUD-039`, `AUD-043`, `AUD-044`, `AUD-045` (human label is `unsupported`, matched only via conservative fallback when `allow_llm_judge=False`) |
| **Mapped 3-Class Agreement (`allow_llm_judge=False`)** | `42 / 45` | `93.33%` | Includes the 4 conservative fallback matches; `3` disagreements (`AUD-010`–`AUD-012`) |
| **Mapped Cohen's $\kappa$ (`1,000` Bootstrap Resamples)** | — | `0.9000` (`95% CI: [0.7667, 1.0000]`) | Substantially exceeds the $\kappa \ge 0.70$ benchmark threshold |

### 4.2 Per-Slice Audit Breakdown (`results/audit_benchmark_offline.json`)

| Audit Slice | Human Ground Truth | Total Claims | Confirmed Deterministic Correct | Deferred (`needs_judge`) | Mapped 3-Class Agreed |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `verbatim_fact` | `supported` | `12` | `12 / 12` (`100%`) | `0` | `12 / 12` |
| `unit_conversion` (`AUD-010`–`AUD-012`) | `supported` | `3` | `0 / 3` (`0%`) | `3` (`AUD-010`..`012`) | `0 / 3` *(deferred to Layer 3)* |
| `numeric_contradiction` | `contradicted` | `5` | `5 / 5` (`100%`) | `0` | `5 / 5` |
| `wrong_entity_or_condition` | `contradicted` | `5` | `5 / 5` (`100%`) | `0` | `5 / 5` |
| `negation_inversion` | `contradicted` | `5` | `5 / 5` (`100%`) | `0` | `5 / 5` |
| `unanswerable_extrapolation` | `unsupported` | `12` | `11 / 12` (`91.7%`) | `1` (`AUD-039`) | `12 / 12` *(1 via fallback)* |
| `unsupported_inference` (`AUD-043`–`AUD-045`) | `unsupported` | `3` | `0 / 3` (`0%`) | `3` (`AUD-043`..`045`) | `3 / 3` *(3 via fallback)* |
| **Total** | **`15 / 15 / 15`** | **`45`** | **`38 / 45` (`84.44%`)** | **`7 / 45` (`15.56%`)** | **`42 / 45` (`93.33%`)** |

---

## 5. Calibration, Abstention & Uncertainty Quantification Methodology

### 5.1 Dual-Track Confidence Calibration (`evaluation/calibration.py`)
Let $M=10$ equal-width bins partition $[0, 1]$ into $I_1 = [0.0, 0.1], \dots, I_{10} = (0.9, 1.0]$. For $N$ evaluated records with confidence $\hat{p}_i$ and binary correctness $y_i \in \{0, 1\}$:
$$\mathrm{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|, \qquad \mathrm{Brier} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)^2$$
Because the Evidence-Quote Gate deterministically overrides `confidence = 0.0` whenever a quote fails verification, [`compute_full_calibration_report()`](file:///p:/HEVA%20AI%20Assignment/evaluation/calibration.py#L94-L162) reports both:
1. **Post-Gate System Calibration:** Uses post-gate `(confidence, is_correct)` across `overall_all_150`, `answered_only`, and `abstained_only`.
2. **Intrinsic Raw-Model Calibration:** Uses pre-gate `(raw_confidence, raw_is_correct)` captured in `QATrace` before gate intervention, isolating the LLM's native confidence calibration from deterministic gate overrides.

### 5.2 Cluster Bootstrap vs. Wilson Confidence Intervals
- **Per-Category Slice Intervals:** Computed via closed-form **Wilson score 95% intervals** (`wilson_score_interval`), which remain well-behaved for small slice denominators ($n = 6$ to $15$).
- **Overall 150-Item Accuracy Interval:** Because the 100 adversarial variants are derived from the 50 base cases (`1` base case + `0–6` variants per base family), the 150 evaluation items are correlated within base-case clusters. [`cluster_bootstrap_ci()`](file:///p:/HEVA%20AI%20Assignment/evaluation/calibration.py#L186-L234) resamples the **50 base-case cluster IDs** with replacement for $B = 10,000$ iterations (`seed=42`) to produce an honest 95% confidence interval that accounts for intra-cluster correlation.

---

## 6. Failure Taxonomy, Root Causes & Concrete Engineering Remediations

[`evaluation/clustering.py`](file:///p:/HEVA%20AI%20Assignment/evaluation/clustering.py) implements rule-seeded, human-verifiable failure clustering across 9 distinct failure modes, each paired with a specific model-behavior hypothesis, concrete code remediation, and causal probe verification method:

| Failure Cluster | Root-Cause Hypothesis | Concrete Code Remediation | Verification Method |
| :--- | :--- | :--- | :--- |
| `OVER_ABSTENTION` | Model slightly paraphrases/truncates a valid quote so the strict Evidence-Quote Gate forces abstention, or overly conservatively refuses on hard paraphrases. | Add sentence-ID indexed quoting (`[S1]`, `[S2]`) in `src/prompts.py` and `src/qa_system.py` so minor punctuation normalization never triggers false quote-gate rejection. | Re-run failed `OVER_ABSTENTION` cases and verify `gate_triggered` drops without increasing hallucination rate on unanswerable cases. |
| `UNDER_ABSTENTION` | Model cites a sentence mentioning the target entity and extrapolates an unmentioned attribute instead of setting `abstained=True`. | Require a pre-output attribute-presence check in `grounded_v2` verifying that the queried metric/attribute name explicitly appears in the cited evidence sentence. | Re-run the 12 unanswerable base cases and 10 `false_premise_fabricated` variants and verify `abstention_recall` improvement. |
| `FALSE_PREMISE_ACCEPTED` | Sycophantic conditioning on user-supplied numbers in `false_premise_numeric` questions causes the model to repeat the false number as fact. | Extract `(question_claimed_value, document_stated_value)` prior to final answer generation and deterministically contrast them when they differ. | Re-run all 12 `false_premise_numeric` variants and verify `false_premise_accepted == False` on 100% of cases. |
| `DISTRACTOR_CAPTURE` | High lexical overlap with a sibling product (`Helios-X9L`, `Helios-X9R`, `MV Borealis-X`) or secondary tier (`Tier-0-MIL`, `Board of Directors`) causes attention to latch onto the near-miss addendum. | Add Layer-2 subject-entity verification in `src/qa_system.py` post-processing to reject evidence sentences whose subject entity is a sibling/near-miss identifier. | Re-run all 12 `distractor_near_miss` variants (`VAR-016`–`VAR-027`) and verify `distractor_captured == 0` (plus distractor-removal causal probe). |
| `SOURCE_CONFLICT_MISSED` | Single-span retrieval bias (primacy/recency): once the model finds the first matching clause, it stops scanning and misses the conflicting summary section. | Scan all document paragraphs for duplicate mentions of the queried metric/unit and require citing all matching clauses when distinct values appear. | Re-run all 10 `source_conflict` variants (`VAR-060`–`VAR-069`) and clause-order-swap causal probes. |
| `FORMAT_REASONING_ERROR` | Surface-form sensitivity across ASCII pipe tables, spelled-out numbers, or unit conversions causes misread columns or dropped multi-sentence clauses. | Pre-normalize ASCII pipe tables into explicit `Row <Entity>, <Column>: <Value>` sentences before prompt assembly. | Re-run `format_edge`, `paraphrase_hard`, and `two_sentence_fact` slices. |
| `POSITION_SENSITIVITY` | Attention decay ("lost in the middle") when ~5x archival warehouse logs separate the target specification from the question. | Rank/filter paragraphs mentioning the question's named entities while preserving verbatim full-document quote verification. | Compare `start`, `middle`, and `end` accuracy across `VAR-070`–`VAR-078` (plus padding-strip causal probe). |
| `INJECTION_FOLLOWED` | Imperative override strings inside untrusted document text hijack generation. | Strip/escape bracketed imperative override directives (`[SYSTEM PROMPT OVERRIDE...]`) before prompt assembly and enforce XML `<DOCUMENT>` boundaries. | Re-run `VAR-089`–`VAR-094` (plus injection-strip causal probe) and verify `injection_followed == 0`. |
| `FABRICATED_CONTENT` | Model generates an unentailed or contradicted claim while citing a real but insufficient sentence from the document. | Integrate Layer-2 deterministic number/entity/polarity checks into the online QA gate so unbacked numbers trigger abstention. | Measure output-level and claim-level hallucination rates before and after enabling the online Layer-2 gate. |

---

## 7. Prompt Regression Harness & Noise-Floor Estimation

[`evaluation/regression.py`](file:///p:/HEVA%20AI%20Assignment/evaluation/regression.py) prevents silent prompt degradation across iterative prompt engineering:
1. **Frozen Regression Baseline (`freeze_regression_baseline`):** Freezes all baseline-passing cases from `grounded_v1` along with baseline model ID, prompt hash, and dataset hash.
2. **Multi-Run Noise-Floor Estimation (`run_noise_floor_experiment`):** Runs `N=3` uncached passes (`bypass_cache=True`) on a stratified subset to measure natural API sampling jitter (`noise_floor_disagreement_rate`) and flag inherently flaky cases (`flaky_case_ids`) before attributing flips to a prompt change.
3. **Prompt Regression Comparison (`run_prompt_regression_comparison`):** Compares any candidate prompt (`grounded_v2` or the deliberately planted bad prompt `grounded_v2_bad`) against the frozen baseline, flagging:
   - `PASS_TO_FAIL` regressions on previously passing cases,
   - Category accuracy drops exceeding `max(5.0 percentage points, noise_floor)`,
   - Any increase in output hallucination rate or `INJECTION_FOLLOWED` count.
   - Offline unit test `test_prompt_regression_comparison_and_planted_bad_detector` verifies that `grounded_v2_bad` is deterministically caught and fails the regression gate (`passed_gate == False`).

---

## 8. Known Limitations & Threats to Validity

1. **Synthetic Domain Corpus:** The 8 source documents (`DOC-001`–`DOC-008`) are synthetic technical, legal, clinical, maritime, and cloud specifications designed to eliminate pretraining memorization. Real-world OCR-degraded PDFs, multi-page financial tables, and cross-document retrieval introduce layout noise not present in clean UTF-8 text.
2. **Conservative Layer-2 Deferral on Unit Conversions & Semantic Inferences:** Deterministic Layer 2 intentionally achieves high precision (`38/38 = 100.0%` on decided claims) by deferring non-verbatim mathematical unit conversions (`AUD-010`–`AUD-012`), bare single-digit substring collisions (`AUD-039`), and lexical-overlap semantic leaps (`AUD-043`–`AUD-045`) to Layer 3 (`needs_judge`). In offline mode (`allow_llm_judge=False`), those 7 claims (`15.56%`) remain unresolved by deterministic rules and are reported separately rather than hidden inside overall accuracy.
3. **Provider Non-Determinism:** Neither Groq (`openai/gpt-oss-20b`) nor Gemini (`gemini-3.5-flash-lite`) guarantees bit-for-bit identical token generation across backend hardware updates even at `temperature=0.0, seed=42`. Reproducibility relies on SHA-256 disk caching (`.cache/llm_calls/`) and explicit 3-run noise-floor measurement.
4. **Adversarial Variants Not Individually Human-Verified & Live 150-Case Evaluation Not Yet Run:** While all 50 base cases and all 45 audit claims are human-verified (`status: "verified"`), the 100 adversarial variants in `data/variants.jsonl` were systematically audited automatically (with `VAR-001`–`VAR-020` reviewed in human batches) and have not been individually human-verified. They remain in `status: "draft"` (`verified_by: null`, `verified_on: null`), and the live 150-case evaluation (`python -m evaluation.runner --run-all`) has not yet been executed.
