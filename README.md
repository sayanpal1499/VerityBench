# VerityBench — Adversarial Reliability & Hallucination Evaluation Harness for Grounded Document QA

VerityBench is a custom, zero-framework evaluation harness and adversarial benchmark for grounded Document Question Answering (QA). It is built from scratch in Python without external evaluation libraries (`RAGAS`, `TruLens`, `DeepEval`, or `LangChain`) to provide transparent, auditable, and statistically rigorous measurement of:

1. **Structured Grounded QA & Evidence-Quote Verification** (`src/qa_system.py`, `src/schemas.py`, `src/prompts.py`)
2. **Reproducible Multi-Provider LLM Caching & Telemetry** (`src/llm_client.py`)
3. **10-Category Adversarial Stress Testing (`100` Variants across `50` Base Cases; `144` LLM-Eligible + `6` Pre-API Malformed + `1` Reserved Category `multi_hop_conditional`)** (`evaluation/adversarial.py`, `evaluation/scoring.py`)
4. **3-Layer Programmatic Hallucination Detection & In-Sample Rule-Development Audit (`45` Claims)** (`evaluation/hallucination.py`)
5. **Dual-Track Confidence Calibration (Intrinsic Raw Model vs. Post-Gate System ECE & Brier Score on `N = 144` LLM-Evaluated Cases)** (`evaluation/calibration.py`)
6. **Failure Mode Clustering, Root-Cause Remediation & Causal Probes** (`evaluation/clustering.py`)
7. **Prompt-Regression Testing (`grounded_v2` vs. `grounded_v2_bad`) & 3-Run Noise-Floor Estimation** (`evaluation/regression.py`, `evaluation/runner.py`)

---

## 1. Architecture & Design Decisions

```
VerityBench/
├── data/
│   ├── documents/                         # 8 multi-paragraph domain source documents (DOC-001..DOC-008, 417-485 words each)
│   ├── base_cases.jsonl                   # 50 human-verified base QA cases (38 answerable, 12 unanswerable)
│   ├── regression_cases.jsonl             # 40 human-verified frozen regression baseline cases (BASE-001..BASE-040: 32 answerable, 8 unanswerable)
│   ├── variants.jsonl                     # 100 adversarial variants across 10 transformation categories (currently status="draft")
│   ├── variants_review.csv                # Companion tabular review export for all 100 variants
│   ├── variants_final_review_checklist.md # Consolidated batch-by-batch human-review checklist & audit log (VAR-001..VAR-100)
│   ├── audit_claims.jsonl                 # 45 human-verified in-sample rule-development claims for detector benchmarking
│   ├── audit_claims_review.csv            # Companion tabular review log for all 45 audit claims
│   └── audit_claims_pending_review.md     # Complete human-review rationale log for AUD-001..AUD-045
├── src/
│   ├── schemas.py                         # Strict Pydantic v2 schemas (QAOutput, QATrace, BaseCaseRecord, VariantRecord, RegressionCaseRecord)
│   ├── prompts.py                         # Versioned prompts (grounded_v1, naive_v0, grounded_v2, grounded_v2_bad)
│   ├── llm_client.py                      # Unified Groq + Gemini client, SHA-256 disk cache (.cache/llm_calls/), retry & pacing
│   └── qa_system.py                       # Input validation, JSON repair retry, deterministic Evidence-Quote Gate
├── evaluation/
│   ├── dataset.py                         # Dataset loaders, verbatim evidence checks & strict verification gates
│   ├── adversarial.py                     # Deterministic generator for all 100 adversarial variants
│   ├── scoring.py                         # Deterministic structured answer rubric + Layer-3 semantic judge fallback
│   ├── hallucination.py                   # 3-layer claim-level hallucination detector + Cohen's kappa audit
│   ├── calibration.py                     # 10-bin ECE, Brier score, reliability curves & Wilson/bootstrap CIs (N=144 LLM-eligible)
│   ├── clustering.py                      # Rule-seeded + human-verified failure clustering & CLI causal probes (--run-probes)
│   ├── regression.py                      # Frozen regression baseline, 3-run noise floor & grounded_v2 vs grounded_v2_bad CLI runner
│   ├── metrics.py                         # Aggregated slice metrics (overall_llm_evaluated_144 & overall_all_150) & Matplotlib charts
│   └── runner.py                          # Strict-gate evaluation orchestrator, CLI entry points & report populator
├── results/
│   ├── preflight_check.json               # Recorded live API preflight & 5 manual smoke-test inspections
│   └── audit_benchmark_offline.json       # Measured offline Layer-1/2 detector audit on all 45 in-sample audit claims
├── reports/
│   └── evaluation_report.md               # Comprehensive technical evaluation report (includes reserved multi_hop_conditional design)
└── tests/                                 # 53 offline unit & integration tests (zero external API calls)
```

### Key Architectural Choices & Reconciled Code Constants

1. **System Under Test (`QAOutput` + Deterministic Evidence-Quote Gate in [`src/qa_system.py`](src/qa_system.py)):**
   - **Pre-API Input Validation (`validate_inputs`):** Rejects empty, whitespace-only, punctuation-only (`< 2` alphanumeric characters), or overly long (`> MAX_QUESTION_CHARS = 2000` characters) questions by raising `InputValidationError` **before** any LLM API call is made.
   - **Structured Output Contract (`QAOutput`):** Every response from `answer_question()` in [`src/qa_system.py`](src/qa_system.py) is validated against the strict `QAOutput` Pydantic schema ([`src/schemas.py`](src/schemas.py): `answer: str`, `evidence: list[str]`, `confidence: float` in `[0.0, 1.0]`, `abstained: bool`). If JSON parsing fails after one repair retry (`seed + 1`) or an API error occurs (`raw_output is None`), `answer_question()` returns a system-error fallback with `abstained=True, evidence=[], confidence=0.0` (`gate_reason="error_fallback"`).
   - **Deterministic Evidence-Quote Gate (`verify_evidence_quote` & `apply_evidence_quote_gate`):** When the Evidence-Quote Gate is enabled (`grounded_v1`, `grounded_v2`), every cited quote in `evidence` must have at least **`MIN_QUOTE_CHARS = 5`** normalized characters and appear verbatim in the source document after conservative whitespace/quote/dash normalization (`normalize_text_for_quote_match`). Near-miss quotes with `difflib.SequenceMatcher` similarity ratio **`>= 0.90`** are logged as diagnostic telemetry (`match_type="fuzzy_candidate"` in `QATrace.quote_checks` and `QATrace.fuzzy_candidate_count`) but **never** accepted as valid evidence.
   - **Post-Gate Forced Abstention Confidence (`GATE_FALLBACK_CONFIDENCE = 0.50`):** If `raw_output.abstained == False` and no cited quote passes verification, `apply_evidence_quote_gate()` overrides the output to `abstained=True`, clears `evidence=[]`, sets post-gate **`confidence = GATE_FALLBACK_CONFIDENCE = 0.50`**, and preserves the unaltered `raw_output` in `QATrace` for intrinsic vs. post-gate calibration comparison.
2. **Why the Nominal 150-Case Benchmark Has 144 LLM-Eligible Cases (`6` `empty_malformed` Pre-API Exclusions):**
   - The full benchmark comprises **150 nominal test cases** (`50` base cases in [`data/base_cases.jsonl`](data/base_cases.jsonl) + `100` adversarial variants in [`data/variants.jsonl`](data/variants.jsonl)).
   - Exactly **6 variants (`VAR-095`–`VAR-100`)** belong to the `empty_malformed` category (`""`, `"   \n\t  "`, `"???!!!"`, `"?"`, a 2,100-character string exceeding `MAX_QUESTION_CHARS = 2000`, and `"\n\n---\n\n"`). These 6 cases are deterministically rejected pre-API by `validate_inputs()` (`InputValidationError`, recorded with `validation_rejected=True, abstained=True, confidence=1.0`, `0` LLM calls).
   - Consequently, **144 cases (`50` base + `94` non-empty variants)** are **LLM-eligible**. [`evaluation/metrics.py`](evaluation/metrics.py) reports both `overall_llm_evaluated_144` (`N = 144`) and `overall_all_150` (`N = 150`), and [`evaluation/calibration.py`](evaluation/calibration.py) computes ECE and Brier scores strictly over the **`N = 144` LLM-evaluated cases** (`excluded_pre_api_validation_records_n = 6`), because pre-API input validation rejections do not have a model-generated probability.
3. **Separation of SUT and Independent Judge Models:**
   - To eliminate self-preference bias, the System Under Test (SUT) uses `groq/openai/gpt-oss-20b` (`temperature=0.0, seed=42`), while semantic answer judging and Layer-3 claim entailment use an independent provider and model family: `gemini/gemini-3.5-flash-lite` (`temperature=0.0, seed=42`).
4. **Honest Reproducibility & Caching (`src/llm_client.py`):**
   - Hosted inference APIs (Groq and Gemini) do not guarantee bit-for-bit determinism across hardware nodes even at `temperature=0.0` and `seed=42`.
   - [`src/llm_client.py`](src/llm_client.py) enforces **replay determinism** via a SHA-256 disk cache (`.cache/llm_calls/<sha256>.json`) keyed on `(provider, model, system_prompt, prompt, response_schema, temperature, seed, max_output_tokens, json_mode, prompt_version)` and logs `system_fingerprint`, `parameters_requested`, `parameters_supported`, and `parameters_unsupported` on every call.
5. **3-Layer Programmatic Hallucination Detector (`evaluation/hallucination.py`):**
   - **Layer 1 (Deterministic Quote Presence):** Verifies whether each cited evidence string exists verbatim in the supplied document.
   - **Layer 2 (Deterministic Fact/Entity/Number/Polarity Check):** Decomposes `output.answer` into atomic claims and checks numeric match, unit-conversion equivalence, negation polarity, and sibling-entity mismatch against the document. Layer 2 returns a 4-state verdict (`supported`, `contradicted`, `unsupported`, or `needs_judge`) so ambiguous semantic entailments are **deferred** rather than guessed.
   - **Layer 3 (Independent LLM Claim Judge):** Invoked only when Layer 2 returns `needs_judge` and `allow_llm_judge=True`. When `allow_llm_judge=False` (offline mode), `needs_judge` claims are tracked and reported separately so conservative fallback mappings (`needs_judge -> unsupported`) are never misrepresented as confirmed deterministic detector accuracy.
6. **Dual-Track Confidence Calibration (`evaluation/calibration.py`):**
   - Because the Evidence-Quote Gate rewrites `confidence = GATE_FALLBACK_CONFIDENCE = 0.50` and `abstained=True` when a quote fails verification, computing Expected Calibration Error (ECE) only on post-gate outputs conflates model calibration with deterministic gate intervention. VerityBench computes 10-bin ECE and Brier scores separately for **Post-Gate System Calibration**, **Intrinsic Raw-Model Calibration**, and **Non-Gate-Overridden Calibration** across the `N = 144` LLM-evaluated records, further decomposed into answered-only and abstained-only subsets.

---

## 2. Current Dataset Verification Status & Strict Governance Gate

VerityBench enforces a hard verification gate ([`verify_all_records_before_run`](evaluation/runner.py)): live full-suite evaluation (`python -m evaluation.runner --run-all`) refuses to run if any record in [`data/base_cases.jsonl`](data/base_cases.jsonl), [`data/variants.jsonl`](data/variants.jsonl), or [`data/audit_claims.jsonl`](data/audit_claims.jsonl) has `status != "verified"`, missing `verified_by`/`verified_on`, or `needs_human_review == True`. Per project governance rules, draft records are **never** auto-promoted to `"verified"` without individual human verification.

| Dataset File | Total Records | Current Status | Verification Metadata | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `data/documents/DOC-001..008.txt` | `8` documents | **Complete** | Canonical synthetic domain corpus | 8 technical/legal/clinical/maritime/cloud specs (`417–485` words each; zero pretraining overlap). |
| [`data/base_cases.jsonl`](data/base_cases.jsonl) | `50` cases | **Verified (`50/50`)** | `verified_by="sayanpal1499"`, `verified_on="2026-10-12"` | `38` answerable (`number_with_unit`, `named_entity`, `date`, `yes_no`, `list`, `two_sentence_fact`) + `12` unanswerable (`should_abstain=True`). |
| [`data/regression_cases.jsonl`](data/regression_cases.jsonl) | `40` cases | **Verified (`40/40`)** | `verified_by="sayanpal1499"`, `verified_on="2026-10-12"` | Frozen regression baseline (`BASE-001`–`BASE-040`: `32` answerable + `8` unanswerable across `DOC-001`–`DOC-008`), validated via `validate_regression_cases()`. |
| [`data/audit_claims.jsonl`](data/audit_claims.jsonl) | `45` claims | **Verified (`45/45`)** | `verified_by="sayanpal1499"`, `verified_on="2026-10-13"` | Non-independent, in-sample rule-development audit set (`15 supported`, `15 contradicted`, `15 unsupported`; `0` `needs_human_review` flags remaining). |
| [`data/variants.jsonl`](data/variants.jsonl) | `100` variants | **Draft (`100/100`)** | `status="draft"`, `verified_by=null`, `verified_on=null` | **Exact Live-Run Prerequisite:** `verify_all_records_before_run()` blocks `--run-all` until all `100` records have `status="verified"` with non-null `verified_by` and `verified_on`. `VAR-001`–`VAR-020` were reviewed in human batches 1–2, and all `100` variants (`VAR-001`–`VAR-100`) were systematically audited in one pass and consolidated in [`data/variants_final_review_checklist.md`](data/variants_final_review_checklist.md). |

---

## 3. Installation & Configuration

### Prerequisites
- Python `3.11+` (tested on Python `3.12.10` on Windows)
- Virtual environment at `.venv`

### Setup (PowerShell)
```powershell
# 1. Create and activate virtual environment (if not already present)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Configure API keys for live runs (never committed to git)
Copy-Item .env.example .env
# Edit .env to set GROQ_API_KEY and GEMINI_API_KEY
```

### Setup (Bash)
```bash
# 1. Create and activate virtual environment (if not already present)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Configure API keys for live runs (never committed to git)
cp .env.example .env
# Edit .env to set GROQ_API_KEY and GEMINI_API_KEY
```

---

## 4. Reproducible Commands (PowerShell & Bash)

### A. Offline Test Suite & Dataset Validators (Zero External API Calls)
All `53` unit and integration tests run completely offline using deterministic fixtures and monkeypatched fake-LLM mocks:

**PowerShell:**
```powershell
# 1. Run the complete 53-test offline suite
.\.venv\Scripts\python.exe -m pytest -v

# 2. Validate all datasets (50 verified base cases, 40 verified regression cases, 45 verified audit claims, 100 draft variants)
.\.venv\Scripts\python.exe -m evaluation.dataset --allow-draft --check-variants --check-audit-claims --check-regression-cases

# 3. Validate only the frozen 40-case regression baseline (strict verified mode)
.\.venv\Scripts\python.exe -m evaluation.regression --validate-only

# 4. Run offline Layer-1 + Layer-2 hallucination detector benchmark on the 45 in-sample audit claims
$env:PYTHONPATH="."; .\.venv\Scripts\python.exe -c "
import json
from evaluation.dataset import load_documents, load_jsonl_audit_claims
from evaluation.hallucination import run_audit_benchmark
res = run_audit_benchmark(load_jsonl_audit_claims(), load_documents(), allow_llm_judge=False)
print(json.dumps({k: v for k, v in res.items() if k != 'evaluated_records'}, indent=2))
"

# 5. Regenerate reports/evaluation_report.md from saved artifacts (includes reserved multi_hop_conditional section)
.\.venv\Scripts\python.exe -m evaluation.runner --build-report
```

**Bash:**
```bash
# 1. Run the complete 53-test offline suite
./.venv/bin/python -m pytest -v

# 2. Validate all datasets (50 verified base cases, 40 verified regression cases, 45 verified audit claims, 100 draft variants)
./.venv/bin/python -m evaluation.dataset --allow-draft --check-variants --check-audit-claims --check-regression-cases

# 3. Validate only the frozen 40-case regression baseline (strict verified mode)
./.venv/bin/python -m evaluation.regression --validate-only

# 4. Run offline Layer-1 + Layer-2 hallucination detector benchmark on the 45 in-sample audit claims
PYTHONPATH=. ./.venv/bin/python -c '
import json
from evaluation.dataset import load_documents, load_jsonl_audit_claims
from evaluation.hallucination import run_audit_benchmark
res = run_audit_benchmark(load_jsonl_audit_claims(), load_documents(), allow_llm_judge=False)
print(json.dumps({k: v for k, v in res.items() if k != "evaluated_records"}, indent=2))
'

# 5. Regenerate reports/evaluation_report.md from saved artifacts (includes reserved multi_hop_conditional section)
./.venv/bin/python -m evaluation.runner --build-report
```

### B. Live Evaluation, Causal Probes & Regression CLI Workflows (Requires Promoted `variants.jsonl` + API Keys)
Once human sign-off authorizes promoting `data/variants.jsonl` from `"draft"` to `"verified"` (and when live API execution is authorized):

**PowerShell:**
```powershell
# 1. Verify the strict ground-truth governance gate across all 195 records (50 base + 100 variants + 45 audit claims)
.\.venv\Scripts\python.exe -m evaluation.runner --check-gate

# 2. Execute full live evaluation (grounded_v1 + naive_v0 ablation + calibration on N=144 + clustering + figures)
.\.venv\Scripts\python.exe -m evaluation.runner --run-all

# 3. Execute counterfactual causal intervention probes on observed failure clusters (writes results/causal_probes.json)
.\.venv\Scripts\python.exe -m evaluation.runner --run-probes

# 4. Execute 3-run uncached noise-floor estimation and grounded_v2 vs grounded_v2_bad prompt regression comparison
.\.venv\Scripts\python.exe -m evaluation.runner --run-noise-floor --run-regression

# 5. Populate reports/evaluation_report.md strictly from saved artifacts in results/
.\.venv\Scripts\python.exe -m evaluation.runner --build-report
```

**Bash:**
```bash
# 1. Verify the strict ground-truth governance gate across all 195 records (50 base + 100 variants + 45 audit claims)
./.venv/bin/python -m evaluation.runner --check-gate

# 2. Execute full live evaluation (grounded_v1 + naive_v0 ablation + calibration on N=144 + clustering + figures)
./.venv/bin/python -m evaluation.runner --run-all

# 3. Execute counterfactual causal intervention probes on observed failure clusters (writes results/causal_probes.json)
./.venv/bin/python -m evaluation.runner --run-probes

# 4. Execute 3-run uncached noise-floor estimation and grounded_v2 vs grounded_v2_bad prompt regression comparison
./.venv/bin/python -m evaluation.runner --run-noise-floor --run-regression

# 5. Populate reports/evaluation_report.md strictly from saved artifacts in results/
./.venv/bin/python -m evaluation.runner --build-report
```

---

## 5. Summary of Offline Measured Results vs. Simulated Dry Runs vs. Live Evaluation Readiness

1. **Offline Test Suite (`pytest -v` — Simulated / Mocked Dry Runs):**
   - **53 / 53 passing (`100%`)** in `~2.7s` with zero network calls.
   - Includes mocked integration dry runs for `execute_verified_evaluation`, 3-run noise-floor estimation (`run_noise_floor_experiment`), `grounded_v2` vs. `grounded_v2_bad` head-to-head comparison (`compare_grounded_v2_vs_bad`), and counterfactual causal probes (`execute_causal_probes_from_saved_failures`). These fake-LLM unit-test dry runs validate code integration paths only and are **never** reported as live model accuracy.
2. **Offline Hallucination Detector Audit Benchmark (`45` Verified Human-Labeled Claims in `data/audit_claims.jsonl`, `allow_llm_judge=False` — Non-Independent In-Sample Rule-Development Set):**
   - **Important Scope Disclosure:** The 45 claims in `data/audit_claims.jsonl` were constructed and boundary-reviewed during Layer-2 deterministic rule development. Accordingly, the metrics below measure **in-sample rule consistency on the rule-development audit set**, not generalization to an independent blind test set.
   - **Layer-2 Resolved-Decision Precision (In-Sample):** **`38 / 38` (`100.0%`)** on the 38 claims where Layer 2 makes a definite deterministic decision (`0` false positives, `0` false negatives).
   - **Layer-2 Unmapped Deterministic Coverage:** **`38 / 45` (`84.44%`)** across all 45 claims.
   - **Layer-2 Unresolved / Deferred (`needs_judge`) Claims:** **`7 / 45` (`15.56%`)**, comprising:
     - `AUD-010`, `AUD-011`, `AUD-012` (`unit_conversion`, human label `supported`): mathematical unit conversions (`42 mA -> 0.042 A`, `36 hours -> 1.5 days`, `3.5-hour -> 210-minute`) that require semantic/unit-conversion entailment beyond surface substring matching.
     - `AUD-039` (`unanswerable_extrapolation`, human label `unsupported`): contains bare integer `"1"`, which collides with `"142"` in `DOC-003` and is conservatively deferred by Layer 2's bare-single-digit guard.
     - `AUD-043`, `AUD-044`, `AUD-045` (`unsupported_inference`, human label `unsupported`): semantic leaps (`up to 3 days -> every Mon/Wed/Fri`, `arbitration in Basel -> manufacturing in Basel`, `three technical remedial actions -> monetary fine`) that share vocabulary with the document and properly require Layer-3 NLI judging.
   - **Mapped 3-Class Agreement (with conservative `needs_judge -> unsupported` fallback):** **`42 / 45` (`93.33%`)**, Cohen's $\kappa = 0.9000$ (`95% bootstrap CI: [0.7667, 1.0000]`). Per governance rules, the 4 `unsupported` claims matched via fallback (`AUD-039`, `AUD-043`, `AUD-044`, `AUD-045`) are reported separately from the 38 confirmed deterministic decisions.
3. **Live API Preflight & Smoke Tests (`results/preflight_check.json` — Measured):**
   - Verified live connectivity, schema compliance, token/latency telemetry, and SHA-256 cache replay for `groq/openai/gpt-oss-20b` (SUT) and `gemini/gemini-3.5-flash-lite` (independent judge), plus 5 manual smoke-test scenarios (`EX-1` through `EX-5`).
4. **Full 150-Nominal / 144-LLM-Eligible Live Evaluation (`grounded_v1` vs. `naive_v0` — Not Yet Run):**
   - The live evaluation (`50` base cases + `94` LLM-eligible variants + `6` pre-API `empty_malformed` variants) has **not yet been run**. Because the 100 variants in `data/variants.jsonl` have not all been individually human-verified (`VAR-001`–`VAR-020` reviewed in human batches 1–2; `VAR-021`–`VAR-100` audited and consolidated in [`data/variants_final_review_checklist.md`](data/variants_final_review_checklist.md)), they remain in `status: "draft"`, and `verify_all_records_before_run()` strictly blocks live evaluation until human sign-off is completed. No live model accuracy, hallucination rates, calibration metrics, or observed failure cluster counts are claimed prior to running the live evaluation.

See [`reports/evaluation_report.md`](reports/evaluation_report.md) for the complete technical report, methodology, pre-run failure hypotheses, calibration equations, regression workflow, and the reserved 11th adversarial category (`multi_hop_conditional`).
