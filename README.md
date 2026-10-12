# VerityBench — Adversarial Reliability & Hallucination Evaluation Harness for Grounded Document QA

VerityBench is a custom, zero-framework evaluation harness and adversarial benchmark for grounded Document Question Answering (QA). It is built from scratch in Python without external evaluation libraries (`RAGAS`, `TruLens`, `DeepEval`, or `LangChain`) to provide transparent, auditable, and statistically rigorous measurement of:

1. **Structured Grounded QA & Evidence-Quote Verification** (`src/qa_system.py`, `src/schemas.py`, `src/prompts.py`)
2. **Reproducible Multi-Provider LLM Caching & Telemetry** (`src/llm_client.py`)
3. **10-Category Adversarial Stress Testing (`100` Variants across `50` Base Cases)** (`evaluation/adversarial.py`, `evaluation/scoring.py`)
4. **3-Layer Programmatic Hallucination Detection & Human Audit Benchmark (`45` Claims)** (`evaluation/hallucination.py`)
5. **Dual-Track Confidence Calibration (Intrinsic Raw Model vs. Post-Gate System ECE & Brier Score)** (`evaluation/calibration.py`)
6. **Failure Mode Clustering, Root-Cause Remediation & Causal Probes** (`evaluation/clustering.py`)
7. **Prompt-Regression Testing & Noise-Floor Estimation** (`evaluation/regression.py`, `evaluation/runner.py`)

---

## 1. Architecture & Design Decisions

```
VerityBench/
├── data/
│   ├── documents/                         # 8 multi-paragraph domain source documents (DOC-001..DOC-008)
│   ├── base_cases.jsonl                   # 50 human-verified base QA cases (38 answerable, 12 unanswerable)
│   ├── variants.jsonl                     # 100 adversarial variants across 10 transformation categories
│   ├── variants_review.csv                # Companion tabular review export for all 100 variants
│   ├── variants_final_review_checklist.md # Batch-by-batch human-review checklist & audit log
│   ├── audit_claims.jsonl                 # 45 human-verified claims for detector agreement benchmarking
│   ├── audit_claims_review.csv            # Companion tabular review log for all 45 audit claims
│   └── audit_claims_pending_review.md     # Complete human-review rationale log for AUD-001..AUD-045
├── src/
│   ├── schemas.py                         # Strict Pydantic v2 schemas (QAOutput, QATrace, dataset records)
│   ├── prompts.py                         # Versioned prompts (grounded_v1, naive_v0, grounded_v2, grounded_v2_bad)
│   ├── llm_client.py                      # Unified Groq + Gemini client, SHA-256 disk cache, retry & pacing
│   └── qa_system.py                       # Input validation, JSON repair retry, deterministic Evidence-Quote Gate
├── evaluation/
│   ├── dataset.py                         # Dataset loaders, verbatim evidence checks & strict verification gates
│   ├── adversarial.py                     # Deterministic generator for all 100 adversarial variants
│   ├── scoring.py                         # Deterministic structured answer rubric + Layer-3 semantic judge fallback
│   ├── hallucination.py                   # 3-layer claim-level hallucination detector + Cohen's kappa audit
│   ├── calibration.py                     # 10-bin ECE, Brier score, reliability curves & Wilson/bootstrap CIs
│   ├── clustering.py                      # Rule-seeded + human-verified failure clustering & causal probes
│   ├── regression.py                      # Frozen regression baseline, 3-run noise floor & prompt diff runner
│   ├── metrics.py                         # Aggregated slice metrics, ablation deltas & Matplotlib chart builders
│   └── runner.py                          # Strict-gate evaluation orchestrator & report populator
├── results/
│   ├── preflight_check.json               # Recorded live API preflight & 5 manual smoke-test inspections
│   └── audit_benchmark_offline.json       # Measured offline Layer-1/2 detector audit on all 45 verified claims
├── reports/
│   └── evaluation_report.md               # Comprehensive technical evaluation report
└── tests/                                 # 49 offline unit & integration tests (zero external API calls)
```

### Key Architectural Choices

1. **System Under Test (`QAOutput` + Deterministic Evidence-Quote Gate):**
   - Every response from `answer_question()` in [`src/qa_system.py`](src/qa_system.py) is validated against the strict `QAOutput` Pydantic schema ([`src/schemas.py`](src/schemas.py): `answer: str`, `evidence: list[str]`, `confidence: float` in `[0.0, 1.0]`, `abstained: bool`).
   - When the Evidence-Quote Gate is enabled (`grounded_v1`, `grounded_v2`), every cited quote in `evidence` must appear verbatim in the source document after conservative whitespace/quote normalization (`normalize_text_for_quote_match`, minimum 15 characters). Fuzzy matches (`SequenceMatcher >= 0.85`) are logged as diagnostic telemetry (`fuzzy_candidate_quotes`) in `QATrace` but **never** accepted as valid evidence. If any cited quote fails verification, the gate overrides the output to `abstained=True`, sets post-gate `confidence=0.0`, and preserves the raw model output (`raw_output`) in `QATrace` for pre- vs. post-gate calibration comparison.
2. **Separation of SUT and Independent Judge Models:**
   - To eliminate self-preference bias, the System Under Test (SUT) uses `groq/openai/gpt-oss-20b` (`temperature=0.0, seed=42`), while semantic answer judging and Layer-3 claim entailment use an independent provider and model family: `gemini/gemini-3.5-flash-lite` (`temperature=0.0, seed=42`).
3. **Honest Reproducibility & Caching (`src/llm_client.py`):**
   - Hosted inference APIs (Groq and Gemini) do not guarantee bit-for-bit determinism across hardware nodes even at `temperature=0.0` and `seed=42`.
   - [`src/llm_client.py`](src/llm_client.py) enforces **replay determinism** via a SHA-256 disk cache (`.cache/llm_calls/<sha256>.json`) keyed on `(provider, model, system_prompt, prompt, response_schema, temperature, seed, max_output_tokens, json_mode, prompt_version)` and logs `system_fingerprint`, `parameters_requested`, `parameters_supported`, and `parameters_unsupported` on every call.
4. **3-Layer Programmatic Hallucination Detector (`evaluation/hallucination.py`):**
   - **Layer 1 (Deterministic Quote Presence):** Verifies whether each cited evidence string exists verbatim in the supplied document.
   - **Layer 2 (Deterministic Fact/Entity/Number/Polarity Check):** Decomposes `output.answer` into atomic claims and checks numeric match, unit-conversion equivalence, negation polarity, and sibling-entity mismatch against the document. Crucially, Layer 2 returns a 4-state verdict (`supported`, `contradicted`, `unsupported`, or `needs_judge`) so ambiguous semantic entailments are **deferred** rather than guessed.
   - **Layer 3 (Independent LLM Claim Judge):** Invoked only when Layer 2 returns `needs_judge` and `allow_llm_judge=True`. When `allow_llm_judge=False` (offline mode), `needs_judge` claims are tracked and reported separately so conservative fallback mappings (`needs_judge -> unsupported`) are never misrepresented as confirmed deterministic detector accuracy.
5. **Dual-Track Confidence Calibration (`evaluation/calibration.py`):**
   - Because the Evidence-Quote Gate rewrites `confidence=0.0` and `abstained=True` when a quote fails verification, computing Expected Calibration Error (ECE) only on post-gate outputs conflates model calibration with deterministic gate intervention. VerityBench computes 10-bin ECE and Brier scores separately for **Post-Gate System Calibration** and **Intrinsic Raw-Model Calibration**, further decomposed across overall (`N=150`), answered-only, and abstained-only subsets.

---

## 2. Current Dataset Verification Status & Strict Governance Gate

VerityBench enforces a hard verification gate ([`verify_all_records_before_run`](evaluation/runner.py)): live full-suite evaluation refuses to run if any record in [`data/base_cases.jsonl`](data/base_cases.jsonl), [`data/variants.jsonl`](data/variants.jsonl), or [`data/audit_claims.jsonl`](data/audit_claims.jsonl) has `status != "verified"` or `needs_human_review == True`. Per project governance rules, draft records are **never** auto-promoted to `"verified"` without individual human verification.

| Dataset File | Total Records | Current Status | Verification Metadata | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `data/documents/DOC-001..008.txt` | `8` documents | **Complete** | Canonical synthetic domain corpus | 8 technical/legal/clinical/maritime/cloud specs (`450–515` words each; zero pretraining overlap). |
| [`data/base_cases.jsonl`](data/base_cases.jsonl) | `50` cases | **Verified (`50/50`)** | `verified_by="sayanpal1499"`, `verified_on="2026-10-12"` | `38` answerable (`number_with_unit`, `named_entity`, `date`, `yes_no`, `list`, `two_sentence_fact`) + `12` unanswerable (`should_abstain=True`). |
| [`data/audit_claims.jsonl`](data/audit_claims.jsonl) | `45` claims | **Verified (`45/45`)** | `verified_by="sayanpal1499"`, `verified_on="2026-10-13"` | `15 supported`, `15 contradicted`, `15 unsupported`; `0` `needs_human_review` flags remaining after boundary review approval. |
| [`data/variants.jsonl`](data/variants.jsonl) | `100` variants | **Draft (`100/100`)** | `status="draft"`, `verified_by=null`, `verified_on=null` | All 100 variants (`VAR-001`–`VAR-100`) were systematically audited automatically across all 10 adversarial categories (with `VAR-001`–`VAR-020` also reviewed in human batches), but the full 100-variant set has **not** been individually human-verified and remains in `"draft"`. |

---

## 3. Installation & Configuration

### Prerequisites
- Python `3.11+` (tested on Python `3.12.10` on Windows)
- Virtual environment at `.venv`

### Setup
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

---

## 4. Reproducible Commands

### A. Offline Test Suite & Dataset Validators (Zero External API Calls)
All `49` unit and integration tests run completely offline using deterministic fixtures and monkeypatched mocks:

```powershell
# Run the complete 49-test offline suite
.\.venv\Scripts\python.exe -m pytest -v
```

```powershell
# Run dataset validation (50 verified base cases, 45 verified audit claims, 100 audited draft variants)
$env:PYTHONPATH="."; .\.venv\Scripts\python.exe -c "
from evaluation.dataset import (
    load_documents, load_jsonl_base_cases, load_jsonl_variants, load_jsonl_audit_claims,
    validate_base_cases, validate_variants, validate_audit_claims
)
docs = load_documents()
bases = load_jsonl_base_cases()
base_map = {b.id: b for b in bases}
variants = load_jsonl_variants()
claims = load_jsonl_audit_claims()
print('Base cases:', validate_base_cases(bases, docs, require_verified=True))
print('Audit claims:', validate_audit_claims(claims, base_map, docs, require_verified=True))
print('Variants (draft check):', validate_variants(variants, base_map, docs, require_verified=False))
"
```

```powershell
# Run offline Layer-1 + Layer-2 hallucination detector benchmark on all 45 verified audit claims
$env:PYTHONPATH="."; .\.venv\Scripts\python.exe -c "
import json
from evaluation.dataset import load_documents, load_jsonl_audit_claims
from evaluation.hallucination import run_audit_benchmark
res = run_audit_benchmark(load_jsonl_audit_claims(), load_documents(), allow_llm_judge=False)
print(json.dumps({k: v for k, v in res.items() if k != 'evaluated_records'}, indent=2))
"
```

### B. Live Evaluation & Report Generation (Requires Promoted `variants.jsonl` + API Keys)
Once human sign-off authorizes promoting `data/variants.jsonl` from `"draft"` to `"verified"`:

```powershell
# 1. Verify the strict ground-truth governance gate across all 195 records (50 base + 100 variants + 45 audit claims)
.\.venv\Scripts\python.exe -m evaluation.runner --check-gate

# 2. Execute full evaluation (grounded_v1 + naive_v0 ablation + calibration + clustering + figures)
.\.venv\Scripts\python.exe -m evaluation.runner --run-all

# 3. Populate reports/evaluation_report.md strictly from saved artifacts in results/metrics.json
.\.venv\Scripts\python.exe -m evaluation.runner --build-report
```

---

## 5. Summary of Offline Measured Results vs. Live Evaluation Readiness

1. **Offline Test Suite (`pytest -v`):** **49 / 49 passing (`100%`)** in `~2.6s` with zero network calls.
2. **Offline Hallucination Detector Audit Benchmark (`45` Verified Human-Labeled Claims, `allow_llm_judge=False`):**
   - **Confirmed Layer-2 Deterministic Accuracy (Unmapped):** **`38 / 45` (`84.44%`)** across all 45 claims, with **`38 / 38` (`100.0%`) precision** on claims where Layer 2 makes a definite decision (`0` false positives, `0` false negatives).
   - **Layer-2 Deferred (`needs_judge`) Rate:** **`7 / 45` (`15.56%`)**, comprising:
     - `AUD-010`, `AUD-011`, `AUD-012` (`unit_conversion`, human label `supported`): exact mathematical unit conversions (`42 mA -> 0.042 A`, `36 hours -> 1.5 days`, `3.5-hour -> 210-minute`) that require semantic/unit-conversion entailment beyond surface substring matching.
     - `AUD-039` (`unanswerable_extrapolation`, human label `unsupported`): contains bare integer `"1"`, which collides with `"142"` in `DOC-003` and is conservatively deferred by Layer 2's bare-single-digit guard.
     - `AUD-043`, `AUD-044`, `AUD-045` (`unsupported_inference`, human label `unsupported`): semantic leaps (`up to 3 days -> every Mon/Wed/Fri`, `arbitration in Basel -> manufacturing in Basel`, `three technical remedial actions -> monetary fine`) that share vocabulary with the document and properly require Layer-3 NLI judging.
   - **Mapped 3-Class Agreement (with conservative `needs_judge -> unsupported` fallback):** **`42 / 45` (`93.33%`)**, Cohen's $\kappa = 0.9000$ (`95% bootstrap CI: [0.7667, 1.0000]`). Per governance rules, the 4 `unsupported` claims matched via fallback (`AUD-039`, `AUD-043`, `AUD-044`, `AUD-045`) are reported separately from the 38 confirmed deterministic decisions.
3. **Live API Preflight & Smoke Tests (`results/preflight_check.json`):**
   - Verified live connectivity, schema compliance, token/latency telemetry, and SHA-256 cache replay for `groq/openai/gpt-oss-20b` (SUT) and `gemini/gemini-3.5-flash-lite` (independent judge), plus 5 manual smoke-test scenarios (`EX-1` through `EX-5`).
4. **Full 150-Case Live Evaluation (`grounded_v1` vs. `naive_v0` — Not Yet Run):**
   - The live 150-case evaluation (`50` base cases + `100` adversarial variants) has **not yet been run**. Because the 100 variants in `data/variants.jsonl` were systematically audited automatically rather than individually human-verified, they remain in `status: "draft"`, and `verify_all_records_before_run()` strictly blocks live evaluation until human verification is completed. No live model accuracy, hallucination rates, calibration metrics, or observed failure cluster counts are claimed prior to running the live evaluation.

See [`reports/evaluation_report.md`](reports/evaluation_report.md) for the complete technical report, methodology, failure taxonomy, calibration equations, and known limitations.
