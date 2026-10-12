# Adversarial Evaluation Harness — Implementation Plan (`PLAN.md`)

**Date Prepared / Revised:** 2026-10-12  
**Status:** CHECKPOINT 1 Approved & `RUN-00-PREFLIGHT` Gate Passed — Now at CHECKPOINT 2 (Human Verification of 50 Base Cases)

---

## 1. Source Document Analysis, Conflicts, Differences, and Resolutions

### 1.1 Source Files Inventory
1. **Official Assignment Instructions:** `HevaAI AI-ML Engineer Assignment 3.md` (confirmed by human as identical in content to `Heva AI-ML Engineer Assignment 3.pdf`).
2. **Implementation Specification & Build Guide:** `Antigravity_Build_Guide_FINAL.md` (authoritative build plan that supersedes the earlier draft specification `HevaAI_Adversarial_Evaluation_Harness_Project_Specification.md`).

### 1.2 Conflicts, Differences, and Ambiguities Across Sources

| # | Dimension | Official Assignment (`HevaAI AI-ML Engineer Assignment 3.md`) | Earlier Spec (Referenced in Build Guide §2) | Authoritative Resolution (`Antigravity_Build_Guide_FINAL.md`) | Rationale |
|---|---|---|---|---|---|
| 1 | **Deliverable Form Factor** | Requires a working LLM system + evaluation harness + report + README; no UI required. | Proposed a React + FastAPI dashboard as a core deliverable. | **Python CLI harness first** (`Phases 0–9`). Optional FastAPI UI (`Phase 10`) only after Phase 9 is 100% complete, timeboxed to 4 hours. | Grading evaluates the rigor of the harness, dataset, detectors, and report—not frontend UI polish. |
| 2 | **Ground-Truth Dataset Volume & Split** | Requires *"at least 50 test cases with known correct outputs"* + adversarial input generation. | Split 50 total cases into 10 clean + 40 adversarial variants (tiny per-category $n$). | Distinguish the **official minimum requirement ($\ge 50$ cases)** from our **design target**: **50 human-verified non-adversarial base cases** (~38 answerable, ~12 unanswerable-by-design) + **100 human-reviewed adversarial variants** = **150 total records**. | Enables paired base-vs-variant causal comparisons and base-case-cluster bootstrap confidence intervals while keeping base accuracy meaningful. Human verification quality takes strict priority over variant volume. |
| 3 | **Document Processing Architecture** | Open choice of LLM task system with at least one non-obvious design decision. | Proposed document chunking + vector retrieval (RAG). | **Full-document context (no chunking/retrieval).** Pass the entire `.txt` document in the prompt; evaluate long documents via the `long_context` adversarial category. | Eliminates retrieval chunk-miss as a confounding failure mode so every error reflects LLM grounding/reasoning behavior. |
| 4 | **Input File Formats** | "Accept text input and produce structured output." | Included both `.txt` and `.pdf` parsing in the evaluation loop. | **`.txt` only** for all evaluation runs (`data/documents/DOC-001.txt` ... `DOC-008.txt`). | Prevents PDF text-extraction artifacts from acting as unmeasured confounders. |
| 5 | **Counterfactual / Fictional Facts** | Requires detecting hallucinations against source material. | Allowed general domain text. | **Source documents must use fictional or counterfactual facts** (invented products, altered specs/policies). | Prevents the LLM from answering correctly via pretraining memory when it actually ignored or misread the document. |
| 6 | **Determinism ("deterministic given a fixed seed")** | Mandates that *"The system must be deterministic given a fixed seed, so that evaluation results are reproducible."* | Assumed passing `seed` to an API guarantees 100% bitwise determinism. | Strictly separate **cached replay reproducibility (100% guaranteed)** from **uncached API inference determinism (best-effort only)**. Pass `temperature=0.0` + `seed=42` where officially supported, log metadata, and measure **uncached variability** via a 3-run cache-bypass noise floor in Phase 8. | Hosted GPU inference is not bit-for-bit deterministic even with fixed `seed`; disk caching guarantees reproducible replay while the noise-floor test quantifies uncached API variance. |
| 7 | **Hallucination Detection & LLM Judge Bias** | Prohibits RAGAS/TruLens/DeepEval; permits disclosed LLM API calls for specific sub-tasks. | Used the same LLM to judge its own claims without deterministic pre-checks or human audit. | **3-layer detector** (Layer 1: quote presence; Layer 2: conservative deterministic entity/value/unit check; Layer 3: **cross-provider/different-model LLM claim judge**) audited against **40+ human-labeled claims** (`data/audit_claims.jsonl`) with Cohen's $\kappa$. | Avoids self-grading bias, minimizes judge API calls, and quantifies detector error modes. |
| 8 | **Failure Clustering & Root Cause Analysis** | Requires $\ge 3$ failure clusters explained at the **model behavior level**, plus 1 specific improvement per cluster. | Relied solely on heuristic rule labels describing surface outputs. | Rule-based first pass (`evaluation/clustering.py`) $\rightarrow$ **human review of every failure** (`results/failure_review.csv`) $\rightarrow$ **targeted causal probes** per cluster. If $< 3$ clusters appear initially, run targeted verified probes; if $< 3$ still occur, prominently flag the unmet requirement. | Causal probes empirically isolate *why* the model failed (e.g., near-miss entity attention vs. user-premise deference vs. parametric memory leakage). |
| 9 | **Interview Preparation (Extra Adversarial Category)** | States: *"We will ask you to design one more adversarial test category you did not include."* | Implemented all brainstormed categories in code. | Implement **10 adversarial categories** in `evaluation/adversarial.py`, and **hold back 1 reserved category on paper** (`multi_hop_conditional`) in Section 10 of `reports/evaluation_report.md`. | Directly prepares a rigorous, dimensionally consistent design for the live interview question. |

---

## 2. Checklist Mapping Official Requirements to Repository-Relative Paths

All documentation, reports, and scripts use portable **repository-relative paths** (never machine-specific absolute paths or `file://` URLs).

| Official Requirement (`HevaAI AI-ML Engineer Assignment 3.md`) | Build Guide Phase | Implementation File / Module | Automated Verification (`tests/`) | Data / Results Artifact | Report Section (`reports/evaluation_report.md`) |
|---|---|---|---|---|---|
| **R1. Working LLM-based system accepting text input & producing structured output** | Phase 0, Phase 1 | `src/llm_client.py` (`call_llm`), `src/schemas.py` (`QAOutput`), `src/prompts.py` (`naive_v0`, `grounded_v1`, `grounded_v2`), `src/qa_system.py` (`answer_question`) | `tests/test_llm_client.py`, `tests/test_qa_system.py` | `results/outputs_grounded_v1.jsonl`, `results/outputs_naive_v0.jsonl` | §1 Executive Summary, §2 System Architecture & Design Decision |
| **R2. Deliberate non-obvious design decision documented & evaluated vs. naive approach** | Phase 1, Phase 6 | `src/qa_system.py` (**Evidence-Quote Gate with Forced Abstention** + storage of both `raw_output` and `final_output`; ablation against `naive_v0`) | `tests/test_qa_system.py` (real quote, fake quote, irrelevant real quote, wrong-entity quote, abstention, malformed JSON, empty question) | `results/metrics.json` (`grounded_v1` vs `naive_v0` ablation comparison) | §2 Architecture & Design Decision (with Ablation Results), `README.md` |
| **R3. Deterministic given a fixed seed / reproducible evaluation** | Phase 0, Phase 8 | `src/llm_client.py` (`temperature=0.0`, `seed=42`, SHA-256 disk cache with `bypass_cache` flag, metadata logging) | `tests/test_llm_client.py` (cache key stability, cache hit/bypass, retry backoff, header pacing) | `results/run_metadata.json`, `results/regression_noise_floor.json` | §4 Methodology & Reproducibility Guarantees, §9 Regression & Noise Floor, `README.md` |
| **R4. Ground-truth dataset of $\ge 50$ cases with human verification & trust documentation** | Phase 2 | `evaluation/dataset.py` (`load_dataset`, `validate_dataset`, strict `status == "verified"` enforcement) | `tests/test_dataset.py` (schema validation, verbatim evidence check, unverified-case rejection) | `data/documents/DOC-001.txt`..`DOC-008.txt`, `data/base_cases.jsonl` (50 verified cases) | §3 Dataset Construction, Human Verification Protocol & Limitations |
| **R5. Principled hallucination detection against source material** | Phase 5 | `evaluation/hallucination.py` (Layer 1: exact/normalized & fuzzy quote check; Layer 2: conservative subject/entity/number/unit consistency; Layer 3: cross-model atomic claim judge + human audit) | `tests/test_hallucination.py` (quote presence, wrong-entity value trap, negation/unit mismatch, claim judge parsing, Cohen's $\kappa$) | `data/audit_claims.jsonl` ($\ge 40$ human-labeled claims), `results/hallucination_audit.json` | §6 Hallucination Definition, 3-Layer Detector, Human Audit & Detector Failure Modes |
| **R6. Adversarial input generation (paraphrase, irrelevant context, subtle factual errors, distribution edges)** | Phase 3 | `evaluation/adversarial.py` (10 categories: `paraphrase_hard`, `distractor_near_miss`, `distractor_irrelevant`, `false_premise_numeric`, `false_premise_fabricated`, `source_conflict`, `long_context`, `format_edge`, `prompt_injection`, `empty_malformed`) | `tests/test_adversarial.py`, `tests/test_scoring.py` | `data/variants.jsonl` (100 human-verified variants linked via `base_id`) | §3 Dataset & Adversarial Variants, §5 Per-Category Accuracy & CIs, §10 Reserved Category Design |
| **R7. Confidence calibration testing & calibration curves** | Phase 6 | `evaluation/calibration.py` (10 equal-width bins on $[0,1]$, ECE, Brier score, 1000-resample bootstrap CI, answered-only & abstained-only subsets, $\ge 0.9$ vs $< 0.9$ table, reliability + histogram plot) | `tests/test_calibration.py` (hand-computed toy example verification for ECE & Brier) | `results/calibration.json`, `reports/figures/calibration_curve.png` | §7 Confidence Calibration Analysis & Curves |
| **R8. Failure mode clustering ($\ge 3$ clusters), model-behavior root cause, & concrete engineering fixes** | Phase 7 | `evaluation/clustering.py` (rule-based initial labeling, human review export/import, causal probe runner) | `tests/test_clustering.py` | `results/failure_review.csv`, `results/causal_probes.json`, `reports/figures/failure_distribution.png` | §8 Failure Mode Clusters, Causal Probes & Actionable Engineering Fixes |
| **R9. Regression test runner detecting prompt regressions** | Phase 8 | `evaluation/regression.py` (3-run uncached noise floor to flag `flaky` cases, `grounded_v1` vs `grounded_v2` comparison, planted regression `grounded_v2_bad`) | `tests/test_regression.py` (asserts `grounded_v2_bad` triggers regression detection while excluding flaky cases) | `data/regression_cases.jsonl`, `results/regression_report.json` | §9 Regression Testing & Noise Floor Analysis |
| **R10. Structured evaluation report & reproducible README** | Phase 9 | `evaluation/runner.py`, `evaluation/report_builder.py` (populates metrics strictly from `results/`) | Clean-environment end-to-end execution check | `reports/evaluation_report.md`, `reports/figures/*.png`, `README.md` | All sections (§1–§10) |

---

## 3. Verified LLM Providers, Model IDs, Capabilities, and Credential Security

**Date Checked & Live-Verified (`RUN-00-PREFLIGHT`):** 2026-10-12 (`results/preflight_check.json`)

### 3.1 Approved Fixed Role Assignments (Verified Live in `RUN-00-PREFLIGHT`)
To satisfy **Decision #7** (*"Judge uses a different model than the system under test if available"*), `RUN-00-PREFLIGHT` tested our live API keys and verified the following cross-provider configuration (explicitly approved by the human after `llama-3.1-8b-instant` returned HTTP 404 on the live Groq account roster):
- **Approved Fixed System Under Test (SUT):** **Groq API** — Model ID: `groq/openai/gpt-oss-20b` (`131,072` context window, `65,536` max completion tokens, live headers confirmed `1,000 RPD` and `8,000 TPM`)
- **Approved Fixed Independent Judge (Layer-3 Hallucination Claim Judge & Phase 4 Semantic Judge):** **Google Gemini API** — Model ID: `gemini/gemini-3.5-flash-lite` (`1,048,576` input, `65,536` output, native `response_json_schema`, verified HTTP 200 on live key)

> **Strict Fixed-Model Rule:**  
> 1. Both `groq/openai/gpt-oss-20b` (SUT) and `gemini/gemini-3.5-flash-lite` (Judge) have been verified live in `RUN-00-PREFLIGHT` (`results/preflight_check.json`) and approved by the human.  
> 2. Throughout all comparable runs (`RUN-01` through `RUN-07`), the SUT model (`groq/openai/gpt-oss-20b`) and Judge model (`gemini/gemini-3.5-flash-lite`) remain **strictly fixed**. We will **never** automatically switch models to bypass quota limits; any model change requires human approval and a separate baseline experiment.

### 3.2 Officially Documented & Live-Verified Specs (`RUN-00-PREFLIGHT`, 2026-10-12)

| Model ID & Fixed Role | Provider | Live-Verified Capabilities (`results/preflight_check.json`) | Live-Verified Account Quotas & Rate-Limit Headers | Alternatives on Live Roster | Official Documentation URLs |
|---|---|---|---|---|---|
| **`openai/gpt-oss-20b`**<br>*(Approved Fixed SUT)* | **Groq Cloud** | • **Context / Output Limits:** `131,072` input context window; `65,536` max completion tokens.<br>• **Structured Output:** Supports both `response_format={"type": "json_object"}` and strict `json_schema`, validated with Pydantic `QAOutput.model_validate_json()` + 1 schema-repair retry.<br>• **`temperature` & `seed`:** Verified live with `temperature=0.0` and `seed=42`; returns `system_fingerprint` (`fp_...`). | • **Live Response Headers (`RUN-00-PREFLIGHT`):** `x-ratelimit-limit-requests: 1000` (1,000 RPD) and `x-ratelimit-limit-tokens: 8000` (**8,000 TPM**, giving a **6,800 TPM** safe pacing target with 15% safety margin). | • `openai/gpt-oss-120b` (Groq)<br>• `qwen/qwen3.8-27b` (Groq)<br>*(Note: `llama-3.1-8b-instant` returned HTTP 404 on live account roster and was replaced with human approval before Phase 1).* | • Models: `https://console.groq.com/docs/models`<br>• Structured Outputs: `https://console.groq.com/docs/structured-outputs`<br>• Rate Limits: `https://console.groq.com/docs/rate-limits`<br>• Chat API: `https://console.groq.com/docs/api-reference#chat-create` |
| **`gemini-3.5-flash-lite`**<br>*(Approved Fixed Judge)* | **Google Gemini API** (`google-genai` SDK) | • **Availability:** Verified HTTP 200 (`model_version: gemini-3.5-flash-lite`).<br>• **Context / Output Limits:** `1,048,576` input tokens; `65,536` output tokens.<br>• **Structured Output:** Verified native schema enforcement via `GenerateContentConfig(response_mime_type="application/json", response_json_schema=Schema.model_json_schema())`.<br>• **`temperature` & `seed`:** Verified live with `temperature=0.0` and `seed=42`. | • **Free Tier Quotas:** Standard Free Tier (`15 RPM / 250,000 TPM / 1,000 RPD` baseline; RPD resets at midnight Pacific Time). Safe pacing target set to `12 RPM` and `212,500 TPM`. | • `gemini-3.8-flash` (Stable, Sept 2026) | • Model Specs: `https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite`<br>• Gemini 3.8 Flash: `https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash`<br>• Models Overview: `https://ai.google.dev/gemini-api/docs/models`<br>• Pricing: `https://ai.google.dev/gemini-api/docs/pricing`<br>• Rate Limits: `https://ai.google.dev/gemini-api/docs/rate-limits` |

### 3.3 Credential Protection & Secret Hygiene Policy (Hard Rule #7)
1. **Storage:** API keys (`GROQ_API_KEY` and `GEMINI_API_KEY`) must reside **only** in process environment variables or a local `.env` file at the repository root loaded via `python-dotenv`.
2. **Git Exclusion:** `.gitignore` will be created first in Phase 0 and must exclude `.env`, `.env.*` (with `!.env.example`), `.cache/`, and `__pycache__/`.
3. **Zero-Leak Policy:** API keys will **never** be written into source files (`src/`, `evaluation/`, `tests/`), `PLAN.md`, console logs, exception tracebacks, disk-cache entries, `results/`, `reports/`, or Git commit history. `src/llm_client.py` will sanitize all logged exceptions and request metadata.
4. **Placeholder Template (`.env.example`):**
   ```ini
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

---

## 4. Reproducibility, Determinism, and Confidence Calibration Specifications

### 4.1 Cached Replay Reproducibility vs. Uncached Inference Determinism
Per Official Requirement R3 (*"deterministic given a fixed seed, so that evaluation results are reproducible"*) and Build Guide Decision #6, we separate two distinct concepts:

1. **Cached Replay Reproducibility (Guaranteed 100% Bit-for-Bit Reproducible):**
   - `src/llm_client.py` hashes `(provider, model_id, prompt_version, prompt_text_sha256, schema_sha256, temperature, seed, document_sha256, question_sha256)` using SHA-256.
   - When `bypass_cache=False`, identical requests replay the exact saved response from `.cache/llm_responses/`. This guarantees 100% deterministic reproduction of all saved outputs, scores, tables, and charts.
2. **Uncached Inference Determinism (Best-Effort Only — Not Guaranteed Bit-for-Bit):**
   - Hosted LLM APIs (`groq` and `gemini`) do **not** guarantee bit-for-bit determinism across uncached calls even with `temperature=0.0` and `seed=42`, due to GPU floating-point non-associativity, dynamic batching, and backend updates.
   - `src/llm_client.py` passes `temperature=0.0` and `seed=42` whenever officially supported, explicitly records any unsupported parameter in `results/run_metadata.json`, and logs full provenance per call (`provider`, `model_id`, `system_fingerprint` / `model_version`, `prompt_version`, `prompt_sha256`, `dataset_sha256`, `temperature`, `seed`, `cache_hit`, `retry_count`, `latency_ms`, `input_tokens`, `output_tokens`, `timestamp_utc`).
   - **Empirical Measurement (`RUN-05-NOISE-FLOOR`):** Phase 8 executes 3 uncached repetitions (`bypass_cache=True`, `seed=42`) on the 40 frozen baseline cases in `data/regression_cases.jsonl` to measure the uncached output/score flip rate and isolate `flaky` cases.

### 4.2 Portable Reproducibility Documentation Standard
- `README.md`, `reports/evaluation_report.md`, and `PLAN.md` must use **only repository-relative paths** (e.g., `src/qa_system.py`, `data/base_cases.jsonl`, `results/metrics.json`).
- `README.md` will provide step-by-step clean-environment instructions from a fresh clone: creating a virtual environment, installing pinned `requirements.txt`, copying `.env.example` to `.env`, running `pytest`, reproducing all metrics and plots from cached runs (`bypass_cache=False`), and running uncached verification (`bypass_cache=True`).

### 4.3 Formal Definition of Confidence Calibration (Phase 6)
1. **Meaning & Range of `confidence`:**
   - `confidence` is a floating-point number in $[0.0, 1.0]$ representing the estimated probability that a response (`answer` or `abstained=True` decision) is **correct** under the evaluation scoring rubric.
   - Out-of-range values returned by a model are rejected by Pydantic validation (`ge=0.0, le=1.0`), triggering the malformed-output retry path.
2. **Binary Correctness Target ($y_i \in \{0, 1\}$):**
   - For each evaluated record $i$, `evaluation/scoring.py` produces a binary correctness indicator $y_i \in \{0, 1\}$:
     - **If the case requires an answer (`should_abstain == False`):** $y_i = 1$ iff `abstained == False` and the returned `answer` passes the category's correctness rubric (exact/normalized match or semantic rubric judge, plus false-premise correction / conflict flagging where required). If the system abstains (`OVER_ABSTENTION`) or gives a wrong answer, $y_i = 0$.
     - **If the case requires abstention (`should_abstain == True`):** $y_i = 1$ iff `abstained == True` (correct abstention). If the system outputs a substantive answer (`UNDER_ABSTENTION`), $y_i = 0$.
3. **Representation of Evidence-Gate Forced Abstentions & Fallback Disclosure:**
   - Both `raw_output` (pre-gate) and `final_output` (post-gate) are stored for every record.
   - When the evidence-quote gate overrides a substantive `raw_output` into `final_output.abstained = True` because cited quotes failed verification, `final_output.confidence` is assigned a fixed `0.50` value. **Crucial Limitation & Disclosure:** This `0.50` value is a **chosen engineering fallback heuristic representing maximum epistemic uncertainty after a failed quote check—it is NOT a model-measured probability.**
   - Therefore, we compute and report **two separate calibration analyses** (and never present the post-gate score as the model's intrinsic confidence):
     1. **Intrinsic Model Calibration (`raw_output`):** Uses the LLM's unaltered pre-gate `raw_output.confidence` evaluated against `raw_output` correctness $y_{i,\text{raw}}$. This measures whether the model's own verbalized confidence is well-calibrated.
     2. **Post-Gate Operational System Calibration (`final_output`):** Uses `final_output.confidence` (which includes the `0.50` heuristic fallback on gate-overridden records) evaluated against `final_output` correctness $y_{i,\text{final}}$, explicitly reporting the count of gate-overridden records placed into the $[0.5, 0.6)$ bin and disclosing that those entries reflect a deterministic rule rather than intrinsic model probability.
4. **Records Included in ECE & Brier Calculations ($N = 144$):**
   - **Evaluated Set ($N = 144$):** Includes all verified evaluation records that execute an LLM call (all 50 base cases + 94 non-empty variants, spanning both answered and abstained outputs).
   - **Excluded Pre-API Rejections ($N = 6$):** The 6 `empty_malformed` input-validation cases are rejected before calling the LLM and have no model-generated confidence; they are excluded from ECE/Brier and documented explicitly with the $N = 144$ denominator.
   - **Formulas:** Using $M = 10$ equal-width bins $B_1, \dots, B_{10}$ across $[0.0, 1.0]$ ($[0.0, 0.1), \dots, [0.9, 1.0]$, skipping and listing empty bins):
     $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|, \qquad \text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (c_i - y_i)^2$$
     with a 1,000-resample bootstrap 95% confidence interval (`seed=42`).
5. **Separate Disaggregated Calibration Analyses (Clearly Labeled in Report §7):**
   - **Intrinsic Model (`raw_output`) vs. Post-Gate System (`final_output`) Calibration:** Side-by-side ECE, Brier score, and reliability curves, plus a third view excluding gate-overridden records so the `0.50` fallback does not conflate model calibration with gate behavior.
   - **Answered-Only Subset (`abstained == False`):** Evaluates whether confidence predicts factual answer accuracy when the system chooses to answer.
   - **Abstained-Only Subset (`abstained == True`):** Evaluates whether confidence when abstaining aligns with true unanswerability (abstention precision), separating voluntary model abstentions from gate-forced abstentions.
   - **High-Confidence Diagnostic Table:** Compares empirical accuracy and sample counts for $\text{confidence} \ge 0.90$ vs. $< 0.90$.

---

## 5. Reconciled Worst-Case API-Call and Token Budget (Provisional Until Live Preflight Passes)

> **Provisional Quota Notice:** All provider rate limits and daily quota ceilings in §5.1–§5.3 (`30 RPM / 6,000 TPM / 14,400 RPD / 500,000 TPD` for Groq `openai/gpt-oss-20b`; `15 RPM / 250,000 TPM / 1,000 RPD` for Gemini `gemini-3.5-flash-lite`) are treated as **provisional planning figures** until confirmed against live response headers and account dashboards during **`RUN-00-PREFLIGHT` (`PF-4`)**. Groq and Google Gemini enforce completely independent per-provider quotas; cross-provider sums are shown strictly as informational project-wide effort estimates, never as a single shared quota pool.

### 5.1 Document Sizing & Token Assumptions (Designed to Fit Worst-Case Quotas)
- **Source Document Length Target:** Our 8 counterfactual documents (`data/documents/DOC-001.txt` ... `DOC-008.txt`) are authored at **350 to 750 words each** (average **~500 words $\approx 680$ tokens**), strictly inside Build Guide Phase 2's `300 to 1500 words` specification.
- **SUT Input/Output Token Averages (`groq/openai/gpt-oss-20b`):**
  - Standard `grounded_v1` call: $\sim 680\text{ doc tokens} + 200\text{ prompt tokens} + 20\text{ question tokens} \approx 900\text{ input tokens}$.
  - `long_context` variants (9 records: 3 base cases $\times$ 3 positions `start/middle/end` padded 5x to $\sim 2,500\text{ words} \approx 3,400\text{ tokens}$): $\sim 3,600\text{ input tokens/call}$ (safely below Groq's provisional `6,000 TPM` ceiling).
  - Weighted average across all 144 calls in `RUN-01-MAIN`:
    $$\frac{(135 \times 827) + (9 \times 3,600)}{144} \le \mathbf{1,000\text{ input tokens/call}}, \quad \mathbf{150\text{ output tokens/call}} \quad (\mathbf{1,150\text{ total tokens/call}})$$
  - `naive_v0` (`RUN-02-ABLATION`, 144 calls, shorter prompt): $\mathbf{900\text{ input tokens/call}} + \mathbf{150\text{ output tokens/call}} = \mathbf{1,050\text{ total tokens/call}}$.
  - Standard non-padded runs (`RUN-04-PROBES`, `RUN-05-NOISE-FLOOR`, `RUN-06`, `RUN-07`): $\mathbf{900\text{ input tokens/call}} + \mathbf{150\text{ output tokens/call}} = \mathbf{1,050\text{ total tokens/call}}$ ($\mathbf{850\text{ in} / 150\text{ out}}$ for preflight smoke calls).
- **Judge Input/Output Token Averages (`gemini/gemini-3.5-flash-lite`):**
  - Phase 4 Semantic Scoring Judge (~15% of SUT outputs): $\mathbf{600\text{ input}} + \mathbf{100\text{ output}} = \mathbf{700\text{ total tokens/call}}$.
  - Phase 5 Layer-3 Combined Atomic Claim Extraction + Entailment Judge (~40% of SUT outputs): $\mathbf{1,100\text{ input}} + \mathbf{200\text{ output}} = \mathbf{1,300\text{ total tokens/call}}$.
  - Phase 5 Human Audit Set (`45` claims in `data/audit_claims.jsonl`): $\mathbf{1,000\text{ input}} + \mathbf{150\text{ output}} = \mathbf{1,150\text{ total tokens/call}}$.
- **Worst-Case Buffer Arithmetic:** Every phase includes **+15% retry calls** ($\lceil \text{Base} \times 0.15 \rceil$) and **+20% contingency re-run calls** ($\lceil \text{Base} \times 0.20 \rceil$), assuming worst-case token billing on every retry and contingency call.

### 5.2 Per-Phase, Per-Provider, Per-Model API Budget Table

| Row | Phase & Scheduled Run | Provider & Fixed Model | Quota Window | Base Calls | Retry (+15%) | Conting. (+20%) | Worst-Case Max Calls | Avg In / Out Tok | Base Input Tokens | Base Output Tokens | Base Total Tokens | Worst-Case Max Input Tokens | Worst-Case Max Output Tokens | Worst-Case Max Total Tokens |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | **Phase 0–1 (`RUN-00-PREFLIGHT`):** SUT Smoke & Gate Checks | `groq/openai/gpt-oss-20b` | Window 1 (Day 1) | 8 | 2 | 2 | **12** | 850 / 150 | 6,800 | 1,200 | **8,000** | 10,200 | 1,800 | **12,000** |
| **2** | **Phase 0–1 (`RUN-00-PREFLIGHT`):** Judge Capability Checks | `gemini/gemini-3.5-flash-lite` | Window 1 (Day 1) | 4 | 1 | 1 | **6** | 850 / 150 | 3,400 | 600 | **4,000** | 5,100 | 900 | **6,000** |
| **3** | **Phase 4 (`RUN-01-MAIN`):** `grounded_v1` Full Suite ($150 - 6$) | `groq/openai/gpt-oss-20b` | Window 1 (Day 1) | 144 | 22 | 29 | **195** | 1,000 / 150 | 144,000 | 21,600 | **165,600** | 195,000 | 29,250 | **224,250** |
| **4** | **Phase 4 (`RUN-02-ABLATION`):** `naive_v0` Full Suite ($150 - 6$) | `groq/openai/gpt-oss-20b` | Window 1 (Day 1) | 144 | 22 | 29 | **195** | 900 / 150 | 129,600 | 21,600 | **151,200** | 175,500 | 29,250 | **204,750** |
| **5** | **Phase 4:** Semantic Scoring Judge ($288 \times 15\%$) | `gemini/gemini-3.5-flash-lite` | Window 1 (Day 1) | 44 | 7 | 9 | **60** | 600 / 100 | 26,400 | 4,400 | **30,800** | 36,000 | 6,000 | **42,000** |
| **6** | **Phase 5:** Layer-3 Claim Extract + Judge ($288 \times 40\%$) | `gemini/gemini-3.5-flash-lite` | Window 1 (Day 1) | 116 | 18 | 24 | **158** | 1,100 / 200 | 127,600 | 23,200 | **150,800** | 173,800 | 31,600 | **205,400** |
| **7** | **Phase 5 (`RUN-03-JUDGE-AUDIT`):** Audit Set (45 claims) | `gemini/gemini-3.5-flash-lite` | Window 2 (Day 2) | 45 | 7 | 9 | **61** | 1,000 / 150 | 45,000 | 6,750 | **51,750** | 61,000 | 9,150 | **70,150** |
| **8** | **Phase 7 (`RUN-04-PROBES`):** Causal Probes ($4 \times 8$) | `groq/openai/gpt-oss-20b` | Window 2 (Day 2) | 32 | 5 | 7 | **44** | 900 / 150 | 28,800 | 4,800 | **33,600** | 39,600 | 6,600 | **46,200** |
| **9** | **Phase 8 (`RUN-05-NOISE-FLOOR`):** $40\text{ cases} \times 3$ uncached | `groq/openai/gpt-oss-20b` | Window 2 (Day 2) | 120 | 18 | 24 | **162** | 900 / 150 | 108,000 | 18,000 | **126,000** | 145,800 | 24,300 | **170,100** |
| **10** | **Phase 8 (`RUN-06` & `RUN-07`):** `grounded_v2` (50) + `v2_bad` (20) | `groq/openai/gpt-oss-20b` | Window 2 (Day 2) | 70 | 11 | 14 | **95** | 900 / 150 | 63,000 | 10,500 | **73,500** | 85,500 | 14,250 | **99,750** |
| **11** | **Phase 7–8:** Semantic Judge on Probes/Regression ($222 \times 15\%$) | `gemini/gemini-3.5-flash-lite` | Window 2 (Day 2) | 34 | 6 | 7 | **47** | 600 / 100 | 20,400 | 3,400 | **23,800** | 28,200 | 4,700 | **32,900** |
| **S1** | **GROQ PROVIDER SUBTOTAL (Rows 1, 3, 4, 8, 9, 10)** | **`groq/openai/gpt-oss-20b`** | **Both Windows** | **518** | **80** | **105** | **703** | — | **480,200** | **77,700** | **557,900** | **651,600** | **105,450** | **757,050** |
| **S2** | **GEMINI PROVIDER SUBTOTAL (Rows 2, 5, 6, 7, 11)** | **`gemini/gemini-3.5-flash-lite`** | **Both Windows** | **243** | **39** | **50** | **332** | — | **222,800** | **38,350** | **261,150** | **304,100** | **52,350** | **356,450** |
| **—** | **COMBINED CROSS-PROVIDER ESTIMATE (Informational Sum Across Separate Quotas — Not a Single Quota)** | **Groq (`757,050` max) + Gemini (`356,450` max)** | **Both Windows** | **761** | **119** | **155** | **1,035** | — | **703,000** | **116,050** | **819,050** | **955,700** | **157,800** | **1,113,500** |

### 5.3 Separate Provider Quota Tables, Reset Windows, and Scheduled Run Order

#### A. Quota Reset Windows & Scheduled Execution Order
- **Groq Daily Quota Reset Window:** Groq tracks RPD and TPD over a 24-hour window (exposed dynamically via `x-ratelimit-reset-requests` and `x-ratelimit-reset-tokens` response headers; UTC midnight / rolling 24h window).
- **Google Gemini Daily Quota Reset Window:** Officially documented (`rate-limits.md`) as resetting daily at **midnight Pacific Time (00:00 PT = 07:00 UTC = 12:30 PM IST during PDT / 13:30 PM IST during PST)**.
- **Strict Run Execution Order Aligned With Reset Windows:**
  1. **Quota Window 1 (Day 1 — Executed Before Daily Quota Reset):**
     - Step 1.1: `RUN-00-PREFLIGHT` (Rows 1 & 2) — verifies live headers and exact reset timestamps before any dataset work.
     - Step 1.2: After Checkpoint 2 human dataset verification, execute `RUN-01-MAIN` (`grounded_v1`, Row 3) first so the primary evaluation run has first claim on Window 1 quota (`224,250` worst-case max tokens out of `500,000` provisional Groq TPD).
     - Step 1.3: Execute `RUN-02-ABLATION` (`naive_v0`, Row 4) and Day 1 Gemini Judge passes (Rows 5 & 6).
  2. **Quota Window 2 (Day 2 — Executed After Daily Quota Reset):**
     - Step 2.1: If any tail records of `RUN-02-ABLATION` were deferred by the rate limiter at the end of Window 1, complete those cached-resume records first.
     - Step 2.2: Execute `RUN-03-JUDGE-AUDIT` (Row 7, Gemini Judge), `RUN-04-PROBES` (Row 8, Groq SUT), `RUN-05-NOISE-FLOOR` (Row 9, Groq SUT), `RUN-06`/`RUN-07` Regression (Row 10, Groq SUT), and Phase 7–8 Judge calls (Row 11, Gemini Judge).

#### B. Separate Per-Provider Quota Verification Tables (Provisional Until `PF-4` Passes)

**Table 5.3a — Groq Cloud Quota Budget (`groq/openai/gpt-oss-20b`, Fixed SUT)**

| Quota Window | Scheduled Runs (Fixed Order) | Base Calls | Worst-Case Max Calls | Provisional RPD Limit | Base Total Tokens | Worst-Case Max Total Tokens | Provisional TPD Limit | Provisional Quota Headroom & Carryover Buffer |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **Window 1 (Day 1)** | `RUN-00-PREFLIGHT` (Row 1) $\rightarrow$ `RUN-01-MAIN` (Row 3) $\rightarrow$ `RUN-02-ABLATION` (Row 4) | 296 | **402** | 14,400 RPD *(provisional)* | **324,800** | **441,000** | **500,000 TPD** *(provisional)* | **Fits Window 1** (`65.0%` base / `88.2%` worst-case max; **+59,000 tokens of unused Window 1 headroom**). |
| **Window 2 (Day 2)** | Any Window 1 carryover $\rightarrow$ `RUN-04-PROBES` (Row 8) $\rightarrow$ `RUN-05-NOISE-FLOOR` (Row 9) $\rightarrow$ `RUN-06`/`07` (Row 10) | 222 | **301** | 14,400 RPD *(provisional)* | **233,100** | **316,050** | **500,000 TPD** *(provisional)* | **Fits Window 2** (`46.6%` base / `63.2%` worst-case max; **+183,950 tokens of unused Window 2 headroom**, sufficient to absorb up to **89% of `RUN-02-ABLATION`** as carryover). |
| **Groq 2-Window Total** | All SUT Runs (Rows 1, 3, 4, 8, 9, 10) | **518** | **703** | 28,800 (2 windows) | **557,900** | **757,050** | **1,000,000** (2 windows) | **Fits 2-window allowance** (`55.8%` base / `75.7%` worst-case max; **+242,950 tokens unused margin**). |

**Table 5.3b — Google Gemini API Quota Budget (`gemini/gemini-3.5-flash-lite`, Fixed Judge)**

| Quota Window (Resets 00:00 PT) | Scheduled Runs (Fixed Order) | Base Calls | Worst-Case Max Calls | Provisional RPD Limit | Base Total Tokens | Worst-Case Max Total Tokens | Provisional TPM Limit | Provisional Quota Headroom & Carryover Buffer |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **Window 1 (Day 1)** | `RUN-00-PREFLIGHT` (Row 2) $\rightarrow$ Phase 4 Judge (Row 5) $\rightarrow$ Phase 5 Judge (Row 6) | 164 | **224** | 1,000 RPD *(provisional)* | **185,600** | **253,400** | 250,000 TPM *(provisional; no TPD cap)* | **Fits Window 1** (`16.4%` base / `22.4%` worst-case max of provisional 1,000 RPD; **+776 requests unused headroom**). |
| **Window 2 (Day 2)** | `RUN-03-JUDGE-AUDIT` (Row 7) $\rightarrow$ Phase 7–8 Judge (Row 11) | 79 | **108** | 1,000 RPD *(provisional)* | **75,550** | **103,050** | 250,000 TPM *(provisional; no TPD cap)* | **Fits Window 2** (`7.9%` base / `10.8%` worst-case max of provisional 1,000 RPD; **+892 requests unused headroom**, sufficient to absorb 100% of Window 1 Judge calls if deferred). |
| **Gemini 2-Window Total** | All Judge Runs (Rows 2, 5, 6, 7, 11) | **243** | **332** | 2,000 (2 windows) | **261,150** | **356,450** | 250,000 TPM *(provisional)* | Cost on Free Tier: **$0.00 USD** (Paid equivalent $\approx \$0.20$ USD). |

---

## 6. Adaptive Header- and Token-Aware Rate Pacing & Schedule Duration

### 6.1 Adaptive Rate Limiter Specification (`src/llm_client.py`)
Instead of relying on a blind fixed sleep interval, `src/llm_client.py` implements a **dual-mechanism adaptive rate limiter**:
1. **Pre-Request Token & Request Window Check (15% Safety Margin):**
   - Tracks a rolling 60-second sliding window of actual `(timestamp, total_tokens)` and request counts per provider/model.
   - Applies a **15% safety margin** below the verified provider ceiling:
     - For Groq `openai/gpt-oss-20b` (`6,000 TPM`, `30 RPM` provisional): targets $\le \mathbf{5,100\text{ TPM}}$ and $\le \mathbf{25\text{ RPM}}$.
     - For Gemini `gemini-3.5-flash-lite` (`250,000 TPM`, `15 RPM` provisional): targets $\le \mathbf{212,500\text{ TPM}}$ and $\le \mathbf{12\text{ RPM}}$.
   - Before dispatching an uncached request with estimated tokens $T_{\text{est}} = \lceil \text{chars}/3.5 \rceil + \text{max\_output\_tokens}$, if $\text{Tokens}_{\text{last\_60s}} + T_{\text{est}} > \text{TPM}_{\text{safe}}$, sleeps for the exact time until the oldest entries in the 60-second window expire.
2. **Live Provider Rate-Limit Header Feedback:**
   - Parses `x-ratelimit-remaining-tokens`, `x-ratelimit-remaining-requests`, `x-ratelimit-reset-tokens`, `x-ratelimit-reset-requests`, and `Retry-After` from every response.
   - If `x-ratelimit-remaining-tokens` drops below $T_{\text{est}} \times 1.15$, sleeps for the exact duration reported in `x-ratelimit-reset-tokens` plus a $250\text{ ms}$ jitter buffer.
   - If an HTTP `429` still occurs, executes bounded exponential backoff with jitter (`base=5s, factor=2.0, max_retries=5`), honoring `Retry-After`.

### 6.2 Reconciled Wall-Clock Run Duration Table

| Run ID | Provider & Model | Base / Worst-Case Calls | Base / Worst-Case Tokens | Safe Throughput (`TPM_safe` / `RPM_safe`) | Estimated Uncached Duration (Base / Worst-Case) | Cached Replay Duration |
|---|---|---:|---:|---|---|---|
| `RUN-00-PREFLIGHT` | Groq SUT (8) + Gemini Judge (4) | 12 / 18 | 12k / 18k | 5,100 TPM (Groq) / 12 RPM (Gemini) | **~2.5 mins / ~3.8 mins** | `< 1 sec` |
| `RUN-01-MAIN` (`grounded_v1`) | Groq `openai/gpt-oss-20b` | 144 / 195 | 165.6k / 224.3k | 5,100 TPM ($\approx 4.4\text{ calls/min}$) | **~32.5 mins / ~44.0 mins** | `< 1 sec` |
| `RUN-02-ABLATION` (`naive_v0`) | Groq `openai/gpt-oss-20b` | 144 / 195 | 151.2k / 204.8k | 5,100 TPM ($\approx 4.8\text{ calls/min}$) | **~29.7 mins / ~40.2 mins** | `< 1 sec` |
| Phase 4 & 5 Judge (`RUN-01` & `02`) | Gemini `gemini-3.5-flash-lite` | 160 / 218 | 181.6k / 247.4k | 12 RPM ($\approx 12\text{ calls/min}$) | **~13.4 mins / ~18.2 mins** | `< 1 sec` |
| `RUN-03-JUDGE-AUDIT` | Gemini `gemini-3.5-flash-lite` | 45 / 61 | 51.8k / 70.2k | 12 RPM ($\approx 12\text{ calls/min}$) | **~3.8 mins / ~5.1 mins** | `< 1 sec` |
| `RUN-04-PROBES` | Groq `openai/gpt-oss-20b` | 32 / 44 | 33.6k / 46.2k | 5,100 TPM ($\approx 4.8\text{ calls/min}$) | **~6.6 mins / ~9.1 mins** | `< 1 sec` |
| `RUN-05-NOISE-FLOOR` | Groq `openai/gpt-oss-20b` | 120 / 162 | 126.0k / 170.1k | 5,100 TPM ($\approx 4.8\text{ calls/min}$) | **~24.8 mins / ~33.4 mins** | N/A (`bypass_cache=True`) |
| `RUN-06` & `RUN-07` (Regression) | Groq `openai/gpt-oss-20b` | 70 / 95 | 73.5k / 99.8k | 5,100 TPM ($\approx 4.8\text{ calls/min}$) | **~14.5 mins / ~19.6 mins** | `< 1 sec` |
| Phase 7–8 Judge | Gemini `gemini-3.5-flash-lite` | 34 / 47 | 23.8k / 32.9k | 12 RPM ($\approx 12\text{ calls/min}$) | **~2.9 mins / ~4.0 mins** | `< 1 sec` |
| **GROQ WALL-CLOCK SUBTOTAL** | `groq/openai/gpt-oss-20b` | **518 / 703** | **557.9k / 757.1k** | 5,100 TPM safe target | **~109.7 mins (1.8 hrs) Base / ~148.6 mins (2.5 hrs) Worst-Case** | `< 3 secs` |
| **GEMINI WALL-CLOCK SUBTOTAL** | `gemini/gemini-3.5-flash-lite` | **243 / 332** | **261.2k / 356.5k** | 12 RPM safe target | **~20.8 mins (0.35 hrs) Base / ~28.8 mins (0.48 hrs) Worst-Case** | `< 2 secs` |
| **COMBINED CROSS-PROVIDER ESTIMATE (Informational Sum)** | **Groq + Gemini (Separate Quotas)** | **761 / 1,035** | **819.1k / 1,113.5k** | — | **~130.5 mins (2.2 hrs) Base / ~177.4 mins (3.0 hrs) Worst-Case** | **`< 5 secs` full suite** |

---

## 7. Evaluation Run Matrix

| Run ID | Phase | Prompt Version | Dataset / Input Slice | Records ($N$) | Base / Max Calls | Fixed SUT Model | Fixed Judge Model | `seed` / `temp` | Cache Mode | Repetitions | Output Artifact (`results/`) |
|---|---|---|---|---:|---:|---|---|---|---|---|---|
| `RUN-00-PREFLIGHT` | Phase 0–1 | `naive_v0` & `grounded_v1` | Synthetic smoke & 5 gate test inputs | 8 | 12 / 18 | `groq/openai/gpt-oss-20b` | `gemini/gemini-3.5-flash-lite` | `seed=42`, `temp=0.0` | Both tested | 1 | `results/preflight_check.json` |
| `RUN-01-MAIN` | Phase 4–6 | `grounded_v1` (with Evidence-Quote Gate) | `data/base_cases.jsonl` (50) + `data/variants.jsonl` (100) | 150 | 144 / 195 SUT | `groq/openai/gpt-oss-20b` | `gemini/gemini-3.5-flash-lite` | `seed=42`, `temp=0.0` | `bypass_cache=False` | 1 | `results/outputs_grounded_v1.jsonl`, `results/scored_grounded_v1.jsonl`, `results/claims_grounded_v1.jsonl` |
| `RUN-02-ABLATION` | Phase 1, 6 | `naive_v0` (No gate, no abstention rule) | `data/base_cases.jsonl` (50) + `data/variants.jsonl` (100) | 150 | 144 / 195 SUT | `groq/openai/gpt-oss-20b` | `gemini/gemini-3.5-flash-lite` | `seed=42`, `temp=0.0` | `bypass_cache=False` | 1 | `results/outputs_naive_v0.jsonl`, `results/scored_naive_v0.jsonl`, `results/claims_naive_v0.jsonl` |
| `RUN-03-JUDGE-AUDIT` | Phase 5 | Fixed Layer-3 Claim Rubric | `data/audit_claims.jsonl` (45 hand-labeled claims) | 45 | 45 / 61 Judge | N/A | `gemini/gemini-3.5-flash-lite` | `seed=42`, `temp=0.0` | `bypass_cache=False` | 1 | `results/hallucination_audit.json` |
| `RUN-04-PROBES` | Phase 7 | `grounded_v1` | Counterfactual probe inputs for observed failure clusters | 32 | 32 / 44 SUT | `groq/openai/gpt-oss-20b` | `gemini/gemini-3.5-flash-lite` | `seed=42`, `temp=0.0` | `bypass_cache=False` | 1 | `results/causal_probes.json` |
| `RUN-05-NOISE-FLOOR` | Phase 8 | `grounded_v1` | `data/regression_cases.jsonl` (40 frozen passing base cases) | $40 \times 3$ | 120 / 162 SUT | `groq/openai/gpt-oss-20b` | `gemini/gemini-3.5-flash-lite` | `seed=42`, `temp=0.0` | **`bypass_cache=True`** | 3 uncached runs | `results/regression_noise_floor.json` |
| `RUN-06-REGRESSION-V2` | Phase 8 | `grounded_v2` (Refined prompt) | `data/regression_cases.jsonl` (40) + 10 prior failures | 50 | 50 / 68 SUT | `groq/openai/gpt-oss-20b` | `gemini/gemini-3.5-flash-lite` | `seed=42`, `temp=0.0` | `bypass_cache=False` | 1 | `results/regression_grounded_v2.json` |
| `RUN-07-REGRESSION-BAD` | Phase 8 | `grounded_v2_bad` (Planted regression prompt) | 20-case subset of `data/regression_cases.jsonl` | 20 | 20 / 27 SUT | `groq/openai/gpt-oss-20b` | Deterministic | `seed=42`, `temp=0.0` | `bypass_cache=False` | 1 | `results/regression_planted_bad.json` |

---

## 8. Ground-Truth Governance, Failure-Cluster Requirement, and Judge Audit

### 8.1 Ground-Truth Verification Protocol (Hard Rules #3 & #6)
- **Official Requirement vs. Build Target:** The official assignment requires *"at least 50 test cases with known correct outputs."* Our Build Guide design target is **50 human-verified non-adversarial base cases** (`data/base_cases.jsonl`, ~38 answerable and ~12 unanswerable-by-design) plus **100 human-reviewed adversarial variants** (`data/variants.jsonl`), totaling **150 evaluation records**. Verification quality takes strict priority over variant quantity.
- **Draft-Only Creation:** Every generated or drafted record in `data/base_cases.jsonl` and `data/variants.jsonl` starts with `"status": "draft"`, `"verified_by": null`, `"verified_on": null`.
- **No Automated Promotion:** No script, validator, or agent action is permitted to set `"status": "verified"` or alter `expected_answer` to make a failing test pass.
- **Hard Runner Refusal:** `evaluation/dataset.py` and `evaluation/runner.py` enforce `status == "verified"` on every record and abort immediately if any unverified record is present.

### 8.2 Resolution of the Three-Failure-Cluster Requirement (Phase 7)
- **Assignment Requirement:** Official Requirement R8 mandates: *"At least 3 distinct failure mode clusters must emerge and be labeled."*
- **Observation-Driven Review:** `evaluation/clustering.py` assigns initial rule-based candidate labels and exports `results/failure_review.csv`. The human reviews every failed output and assigns verified cluster labels based on observed evidence.
- **Mandatory Escalation Protocol if $< 3$ Clusters Appear Initially:**
  1. **Never Fabricate or Split Clusters:** We will never invent synthetic failure clusters or artificially subdivide a single failure mechanism into cosmetic sub-labels.
  2. **Targeted Human-Verified Expansion:** If `RUN-01-MAIN` produces fewer than 3 distinct genuine failure clusters, we will design and human-verify additional targeted adversarial test cases specifically probing unexercised vulnerability dimensions (e.g., cross-sentence unit arithmetic, multi-entity table confusion, conditional scope negation, or conflicting temporal clauses) and execute a supplementary verified evaluation pass.
  3. **Prominent Unmet-Requirement Flag if Still $< 3$:** If fewer than 3 genuine failure mode clusters can still be demonstrated after the targeted expansion pass, we must **prominently flag the unmet requirement at Checkpoint 3 and in the Executive Summary of `reports/evaluation_report.md`**, documenting the exact observed clusters and all targeted probes executed, rather than treating the deliverable as complete.

### 8.3 Hallucination Judge Limitations & Human Audit Procedure (Phase 5)
- **Documented Judge Failure Modes:**
  1. *Multi-Sentence / Unit-Conversion Reasoning:* May mislabel valid unit conversions ($1,500\text{ mA} = 1.5\text{ A}$) as `unsupported` or accept plausible arithmetic errors.
  2. *Negation & Conditional Scope Inversions:* High lexical overlap can trick the judge into labeling a contradicted conditional claim as `supported`.
  3. *Shared Parametric Bias:* May forgive ungrounded claims that align with real-world pretraining priors.
- **Human Audit Protocol (`data/audit_claims.jsonl`):**
  1. Construct **45 atomic claims** ($\ge 40$ required) across clean facts, hard paraphrases, unit conversions, negations, near-miss entity traps, and unsupported inferences.
  2. The human independently records `human_label` (`supported / contradicted / unsupported`) and `human_notes` **prior** to comparing with judge verdicts.
  3. `evaluation/hallucination.py` computes overall agreement %, per-class confusion matrix, unweighted **Cohen's $\kappa$** with a 95% confidence interval, and lists all disagreement cases in `reports/evaluation_report.md` §6.

---

## 9. Reserved Adversarial Category Design (Dimensionally Consistent — Held Back for Interview)

Per Build Guide Phase 3, we implement 10 adversarial categories in `evaluation/adversarial.py` and hold back **Category 11: `multi_hop_conditional` (Cross-Clause Conditional & Electric Current Override)** on paper in Section 10 of `reports/evaluation_report.md`:

- **What It Tests:** Whether the QA system reads past a baseline numerical specification in Paragraph 1 to apply a conditional override in Paragraph 4, combining values that share identical physical dimensions.
- **Concrete Example (Dimensionally Consistent in Milliamperes, $\text{mA}$):**
  - *Source Fact 1 (Paragraph 1 — Standard Operation):*
    > `"Under standard ambient temperatures (20 °C to 40 °C), the CryoSense Telemetry Module draws a continuous standby current of 15 mA."`
  - *Source Fact 2 (Paragraph 4 — Low-Temperature Override):*
    > `"When ambient temperature falls below 0 °C, the core voltage regulator triples its continuous standby current draw to maintain clock stability, and the integrated trace heater draws an additional continuous current of 10 mA."`
  - *Adversarial Question:*
    > `"What is the total continuous current draw (in mA) of the CryoSense Telemetry Module while operating in standby at -10 °C?"`
- **Dimensional & Arithmetic Derivation of Ground Truth:**
  1. Since $-10\text{ }^\circ\text{C} < 0\text{ }^\circ\text{C}$, the Paragraph 4 sub-zero override clause applies.
  2. Core regulator standby current: $3 \times 15\text{ mA} = 45\text{ mA}$.
  3. Integrated trace heater current: $+10\text{ mA}$.
  4. Total continuous current draw:
     $$I_{\text{total}} = 45\text{ mA} + 10\text{ mA} = 55\text{ mA}$$
  - Every term has identical physical units (electric current in milliamperes, $\text{mA}$).
- **Expected Output & Verification Rule:**
  - `expected_answer`: `"55 mA"`; `expected_evidence`: must cite **both** the Paragraph 1 baseline sentence and the Paragraph 4 override sentence.
  - Any instantiated record starts as `"status": "draft"` and must be human-verified for source fidelity, unit consistency, and arithmetic accuracy before use.

---

## 10. Implementation Feasibility & Hard Preflight Verification Gate (`RUN-00-PREFLIGHT`)

### 10.1 Feasibility & 48-Hour Priority Order
- **Total Uncached API Runtime:** **~2.2 hours base / ~3.0 hours worst-case** across the 48-hour schedule, with `< 5 seconds` for cached replays.
- **Strict Deliverable Priority:** Phases 0–9 (CLI QA system, verified dataset & variants, 3-layer hallucination detector + human audit, calibration curves, $\ge 3$ failure clusters + causal probes, regression runner + noise floor, `reports/evaluation_report.md`, and `README.md`) take 100% priority over the optional Phase 10 UI.

### 10.2 Hard Preflight Gate (`RUN-00-PREFLIGHT` — Must Pass Before Dataset Generation or Full Runs)
Every item below is treated as **unconfirmed until tested** in Phase 0–1. We distinguish basic HTTP connectivity from capability verification:

- [ ] **PF-1. Credential & Git Hygiene Verification:** Confirm `.gitignore` excludes `.env` and `.cache/`, confirm `.env.example` contains only placeholders, and verify via `git status --ignored` that no secrets can be committed.
- [ ] **PF-2. Live Model Availability & Basic Connectivity Check:** Execute 1 minimal live call to `groq/openai/gpt-oss-20b` and 1 minimal live call to `gemini/gemini-3.5-flash-lite`. Verify HTTP 200 and confirm neither model ID returns a deprecation, permission, or not-found error.
- [ ] **PF-3. Individual Capability Verification (Structured Output, `seed`, `temperature`, Token Usage):**
  - Verify `groq/openai/gpt-oss-20b` accepts `temperature=0.0`, `seed=42`, and `response_format={"type": "json_object"}`, returns `system_fingerprint` and `usage` (`prompt_tokens`, `completion_tokens`), and parses cleanly into `QAOutput`.
  - Verify `gemini/gemini-3.5-flash-lite` accepts `temperature=0.0`, `seed=42`, and `response_json_schema`, returns `usage_metadata`, and parses cleanly into the Judge Pydantic schema.
- [ ] **PF-4. Live Account Quota & Rate-Limit Header Inspection:**
  - Inspect live Groq response headers (`x-ratelimit-limit-requests`, `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens`) and check the Google AI Studio Rate Limit page (`https://aistudio.google.com/rate-limit`).
  - Confirm that the verified account limits accommodate the worst-case budget in §5.3 (`441,000` Day 1 tokens and `316,050` Day 2 tokens on Groq; `224` Day 1 calls and `108` Day 2 calls on Gemini).
- [ ] **PF-5. Cache Hit, Cache Bypass & Error-Handling Tests (`pytest`):**
  - Verify `bypass_cache=False` replays from `.cache/` with 0 network calls, and `bypass_cache=True` forces a live provider call.
  - Verify malformed JSON triggers 1 retry then returns a structured error fallback without crashing, and empty/whitespace questions fail validation before any API call is made.
- [ ] **PF-6. Evidence-Quote Gate Verification (`pytest`):** Verify all 5 gate test cases (exact quote pass, whitespace/quote normalized pass, `difflib` $\ge 0.90$ recorded as `fuzzy_candidate`, fabricated quote forced abstention preserving `raw_output`, and real-but-irrelevant quote passed by Layer 1 and flagged by scoring/Layer 2–3).

> **Hard Stop Condition:** If any check in `PF-2`, `PF-3`, or `PF-4` fails or shows that active quotas cannot support the worst-case budget in §5, **stop immediately and request human approval** for a revised fixed model/quota configuration before starting Phase 2 or `RUN-01-MAIN`.
