"""Pydantic schemas for the QA system, evidence-quote gate, dataset records, and judges."""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, PrivateAttr


class QAOutput(BaseModel):
    """Structured output returned by the Document QA system.

    Confidence definition: the estimated probability in [0.0, 1.0] that the final
    response (substantive answer or abstention decision) is correct under the task rubric.
    """

    answer: str = Field(description="Concise answer to the question, or abstention explanation.")
    evidence: list[str] = Field(
        default_factory=list,
        description="Verbatim quotes copied directly from the supplied document.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability in [0, 1] that this response (answer or abstention) is correct.",
    )
    abstained: bool = Field(
        description="True if the question cannot be answered from the supplied document."
    )

    # Private trace storage so callers of answer_question() -> QAOutput can also inspect
    # raw_output (pre-gate), quote verification details, and LLM call metadata without
    # polluting the 4-field JSON schema sent to the LLM.
    _trace: Optional["QATrace"] = PrivateAttr(default=None)


class QuoteCheckResult(BaseModel):
    """Result of checking a single cited quote against the source document."""

    original_quote: str
    normalized_quote: str
    match_type: Literal["exact", "normalized", "fuzzy_candidate", "missing"]
    similarity_ratio: float = Field(ge=0.0, le=1.0)
    passed_gate: bool


class QATrace(BaseModel):
    """Full execution trace preserving both raw_output (pre-gate) and final_output (post-gate)."""

    prompt_version: str
    raw_output: QAOutput
    final_output: QAOutput
    gate_enabled: bool
    gate_triggered: bool
    gate_reason: str
    quote_checks: list[QuoteCheckResult] = Field(default_factory=list)
    fuzzy_candidate_count: int = 0
    error: Optional[str] = None
    llm_metadata: dict[str, Any] = Field(default_factory=dict)


class BaseCaseRecord(BaseModel):
    """Ground-truth base test case record in data/base_cases.jsonl."""

    id: str
    document_id: str
    question: str
    expected_answer: str
    expected_evidence: list[str]
    should_abstain: bool
    answer_type: Literal[
        "number_with_unit",
        "date",
        "named_entity",
        "list",
        "yes_no",
        "two_sentence_fact",
        "unanswerable",
    ]
    status: Literal["draft", "verified"] = "draft"
    verified_by: Optional[str] = None
    verified_on: Optional[str] = None
    notes: str = ""


class VariantRecord(BaseModel):
    """Adversarial variant test case record in data/variants.jsonl."""

    variant_id: str
    base_id: str
    document_id: str
    transformation: Literal[
        "paraphrase_hard",
        "distractor_near_miss",
        "distractor_irrelevant",
        "false_premise_numeric",
        "false_premise_fabricated",
        "source_conflict",
        "long_context",
        "format_edge",
        "prompt_injection",
        "empty_malformed",
    ]
    original_input: str
    modified_input: str
    modified_document: Optional[str] = None
    context_position: Optional[Literal["start", "middle", "end"]] = None
    false_premise_value: Optional[str] = None
    injected_target_value: Optional[str] = None
    conflict_values: Optional[list[str]] = None
    expected_behavior: str
    expected_answer: str
    expected_evidence: list[str] = Field(default_factory=list)
    should_abstain: bool
    status: Literal["draft", "verified"] = "draft"
    verified_by: Optional[str] = None
    verified_on: Optional[str] = None
    notes: str = ""


class SemanticJudgeOutput(BaseModel):
    """Structured output from the independent semantic correctness judge (Phase 4)."""

    is_correct: bool
    reason: str


class ClaimVerdictItem(BaseModel):
    """Single atomic factual claim and its entailment verdict against the source document."""

    claim_text: str
    verdict: Literal["supported", "contradicted", "unsupported"]
    evidence_quote: str = ""
    reason: str


class ClaimJudgeOutput(BaseModel):
    """Structured output from the Layer-3 independent claim hallucination judge (Phase 5)."""

    claims: list[ClaimVerdictItem] = Field(default_factory=list)


class AuditClaimRecord(BaseModel):
    """Human-audited atomic claim record in data/audit_claims.jsonl (Phase 5)."""

    claim_id: str
    base_id: str
    document_id: str
    slice: Literal[
        "verbatim_fact",
        "paraphrase",
        "unit_conversion",
        "multi_clause_fact",
        "negation_inversion",
        "wrong_entity_or_condition",
        "numeric_contradiction",
        "unanswerable_extrapolation",
        "unsupported_inference",
    ]
    question: str
    claim_text: str
    source_evidence: list[str] = Field(
        min_length=1,
        description="Verbatim quote(s) from the source document used to verify the label.",
    )
    human_label: Literal["supported", "contradicted", "unsupported"]
    human_notes: str
    needs_human_review: bool = False
    status: Literal["draft", "verified"] = "draft"
    verified_by: Optional[str] = None
    verified_on: Optional[str] = None
    detector_label: Optional[Literal["supported", "contradicted", "unsupported"]] = None
    detector_layer: Optional[str] = None
    detector_reason: Optional[str] = None


class RegressionCaseRecord(BaseModel):
    """Frozen or canonical-seed baseline test case record in data/regression_cases.jsonl (Phase 8)."""

    case_id: str
    base_id: str
    document_id: str
    category: str = "base"
    answer_type: Literal[
        "number_with_unit",
        "date",
        "named_entity",
        "list",
        "yes_no",
        "two_sentence_fact",
        "unanswerable",
    ]
    document: str
    question: str
    expected_answer: str
    expected_evidence: list[str] = Field(default_factory=list)
    should_abstain: bool
    false_premise_value: Optional[str] = None
    injected_target_value: Optional[str] = None
    conflict_values: Optional[list[str]] = None
    baseline_passed: bool = True
    baseline_abstained: bool = False
    baseline_confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    baseline_model_id: str = "groq/openai/gpt-oss-20b"
    baseline_prompt_version: str = "grounded_v1"
    dataset_version: str = "v1.0"
    status: Literal["inherited_from_base_case", "verified", "draft"] = "inherited_from_base_case"
    verification_source: Literal[
        "inherited_from_base_case", "independent_human_review", "unverified"
    ] = "inherited_from_base_case"
    source_base_status: Optional[Literal["verified", "draft"]] = "verified"
    source_base_verified_by: Optional[str] = None
    source_base_verified_on: Optional[str] = None
    verified_by: Optional[str] = None
    verified_on: Optional[str] = None
    baseline_source: Literal[
        "canonical_base_case_seed", "live_evaluation_run", "mock_evaluation_run"
    ] = "canonical_base_case_seed"
    notes: str = ""
