"""Unit tests for Phase 5 3-layer hallucination detector and Cohen's kappa audit (evaluation/hallucination.py)."""

import json

from evaluation.dataset import load_documents, load_jsonl_audit_claims
from evaluation.hallucination import (
    check_layer1_evidence_presence,
    check_layer2_entity_value_consistency,
    compute_cohens_kappa,
    compute_hallucination_rates,
    evaluate_human_audit,
    evaluate_output_hallucinations,
    run_audit_benchmark,
)
from src.schemas import QAOutput


def test_layer1_evidence_presence() -> None:
    """Layer 1 must distinguish exact/normalized present quotes from missing quotes."""
    doc = load_documents()["DOC-001"]
    quotes = [
        "In active telemetry acquisition mode at 20 °C, the controller draws a continuous supply current of 42 mA.",
        "This quote is completely fabricated and does not exist.",
    ]
    res = check_layer1_evidence_presence(quotes, doc)
    assert res["total_quotes"] == 2
    assert res["verified_quote_count"] == 1
    assert res["missing_quote_count"] == 1
    assert res["all_quotes_present"] is False


def test_layer2_conservative_entity_value_and_negation_checks() -> None:
    """Layer 2 must catch wrong-entity values, absent numbers, and negation inversions, and defer paraphrases."""
    doc = load_documents()["DOC-001"]

    # 1. Verbatim clause -> supported deterministically
    res_sup = check_layer2_entity_value_consistency(
        "the controller draws a continuous supply current of 42 mA.",
        doc,
        question="What is the active current of the Helios-X9?",
    )
    assert res_sup["verdict"] == "supported"
    assert res_sup["is_deterministic"] is True

    # 2. Wrong-entity value trap: 95 mA belongs to Helios-X7, not Helios-X9
    res_wrong_ent = check_layer2_entity_value_consistency(
        "The Helios-X9 draws 95 mA in active mode.",
        doc,
        question="What is the active current of the Helios-X9?",
    )
    assert res_wrong_ent["verdict"] == "contradicted"
    assert res_wrong_ent["deciding_layer"] == "layer2_deterministic"

    # 3. Absent numeric value for no-entity claim -> unsupported
    res_unsup = check_layer2_entity_value_consistency(
        "The shipping box weighs 312 kilograms.",
        doc,
    )
    assert res_unsup["verdict"] == "unsupported"

    # 4. Negation inversion on same entity and numbers -> contradicted
    res_neg = check_layer2_entity_value_consistency(
        "Under nominal operation, the Helios-X9 does not require a regulated core supply voltage of 3.3 V and 1.8 V.",
        doc,
    )
    assert res_neg["verdict"] == "contradicted"

    # 5. Unit conversion / paraphrase -> conservative 'needs_judge'
    res_judge = check_layer2_entity_value_consistency(
        "At room temperature, the Helios-X9 consumes 0.042 A of electric current.",
        doc,
    )
    assert res_judge["verdict"] == "needs_judge"


def test_evaluate_output_hallucinations_abstention_and_layer3_mock() -> None:
    """Abstentions have 0 hallucinations; needs_judge claims invoke Layer 3 LLM judge."""
    doc = load_documents()["DOC-001"]

    abstained_out = QAOutput(
        answer="The document does not state the unit price.",
        evidence=[],
        confidence=0.92,
        abstained=True,
    )
    res_abs = evaluate_output_hallucinations(
        case_id="BASE-006",
        category="base",
        document=doc,
        question="What is the unit price?",
        output=abstained_out,
    )
    assert res_abs["has_hallucination"] is False
    assert res_abs["total_claims"] == 0

    def _mock_claim_judge(**kwargs):  # type: ignore[no-untyped-def]
        return {
            "text": json.dumps({
                "claims": [
                    {
                        "claim_text": "The Helios-X9 consumes 0.042 A in active mode.",
                        "verdict": "supported",
                        "evidence_quote": "In active telemetry acquisition mode at 20 °C, the controller draws a continuous supply current of 42 mA.",
                        "reason": "0.042 A is an exact unit conversion of 42 mA.",
                    },
                    {
                        "claim_text": "It is manufactured in Munich.",
                        "verdict": "contradicted",
                        "evidence_quote": "All production wafers are fabricated exclusively at Vaelen MicroSystems Fab-4 in Dresden",
                        "reason": "Document states Dresden, not Munich.",
                    },
                ]
            }),
            "metadata": {"model_id": "gemini-3.5-flash-lite"},
        }

    mixed_out = QAOutput(
        answer="The Helios-X9 consumes 0.042 A in active mode. It is manufactured in Munich.",
        evidence=["In active telemetry acquisition mode at 20 °C, the controller draws a continuous supply current of 42 mA."],
        confidence=0.85,
        abstained=False,
    )
    res_mixed = evaluate_output_hallucinations(
        case_id="BASE-001",
        category="base",
        document=doc,
        question="What is the active current and manufacturing city?",
        output=mixed_out,
        llm_caller=_mock_claim_judge,
    )
    assert res_mixed["has_hallucination"] is True
    assert res_mixed["total_claims"] == 2
    assert res_mixed["supported_claims"] == 1
    assert res_mixed["contradicted_claims"] == 1
    assert res_mixed["unsupported_claims"] == 0

    rates = compute_hallucination_rates([res_abs, res_mixed])
    assert rates["output_level_all"]["numerator"] == 1
    assert rates["output_level_all"]["denominator"] == 2
    assert rates["output_level_answered_only"]["numerator"] == 1
    assert rates["output_level_answered_only"]["denominator"] == 1
    assert rates["claim_level_all"]["numerator"] == 1
    assert rates["claim_level_all"]["denominator"] == 2


def test_cohens_kappa_and_human_audit_hand_computed() -> None:
    """Verify Cohen's kappa and confusion matrix on a hand-computed 10-claim example."""
    # 10 items: 8 agree, 2 disagree
    audit_items = [
        {"claim_id": "C1", "slice": "clean", "human_label": "supported", "detector_label": "supported"},
        {"claim_id": "C2", "slice": "clean", "human_label": "supported", "detector_label": "supported"},
        {"claim_id": "C3", "slice": "clean", "human_label": "supported", "detector_label": "supported"},
        {"claim_id": "C4", "slice": "unit", "human_label": "supported", "detector_label": "unsupported"},
        {"claim_id": "C5", "slice": "negation", "human_label": "contradicted", "detector_label": "contradicted"},
        {"claim_id": "C6", "slice": "negation", "human_label": "contradicted", "detector_label": "contradicted"},
        {"claim_id": "C7", "slice": "negation", "human_label": "contradicted", "detector_label": "supported"},
        {"claim_id": "C8", "slice": "inference", "human_label": "unsupported", "detector_label": "unsupported"},
        {"claim_id": "C9", "slice": "inference", "human_label": "unsupported", "detector_label": "unsupported"},
        {"claim_id": "C10", "slice": "inference", "human_label": "unsupported", "detector_label": "unsupported"},
    ]
    # Human counts: supported=4, contradicted=3, unsupported=3
    # Detector counts: supported=4, contradicted=2, unsupported=4
    # Po = 8/10 = 0.80
    # Pe = (0.4*0.4) + (0.3*0.2) + (0.3*0.4) = 0.16 + 0.06 + 0.12 = 0.34
    # kappa = (0.80 - 0.34) / (1 - 0.34) = 0.46 / 0.66 = 0.696969... -> 0.697
    summary = evaluate_human_audit(audit_items)
    assert summary["exact_agreement"]["numerator"] == 8
    assert summary["exact_agreement"]["denominator"] == 10
    assert summary["cohens_kappa"]["value"] == 0.697
    assert summary["disagreement_count"] == 2
    assert compute_cohens_kappa(["supported", "contradicted"], ["supported", "contradicted"]) == 1.0


def test_audit_claims_benchmark_offline_and_allow_llm_judge_false() -> None:
    """Verify deterministic Layer-2 benchmark on the 45 audit claims and allow_llm_judge=False mode."""
    docs = load_documents()
    claims = load_jsonl_audit_claims()
    assert len(claims) == 45

    bench = run_audit_benchmark(claims, docs, allow_llm_judge=False)
    assert bench["total_audited_claims"] == 45
    assert bench["exact_agreement"]["numerator"] == 42
    assert bench["exact_agreement"]["denominator"] == 45
    assert bench["cohens_kappa"]["value"] == 0.9
    raw_res = bench["raw_layer2_resolution"]
    assert raw_res["deterministic_decided_count"] == 38
    assert raw_res["deterministic_decided_rate"] == 0.8444
    assert raw_res["deterministic_correct_count"] == 38
    assert raw_res["strict_unmapped_agreement"] == 0.8444
    assert raw_res["needs_judge_count"] == 7
    assert raw_res["needs_judge_rate"] == 0.1556
    assert raw_res["needs_judge_claim_ids"] == [
        "AUD-010",
        "AUD-011",
        "AUD-012",
        "AUD-039",
        "AUD-043",
        "AUD-044",
        "AUD-045",
    ]
    assert raw_res["fallback_inflated_claim_ids"] == [
        "AUD-039",
        "AUD-043",
        "AUD-044",
        "AUD-045",
    ]
    # Only the 3 unit_conversion boundary claims (AUD-010..AUD-012) differ when needs_judge -> unsupported offline
    disagreed_ids = [d["claim_id"] for d in bench["disagreements"]]
    assert disagreed_ids == ["AUD-010", "AUD-011", "AUD-012"]

    # Also verify evaluate_output_hallucinations with allow_llm_judge=False makes zero LLM calls
    # and does NOT count a deferred 'needs_judge' claim as a confirmed 'supported' claim.
    paraphrased_out = QAOutput(
        answer="At room temperature, the Helios-X9 consumes 0.042 A of electric current.",
        evidence=[
            "In active telemetry acquisition mode at 20 °C, the controller draws a continuous supply current of 42 mA."
        ],
        confidence=0.90,
        abstained=False,
    )
    offline_eval = evaluate_output_hallucinations(
        case_id="BASE-001",
        category="base",
        document=docs["DOC-001"],
        question="What is the active current?",
        output=paraphrased_out,
        allow_llm_judge=False,
    )
    assert offline_eval["has_hallucination"] is False
    assert offline_eval["supported_claims"] == 0
    assert offline_eval["needs_judge_claims"] == 1
    assert offline_eval["unresolved_needs_judge_claims"] == 1


def test_claim_splitting_honorifics_and_multi_numerical_evidence() -> None:
    """Regression test: claim splitting preserves honorific titles and Layer 2 avoids false numeric contradictions."""
    from evaluation.hallucination import split_into_atomic_claims_deterministic

    # 1. Honorific titles (Dr., Prof., etc.) must not split mid-name
    text_with_honorific = (
        "The certification was granted by Dr. Maren Lindqvist after testing. "
        "Prof. Eleanor Vance verified the second phase."
    )
    claims = split_into_atomic_claims_deterministic(text_with_honorific)
    assert len(claims) == 2
    assert "Dr. Maren Lindqvist" in claims[0]
    assert "Prof. Eleanor Vance" in claims[1]

    # 2. Multi-tier numerical document: tier-specific numbers must not be falsely contradicted by earlier tiers
    doc_multi_tier = (
        "Standard Merchant Tier: 120 RPM / 25 burst ceiling.\n"
        "Enterprise Clearing Tier: 900 RPM / 150 burst ceiling."
    )
    claim_enterprise = "The Enterprise Clearing Tier has a rate limit of 900 RPM and a burst ceiling of 150."
    res_enterprise = check_layer2_entity_value_consistency(claim_enterprise, doc_multi_tier)
    assert res_enterprise["verdict"] != "contradicted"

    # 3. Multi-condition numerical facts spanning sentences: should not be falsely contradicted
    doc_multi_cure = (
        "Under Section 8.2, cure period for material non-payment breach is 45 calendar days. "
        "Operational non-financial breaches require a 60 calendar day cure period."
    )
    claim_both_cures = "The cure period for non-payment is 45 days, while operational breach is 60 days."
    res_both_cures = check_layer2_entity_value_consistency(claim_both_cures, doc_multi_cure)
    assert res_both_cures["verdict"] != "contradicted"

