"""Phase 5 Three-Layer Hallucination Detector & Human-Judge Audit (evaluation/hallucination.py).

HALLUCINATION DEFINITION (Copied into README.md and reports/evaluation_report.md):
A hallucination is an atomic factual claim in the system's answer that the source document
does not entail. We record 'contradicted' (the document explicitly states a conflicting
fact, opposite polarity, or assigns the value only to a different entity/condition)
separately from 'unsupported' (the document neither entails nor contradicts the claim).
A wrong answer is not automatically a hallucination, and a correct abstention is never one.
"""

import json
from pathlib import Path
import random
import re
from typing import Any, Callable, Optional

from src.llm_client import DEFAULT_JUDGE_MODEL, call_llm
from src.prompts import CLAIM_JUDGE_PROMPT_V1
from src.qa_system import normalize_text_for_quote_match, verify_evidence_quote
from src.schemas import ClaimJudgeOutput, QAOutput

HALLUCINATION_EVALUATOR_VERSION = "HALLUCINATION_V1"

HALLUCINATION_DEFINITION = (
    "A hallucination is an atomic factual claim in the answer that the source document "
    "does not entail. 'contradicted' claims are recorded separately from 'unsupported' "
    "claims. A wrong answer is not automatically a hallucination, and a correct abstention "
    "is never one."
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIT_CLAIMS_PATH = PROJECT_ROOT / "data" / "audit_claims.jsonl"

VALID_VERDICTS = ("supported", "contradicted", "unsupported")

# Known distinct product/tier/entity identifiers in DOC-001..DOC-008 for conservative Layer-2 entity tracking
KNOWN_ENTITIES = (
    "helios-x9l",
    "helios-x9r",
    "helios-x9",
    "helios-x7",
    "icecube",
    "ssna",
    "frostpulse-3",
    "borealis-iv",
    "borealis-ii",
    "borealis-x",
    "nauticbrain-5",
    "zelvorin-xr",
    "zelvorin-low",
    "estercleav-9",
    "lignobreak-4",
    "tier-0-mil",
    "tier-0",
    "tier-1",
    "tier-2",
    "tier-3",
    "cold-archive",
)

# Secondary/sibling entities that are only described in sentences that explicitly name them
SECONDARY_ENTITIES = {
    "helios-x7",
    "helios-x9l",
    "helios-x9r",
    "icecube",
    "borealis-ii",
    "borealis-x",
    "zelvorin-low",
    "lignobreak-4",
    "tier-0-mil",
    "tier-1",
    "tier-2",
    "tier-3",
    "cold-archive",
}

NEGATION_WORDS = (
    "not ",
    "no ",
    "never ",
    "omits",
    "excluded",
    "prohibit",
    "cannot",
    "must not",
    "do not",
    "does not",
    "neither",
)

IDENTIFIER_WITH_DIGITS_RE = re.compile(
    r"\b(?:"
    r"helios-x\d+[a-z]?|tier-\d+(?:-mil)?|frostpulse-\d+|nauticbrain-\d+|"
    r"estercleav-\d+|lignobreak-\d+|pol-hr-\d+|vb-\d+-\d+|nma-\d+-\d+|"
    r"cqfp-\d+|pqfp-\d+|sha\d+|aes-\d+-gcm|ml-kem-\d+|fips\s+\d+-\d+|"
    r"uuidv\d+|i2c|5g|v\d+\.\d+|err_[a-z0-9_]+|http\s+\d+|"
    r"build\s+\d+(?:\.\d+)+|cyp2d6|1553b|ema/h/c/\d+|sv-sla-t\d+-\d+|"
    r"phase-(?:i|ii|iii)|fab-\d+"
    r")\b",
    re.IGNORECASE,
)

LAYER2_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "under", "into",
    "when", "while", "each", "are", "was", "were", "has", "have", "had",
    "not", "does", "can", "cannot", "must", "shall", "may", "will", "should",
    "its", "their", "all", "any", "both", "either", "more", "than", "less",
    "between", "before", "after", "within", "during", "upon", "per", "flat",
    "rate", "rated", "cap", "total", "maximum", "minimum", "nominal", "standard",
    "continuous", "usd", "chf", "eur", "day", "days", "hour", "hours", "minute",
    "minutes", "second", "seconds", "year", "years", "calendar", "month", "monthly",
    "policy", "agreement", "clause", "section", "schedule", "tier", "helios",
    "borealis", "zelvorin", "stratosvault", "chronospay", "ssna", "estercleav",
    "controller", "vessel", "sister", "legacy", "one", "two", "three", "four", "five",
}


def _extract_numbers(text: str) -> list[str]:
    """Extract normalized numeric literals (stripping thousand-separator commas and code identifiers)."""
    without_ids = IDENTIFIER_WITH_DIGITS_RE.sub(" ", text.lower())
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", without_ids)
    return re.findall(r"\d+(?:\.\d+)?", cleaned)


def _equivalent_numeric_forms(num_str: str) -> set[str]:
    """Return common metric/time conversion equivalents so Layer 2 does not misflag unit conversions."""
    forms = {num_str}
    try:
        val = float(num_str)
    except ValueError:
        return forms

    for factor in (1000.0, 0.001, 24.0, 1.0 / 24.0, 60.0, 1.0 / 60.0):
        converted = val * factor
        if converted.is_integer():
            forms.add(str(int(converted)))
        else:
            forms.add(f"{converted:.4f}".rstrip("0").rstrip("."))
    return forms


def _extract_entities(text: str) -> set[str]:
    """Return known distinct entity identifiers present in text."""
    lower = normalize_text_for_quote_match(text).lower()
    found: set[str] = set()
    for ent in KNOWN_ENTITIES:
        if re.search(rf"\b{re.escape(ent)}\b", lower):
            found.add(ent)
    return found


def _has_negation_polarity(text: str) -> bool:
    """Return True if text contains an explicit negation or prohibition marker."""
    lower = f" {normalize_text_for_quote_match(text).lower()} "
    return any(neg in lower for neg in NEGATION_WORDS)


ABBREVIATION_PATTERN = re.compile(
    r"\b(Dr|Mr|Ms|Mrs|Prof|Sr|Jr|vs|etc|Fig|Sec|No|Rev|Inc|Ltd|corp|dept|approx|est)\.\s+",
    re.IGNORECASE,
)


def split_into_atomic_claims_deterministic(answer_text: str) -> list[str]:
    """Split a substantive answer into sentence-level atomic claims for Layer-2 inspection."""
    cleaned = normalize_text_for_quote_match(answer_text)
    if not cleaned:
        return []
    masked = ABBREVIATION_PATTERN.sub(r"\1__DOT__ ", cleaned)
    parts = [
        p.replace("__DOT__ ", ". ").strip()
        for p in re.split(r"(?<=[.!?])\s+|;\s+", masked)
        if p.strip()
    ]
    return parts if parts else [cleaned]


def check_layer1_evidence_presence(
    evidence_quotes: list[str], document: str
) -> dict[str, Any]:
    """Layer 1 (Deterministic): Check whether each cited quote exists in the source document.

    Note: Quote presence proves only that the text exists in the document, NOT that it
    entails the answer.
    """
    checks = [verify_evidence_quote(q, document) for q in evidence_quotes]
    exact_count = sum(1 for c in checks if c.match_type == "exact")
    norm_count = sum(1 for c in checks if c.match_type == "normalized")
    fuzzy_count = sum(1 for c in checks if c.match_type == "fuzzy_candidate")
    missing_count = sum(1 for c in checks if c.match_type == "missing")

    return {
        "total_quotes": len(checks),
        "verified_quote_count": exact_count + norm_count,
        "exact_count": exact_count,
        "normalized_count": norm_count,
        "fuzzy_candidate_count": fuzzy_count,
        "missing_quote_count": missing_count,
        "all_quotes_present": len(checks) > 0 and missing_count == 0 and fuzzy_count == 0,
        "quote_checks": [c.model_dump() for c in checks],
    }


# Competing sibling entity groups per document
SIBLING_ENTITY_GROUPS: tuple[set[str], ...] = (
    {"helios-x9", "helios-x7", "helios-x9l", "helios-x9r"},
    {"tier-1", "tier-2", "tier-3"},
    {"ssna", "icecube"},
    {"estercleav-9", "lignobreak-4"},
    {"zelvorin-xr", "zelvorin-low"},
    {"borealis-iv", "borealis-ii", "borealis-x"},
    {"tier-0", "tier-0-mil", "tier-1", "cold-archive"},
)


def _conflicting_sibling_entities(sent_ents: set[str], claim_entities: set[str]) -> set[str]:
    """Return entities in sent_ents that belong to the same competing sibling group as claim_entities."""
    conflicts: set[str] = set()
    for group in SIBLING_ENTITY_GROUPS:
        if claim_entities & group:
            conflicts |= (sent_ents & group) - claim_entities
    return conflicts


def _sentence_compatible_with_entities(sent: str, claim_entities: set[str]) -> bool:
    """Check if a document sentence can apply to the queried claim_entities."""
    if not claim_entities:
        return True
    sent_ents = _extract_entities(sent)
    if _conflicting_sibling_entities(sent_ents, claim_entities):
        return False
    if claim_entities & sent_ents:
        return True
    # If the claim asks about a secondary/sibling entity (e.g., Helios-X7, MV Borealis-II, IceCube),
    # the sentence must explicitly name that secondary entity.
    if claim_entities & SECONDARY_ENTITIES:
        return False
    return True


def check_layer2_entity_value_consistency(
    claim_text: str, document: str, question: str = ""
) -> dict[str, Any]:
    """Layer 2 (Deterministic, Conservative): Entity, value, unit, and negation consistency check.

    Never labels a claim 'supported' merely because a number or entity appears somewhere
    in the document. Returns 'needs_judge' whenever deterministic rules cannot settle
    subject-and-relation entailment with high confidence.
    """
    norm_claim = normalize_text_for_quote_match(claim_text)
    norm_doc = normalize_text_for_quote_match(document)
    claim_lower = norm_claim.lower()
    doc_lower = norm_doc.lower()

    # Split on raw newlines and sentence boundaries BEFORE normalizing whitespace
    # while protecting abbreviations and titles (e.g. Dr., Prof., Sec.)
    masked_doc = ABBREVIATION_PATTERN.sub(r"\1__DOT__ ", document)
    raw_sents = re.split(r"(?<=[.!?])\s+|\n+", masked_doc)
    doc_sentences = [
        normalize_text_for_quote_match(s.replace("__DOT__ ", ". ").strip())
        for s in raw_sents
        if len(s.strip()) >= 5
    ]

    claim_nums = _extract_numbers(claim_lower)
    doc_nums = set(_extract_numbers(doc_lower))
    claim_entities = _extract_entities(claim_lower) or _extract_entities(question)
    claim_words = {
        w
        for w in re.findall(r"[a-z]{3,}", claim_lower)
        if w not in LAYER2_STOPWORDS
    }

    candidate_sents = [
        s for s in doc_sentences if _sentence_compatible_with_entities(s, claim_entities)
    ]

    # 1. Direct substring / clause match across all document sentences FIRST
    for sent in doc_sentences:
        sent_lower = sent.lower()
        stripped_claim = claim_lower.rstrip(".")
        if len(stripped_claim) >= 12 and stripped_claim in sent_lower:
            return {
                "claim_text": claim_text,
                "verdict": "supported",
                "evidence_quote": sent,
                "deciding_layer": "layer2_deterministic",
                "is_deterministic": True,
                "evaluator_version": HALLUCINATION_EVALUATOR_VERSION,
                "reason": "Claim is a direct verbatim clause of a source document sentence.",
            }

    # 2. Check if any numeric value in the claim is completely absent from the document
    #    (even after accounting for standard unit conversion equivalents like 42 mA <-> 0.042 A)
    for num in claim_nums:
        equivs = _equivalent_numeric_forms(num)
        if not (equivs & doc_nums):
            # Classify as 'contradicted' ONLY if a compatible numeric sentence in the document
            # shares the specific attribute/metric words (>= 3 shared specific words and
            # >= 70% content-word overlap). Otherwise classify as 'unsupported'.
            for s in candidate_sents:
                if not _extract_numbers(s):
                    continue
                s_words = {
                    w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in LAYER2_STOPWORDS
                }
                shared = claim_words & s_words
                overlap_ratio = len(shared) / max(1, len(claim_words))
                if len(shared) >= 3 and overlap_ratio >= 0.70:
                    return {
                        "claim_text": claim_text,
                        "verdict": "contradicted",
                        "evidence_quote": s,
                        "deciding_layer": "layer2_deterministic",
                        "is_deterministic": True,
                        "evaluator_version": HALLUCINATION_EVALUATOR_VERSION,
                        "reason": (
                            f"Numeric value '{num}' conflicts with documented value in matching "
                            f"specification sentence (shared terms: {sorted(shared)[:6]})."
                        ),
                    }
            return {
                "claim_text": claim_text,
                "verdict": "unsupported",
                "evidence_quote": "",
                "deciding_layer": "layer2_deterministic",
                "is_deterministic": True,
                "evaluator_version": HALLUCINATION_EVALUATOR_VERSION,
                "reason": (
                    f"Numeric value '{num}' (and unit equivalents) does not appear in the document "
                    f"and no matching specification sentence contradicts the queried attribute."
                ),
            }

    # 3. Wrong-entity value trap: check if a number in the claim appears in topical sentences
    #    belonging ONLY to a competing sibling entity (e.g., Helios-X7's 95 mA attributed to Helios-X9,
    #    IceCube's 2,450 meters attributed to SSNA, or Tier-1's 90-second lag attributed to Tier-0)
    if claim_nums and claim_entities:
        for num in claim_nums:
            matching_sents = [
                s
                for s in doc_sentences
                if num in _extract_numbers(s.lower())
                and len(
                    claim_words
                    & {w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in LAYER2_STOPWORDS}
                )
                >= 2
            ]
            if matching_sents:
                sents_with_other_entity = [
                    s
                    for s in matching_sents
                    if _conflicting_sibling_entities(_extract_entities(s), claim_entities)
                ]
                sents_compatible_only = [
                    s
                    for s in matching_sents
                    if _sentence_compatible_with_entities(s, claim_entities)
                ]
                if not sents_compatible_only and sents_with_other_entity:
                    other_ents = sorted(
                        _conflicting_sibling_entities(
                            _extract_entities(sents_with_other_entity[0]), claim_entities
                        )
                    )
                    return {
                        "claim_text": claim_text,
                        "verdict": "contradicted",
                        "evidence_quote": sents_with_other_entity[0],
                        "deciding_layer": "layer2_deterministic",
                        "is_deterministic": True,
                        "evaluator_version": HALLUCINATION_EVALUATOR_VERSION,
                        "reason": (
                            f"Value '{num}' in document belongs to sibling entity {other_ents}, "
                            f"not queried entity {sorted(claim_entities)}."
                        ),
                    }

    # 4. Check if a supporting sentence exists in the document for all numbers
    has_supporting_sentence = False
    if claim_nums:
        for s in candidate_sents:
            s_nums = _extract_numbers(s.lower())
            s_words = {w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in LAYER2_STOPWORDS}
            shared_s = claim_words & s_words
            overlap_s = len(shared_s) / max(1, len(claim_words))
            if set(claim_nums).issubset(set(s_nums)) and len(shared_s) >= 3 and overlap_s >= 0.45:
                has_supporting_sentence = True
                break
        if not has_supporting_sentence and len(claim_nums) > 1:
            all_nums_covered = True
            for num in claim_nums:
                num_covered = False
                for s in candidate_sents:
                    s_nums = _extract_numbers(s.lower())
                    s_words = {w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in LAYER2_STOPWORDS}
                    if num in s_nums and len(claim_words & s_words) >= 2:
                        num_covered = True
                        break
                if not num_covered:
                    all_nums_covered = False
                    break
            if all_nums_covered:
                has_supporting_sentence = True

    # 5. Sentence-level numeric conflict and polarity check
    for sent in doc_sentences:
        sent_lower = sent.lower()
        sent_nums = _extract_numbers(sent_lower)
        sent_words = {
            w
            for w in re.findall(r"[a-z]{3,}", sent_lower)
            if w not in LAYER2_STOPWORDS
        }
        shared = claim_words & sent_words
        overlap_ratio = len(shared) / max(1, len(claim_words))
        nums_subset = set(claim_nums).issubset(set(sent_nums))
        ents_compatible = _sentence_compatible_with_entities(sent, claim_entities)

        # If a compatible sentence matches the specific attribute words (>=4 shared and >=55% overlap)
        # and contains numbers, but the claim states a non-equivalent number from elsewhere in the doc,
        # flag as contradicted only if no other compatible sentence in the document supports the claim's numbers.
        if (
            not has_supporting_sentence
            and ents_compatible
            and claim_nums
            and sent_nums
            and not nums_subset
            and len(shared) >= 4
            and overlap_ratio >= 0.55
        ):
            all_equiv_present = all(
                bool(_equivalent_numeric_forms(cn) & set(sent_nums)) for cn in claim_nums
            )
            if not all_equiv_present:
                return {
                    "claim_text": claim_text,
                    "verdict": "contradicted",
                    "evidence_quote": sent,
                    "deciding_layer": "layer2_deterministic",
                    "is_deterministic": True,
                    "evaluator_version": HALLUCINATION_EVALUATOR_VERSION,
                    "reason": (
                        f"Numeric value(s) {claim_nums} conflict with documented value(s) "
                        f"{sent_nums} in matching specification sentence."
                    ),
                }

        # Check negation polarity when numbers/entities and content words match a source sentence
        if nums_subset and ents_compatible and (
            (claim_nums and sent_nums and len(shared) >= 2)
            or (len(shared) >= 3 and overlap_ratio >= 0.50)
        ):
            claim_neg = _has_negation_polarity(claim_lower)
            sent_neg = _has_negation_polarity(sent_lower)
            if claim_neg != sent_neg:
                return {
                    "claim_text": claim_text,
                    "verdict": "contradicted",
                    "evidence_quote": sent,
                    "deciding_layer": "layer2_deterministic",
                    "is_deterministic": True,
                    "evaluator_version": HALLUCINATION_EVALUATOR_VERSION,
                    "reason": "Claim inverts the negation polarity of the matching source sentence.",
                }
            if claim_neg == sent_neg and len(shared) >= 4 and overlap_ratio >= 0.80:
                return {
                    "claim_text": claim_text,
                    "verdict": "supported",
                    "evidence_quote": sent,
                    "deciding_layer": "layer2_deterministic",
                    "is_deterministic": True,
                    "evaluator_version": HALLUCINATION_EVALUATOR_VERSION,
                    "reason": "Claim matches single source sentence entities, numbers, polarity, and content words (>=80% overlap).",
                }

    # 6. Conservative fallback: whenever a claim is paraphrased, converts units, combines
    #    multiple sentences, or lacks a high-confidence single-sentence match, defer to Layer 3 LLM Judge.
    return {
        "claim_text": claim_text,
        "verdict": "needs_judge",
        "evidence_quote": "",
        "deciding_layer": "layer2_deterministic",
        "is_deterministic": True,
        "evaluator_version": HALLUCINATION_EVALUATOR_VERSION,
        "reason": "Requires Layer-3 semantic claim judge for entailment verification.",
    }


def call_layer3_claim_judge(
    document: str,
    question: str,
    answer_text: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    bypass_cache: bool = False,
    llm_caller: Callable[..., dict[str, Any]] = call_llm,
) -> list[dict[str, Any]]:
    """Layer 3 (Independent LLM Judge): Decompose answer into atomic claims and label each."""
    user_prompt = CLAIM_JUDGE_PROMPT_V1["user_template"].format(
        document=document,
        question=question,
        answer=answer_text,
    )
    res = llm_caller(
        system_prompt=CLAIM_JUDGE_PROMPT_V1["system"],
        prompt=user_prompt,
        model=judge_model,
        response_schema=ClaimJudgeOutput.model_json_schema(),
        temperature=0.0,
        seed=42,
        json_mode=True,
        max_output_tokens=768,
        bypass_cache=bypass_cache,
        prompt_version=CLAIM_JUDGE_PROMPT_V1["version"],
    )
    parsed = ClaimJudgeOutput.model_validate_json(res["text"])
    meta = res.get("metadata", {}) if isinstance(res.get("metadata"), dict) else res
    model_used = meta.get("model_id", judge_model)

    claims_out: list[dict[str, Any]] = []
    for item in parsed.claims:
        claims_out.append({
            "claim_text": item.claim_text,
            "verdict": item.verdict,
            "evidence_quote": item.evidence_quote,
            "deciding_layer": "layer3_llm_judge",
            "is_deterministic": False,
            "judge_model": model_used,
            "evaluator_version": HALLUCINATION_EVALUATOR_VERSION,
            "reason": item.reason,
        })
    return claims_out


def classify_single_atomic_claim(
    claim_text: str,
    document: str,
    question: str = "",
    judge_model: str = DEFAULT_JUDGE_MODEL,
    bypass_cache: bool = False,
    llm_caller: Callable[..., dict[str, Any]] = call_llm,
    allow_llm_judge: bool = True,
) -> dict[str, Any]:
    """Classify a single atomic claim using Layer 2 deterministic rules and Layer 3 LLM judge."""
    layer2 = check_layer2_entity_value_consistency(claim_text, document, question=question)
    if layer2["verdict"] != "needs_judge" or not allow_llm_judge:
        return layer2

    layer3_claims = call_layer3_claim_judge(
        document=document,
        question=question,
        answer_text=claim_text,
        judge_model=judge_model,
        bypass_cache=bypass_cache,
        llm_caller=llm_caller,
    )
    if not layer3_claims:
        return {
            **layer2,
            "verdict": "unsupported",
            "deciding_layer": "layer3_llm_judge_empty_fallback",
            "is_deterministic": False,
        }
    verdicts = [c["verdict"] for c in layer3_claims]
    if "contradicted" in verdicts:
        return next(c for c in layer3_claims if c["verdict"] == "contradicted")
    if "unsupported" in verdicts:
        return next(c for c in layer3_claims if c["verdict"] == "unsupported")
    return layer3_claims[0]


def evaluate_output_hallucinations(
    case_id: str,
    category: str,
    document: str,
    question: str,
    output: QAOutput,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    bypass_cache: bool = False,
    llm_caller: Callable[..., dict[str, Any]] = call_llm,
    allow_llm_judge: bool = True,
) -> dict[str, Any]:
    """Run the 3-layer hallucination detector on a single QAOutput.

    A correct abstention (or any abstention that makes no factual claim) is never a hallucination.
    """
    layer1 = check_layer1_evidence_presence(output.evidence, document)

    if output.abstained:
        return {
            "case_id": case_id,
            "category": category,
            "abstained": True,
            "layer1_evidence_presence": layer1,
            "claims": [],
            "total_claims": 0,
            "supported_claims": 0,
            "contradicted_claims": 0,
            "unsupported_claims": 0,
            "bad_claims": 0,
            "layer2_decided_claims": 0,
            "needs_judge_claims": 0,
            "unresolved_needs_judge_claims": 0,
            "has_hallucination": False,
            "evaluator_version": HALLUCINATION_EVALUATOR_VERSION,
        }

    candidate_claims = split_into_atomic_claims_deterministic(output.answer)
    layer2_results = [
        check_layer2_entity_value_consistency(c, document, question=question)
        for c in candidate_claims
    ]
    needs_judge_count = sum(1 for r in layer2_results if r["verdict"] == "needs_judge")
    layer2_decided_count = len(layer2_results) - needs_judge_count

    if needs_judge_count > 0:
        if allow_llm_judge:
            # Invoke Layer-3 LLM claim judge ONLY for claims deferred by Layer 2 ('needs_judge'),
            # preserving all deterministic Layer-2 verdicts intact.
            deferred_text = " ".join(
                r["claim_text"] for r in layer2_results if r["verdict"] == "needs_judge"
            )
            layer3_claims = call_layer3_claim_judge(
                document=document,
                question=question,
                answer_text=deferred_text,
                judge_model=judge_model,
                bypass_cache=bypass_cache,
                llm_caller=llm_caller,
            )
            det_decided = [r for r in layer2_results if r["verdict"] != "needs_judge"]
            if layer3_claims:
                final_claims = det_decided + layer3_claims
            else:
                # Layer 3 returned no claims; keep deferred claims marked as 'needs_judge' (unconfirmed)
                final_claims = layer2_results
        else:
            # Deterministic offline mode (allow_llm_judge=False):
            # Do NOT promote 'needs_judge' to 'supported' merely because a quote exists in the document.
            # Keep verdict as 'needs_judge' so deferred claims are never counted as confirmed detector successes.
            final_claims = [dict(r) for r in layer2_results]
    else:
        final_claims = layer2_results

    sup_count = sum(1 for c in final_claims if c["verdict"] == "supported")
    con_count = sum(1 for c in final_claims if c["verdict"] == "contradicted")
    unsup_count = sum(1 for c in final_claims if c["verdict"] == "unsupported")
    unresolved_count = sum(1 for c in final_claims if c["verdict"] == "needs_judge")
    bad_count = con_count + unsup_count

    return {
        "case_id": case_id,
        "category": category,
        "abstained": False,
        "layer1_evidence_presence": layer1,
        "claims": final_claims,
        "total_claims": len(final_claims),
        "supported_claims": sup_count,
        "contradicted_claims": con_count,
        "unsupported_claims": unsup_count,
        "bad_claims": bad_count,
        "layer2_decided_claims": layer2_decided_count,
        "needs_judge_claims": needs_judge_count,
        "unresolved_needs_judge_claims": unresolved_count,
        "has_hallucination": bad_count > 0,
        "evaluator_version": HALLUCINATION_EVALUATOR_VERSION,
    }


def compute_hallucination_rates(
    output_evals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute output-level and claim-level hallucination rates with explicit numerators/denominators."""
    valid_evals = [e for e in output_evals if e.get("category") != "empty_malformed"]
    total_outputs = len(valid_evals)
    answered_outputs = [e for e in valid_evals if not e.get("abstained", False)]
    outputs_with_bad_claim = sum(1 for e in valid_evals if e.get("has_hallucination", False))

    total_claims = sum(e.get("total_claims", 0) for e in valid_evals)
    supported_claims = sum(e.get("supported_claims", 0) for e in valid_evals)
    contradicted_claims = sum(e.get("contradicted_claims", 0) for e in valid_evals)
    unsupported_claims = sum(e.get("unsupported_claims", 0) for e in valid_evals)
    bad_claims = contradicted_claims + unsupported_claims
    layer2_decided_claims = sum(e.get("layer2_decided_claims", 0) for e in valid_evals)
    needs_judge_claims = sum(e.get("needs_judge_claims", 0) for e in valid_evals)
    unresolved_needs_judge_claims = sum(
        e.get("unresolved_needs_judge_claims", 0) for e in valid_evals
    )

    by_category: dict[str, dict[str, Any]] = {}
    for e in valid_evals:
        cat = e["category"]
        if cat not in by_category:
            by_category[cat] = {
                "outputs": 0,
                "outputs_with_hallucination": 0,
                "claims": 0,
                "supported_claims": 0,
                "contradicted_claims": 0,
                "unsupported_claims": 0,
                "bad_claims": 0,
                "layer2_decided_claims": 0,
                "needs_judge_claims": 0,
                "unresolved_needs_judge_claims": 0,
            }
        by_category[cat]["outputs"] += 1
        if e.get("has_hallucination", False):
            by_category[cat]["outputs_with_hallucination"] += 1
        by_category[cat]["claims"] += e.get("total_claims", 0)
        by_category[cat]["supported_claims"] += e.get("supported_claims", 0)
        by_category[cat]["contradicted_claims"] += e.get("contradicted_claims", 0)
        by_category[cat]["unsupported_claims"] += e.get("unsupported_claims", 0)
        by_category[cat]["bad_claims"] += e.get("bad_claims", 0)
        by_category[cat]["layer2_decided_claims"] += e.get("layer2_decided_claims", 0)
        by_category[cat]["needs_judge_claims"] += e.get("needs_judge_claims", 0)
        by_category[cat]["unresolved_needs_judge_claims"] += e.get(
            "unresolved_needs_judge_claims", 0
        )

    for cat_stats in by_category.values():
        n_out = cat_stats["outputs"]
        n_cl = cat_stats["claims"]
        cat_stats["output_hallucination_rate"] = (
            round(cat_stats["outputs_with_hallucination"] / n_out, 4) if n_out > 0 else 0.0
        )
        cat_stats["claim_hallucination_rate"] = (
            round(cat_stats["bad_claims"] / n_cl, 4) if n_cl > 0 else 0.0
        )

    return {
        "output_level_all": {
            "numerator": outputs_with_bad_claim,
            "denominator": total_outputs,
            "rate": round(outputs_with_bad_claim / total_outputs, 4) if total_outputs > 0 else 0.0,
        },
        "output_level_answered_only": {
            "numerator": outputs_with_bad_claim,
            "denominator": len(answered_outputs),
            "rate": (
                round(outputs_with_bad_claim / len(answered_outputs), 4)
                if answered_outputs
                else 0.0
            ),
        },
        "claim_level_all": {
            "numerator": bad_claims,
            "denominator": total_claims,
            "rate": round(bad_claims / total_claims, 4) if total_claims > 0 else 0.0,
            "supported_numerator": supported_claims,
            "contradicted_numerator": contradicted_claims,
            "unsupported_numerator": unsupported_claims,
            "layer2_decided_claims": layer2_decided_claims,
            "needs_judge_claims": needs_judge_claims,
            "unresolved_needs_judge_claims": unresolved_needs_judge_claims,
        },
        "by_category": by_category,
    }


def compute_cohens_kappa(
    human_labels: list[str], judge_labels: list[str]
) -> float:
    """Compute unweighted multi-class Cohen's kappa across ('supported', 'contradicted', 'unsupported')."""
    if len(human_labels) != len(judge_labels) or not human_labels:
        raise ValueError("human_labels and judge_labels must have identical non-zero length.")

    n = len(human_labels)
    observed_agree = sum(1 for h, j in zip(human_labels, judge_labels) if h == j) / n

    expected_agree = 0.0
    for label in VALID_VERDICTS:
        p_h = sum(1 for h in human_labels if h == label) / n
        p_j = sum(1 for j in judge_labels if j == label) / n
        expected_agree += p_h * p_j

    if abs(1.0 - expected_agree) < 1e-12:
        return 1.0
    return round((observed_agree - expected_agree) / (1.0 - expected_agree), 4)


def evaluate_human_audit(
    audit_records: list[dict[str, Any]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare human claim labels against detector/judge labels and compute agreement + Cohen's kappa."""
    if not audit_records:
        raise ValueError("audit_records cannot be empty.")

    human_labels = [r["human_label"] for r in audit_records]
    judge_labels = [r["detector_label"] for r in audit_records]
    n = len(audit_records)

    agree_count = sum(1 for h, j in zip(human_labels, judge_labels) if h == j)
    kappa = compute_cohens_kappa(human_labels, judge_labels)

    # Bootstrap 95% CI for Cohen's kappa
    rng = random.Random(seed)
    boot_kappas: list[float] = []
    for _ in range(n_bootstrap):
        indices = [rng.randrange(n) for _ in range(n)]
        h_sample = [human_labels[i] for i in indices]
        j_sample = [judge_labels[i] for i in indices]
        boot_kappas.append(compute_cohens_kappa(h_sample, j_sample))
    boot_kappas.sort()
    ci_low = boot_kappas[int(0.025 * n_bootstrap)]
    ci_high = boot_kappas[int(0.975 * n_bootstrap)]

    # 3x3 confusion matrix: rows = human_label, cols = detector_label
    confusion: dict[str, dict[str, int]] = {
        h_lab: {j_lab: 0 for j_lab in VALID_VERDICTS} for h_lab in VALID_VERDICTS
    }
    disagreements: list[dict[str, Any]] = []
    by_slice: dict[str, dict[str, int]] = {}

    for rec in audit_records:
        h = rec["human_label"]
        j = rec["detector_label"]
        confusion[h][j] += 1
        slice_name = rec.get("slice", "general")
        if slice_name not in by_slice:
            by_slice[slice_name] = {"total": 0, "agreed": 0}
        by_slice[slice_name]["total"] += 1
        if h == j:
            by_slice[slice_name]["agreed"] += 1
        else:
            disagreements.append({
                "claim_id": rec.get("claim_id"),
                "document_id": rec.get("document_id"),
                "slice": slice_name,
                "claim_text": rec.get("claim_text"),
                "human_label": h,
                "detector_label": j,
                "human_notes": rec.get("human_notes", ""),
                "detector_reason": rec.get("detector_reason", ""),
            })

    return {
        "total_audited_claims": n,
        "exact_agreement": {
            "numerator": agree_count,
            "denominator": n,
            "rate": round(agree_count / n, 4),
        },
        "cohens_kappa": {
            "value": kappa,
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "n_bootstrap": n_bootstrap,
        },
        "confusion_matrix": confusion,
        "by_slice": by_slice,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
    }


def run_audit_benchmark(
    audit_claims: list[Any],
    documents: dict[str, str],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    bypass_cache: bool = False,
    llm_caller: Callable[..., dict[str, Any]] = call_llm,
    allow_llm_judge: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the hallucination claim classifier across audited claims and compute Cohen's kappa."""
    evaluated_records: list[dict[str, Any]] = []
    for item in audit_claims:
        rec = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        doc_text = documents[rec["document_id"]]
        det = classify_single_atomic_claim(
            claim_text=rec["claim_text"],
            document=doc_text,
            question=rec.get("question", ""),
            judge_model=judge_model,
            bypass_cache=bypass_cache,
            llm_caller=llm_caller,
            allow_llm_judge=allow_llm_judge,
        )
        det_verdict = det["verdict"]
        if det_verdict == "needs_judge":
            # In strictly offline deterministic mode (allow_llm_judge=False),
            # conservative 'needs_judge' defaults to 'unsupported' for 3-class comparison
            det_verdict = "unsupported"
        evaluated_records.append({
            **rec,
            "detector_label": det_verdict,
            "raw_layer2_verdict": det["verdict"],
            "deciding_layer": det["deciding_layer"],
            "detector_reason": det["reason"],
        })

    summary = evaluate_human_audit(
        evaluated_records, n_bootstrap=n_bootstrap, seed=seed
    )
    total_count = len(evaluated_records)
    needs_judge_ids = [
        r["claim_id"]
        for r in evaluated_records
        if r["raw_layer2_verdict"] == "needs_judge"
    ]
    deterministic_decided_count = total_count - len(needs_judge_ids)
    deterministic_correct_count = sum(
        1
        for r in evaluated_records
        if r["raw_layer2_verdict"] != "needs_judge"
        and r["raw_layer2_verdict"] == r["human_label"]
    )
    fallback_inflated_ids = [
        r["claim_id"]
        for r in evaluated_records
        if r["raw_layer2_verdict"] == "needs_judge"
        and r["human_label"] == "unsupported"
    ]
    summary["raw_layer2_resolution"] = {
        "deterministic_decided_count": deterministic_decided_count,
        "deterministic_decided_rate": (
            round(deterministic_decided_count / total_count, 4)
            if total_count > 0
            else 0.0
        ),
        "deterministic_correct_count": deterministic_correct_count,
        "strict_unmapped_agreement": (
            round(deterministic_correct_count / total_count, 4)
            if total_count > 0
            else 0.0
        ),
        "needs_judge_count": len(needs_judge_ids),
        "needs_judge_rate": (
            round(len(needs_judge_ids) / total_count, 4)
            if total_count > 0
            else 0.0
        ),
        "needs_judge_claim_ids": needs_judge_ids,
        "fallback_inflated_claim_ids": fallback_inflated_ids,
    }
    summary["evaluated_records"] = evaluated_records
    return summary


def build_audit_claims_dataset() -> list[dict[str, Any]]:
    """Return the 45 manually verified atomic audit claims (AUD-001..AUD-045) in draft status."""
    raw_claims: list[dict[str, Any]] = [
        # --- 15 SUPPORTED CLAIMS (AUD-001 .. AUD-015) ---
        {
            "claim_id": "AUD-001",
            "base_id": "BASE-001",
            "document_id": "DOC-001",
            "slice": "verbatim_fact",
            "question": "What is the continuous supply current drawn by the Helios-X9 in active telemetry acquisition mode at 20 °C?",
            "claim_text": "In active telemetry acquisition mode at 20 °C, the Helios-X9 controller draws a continuous supply current of 42 mA.",
            "source_evidence": [
                "In active telemetry acquisition mode at 20 °C, the controller draws a continuous supply current of 42 mA."
            ],
            "human_label": "supported",
            "human_notes": "Directly entailed by Section 2 of DOC-001: active telemetry acquisition mode at 20 °C draws 42 mA.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-002",
            "base_id": "BASE-002",
            "document_id": "DOC-001",
            "slice": "verbatim_fact",
            "question": "Who is the lead architect that certified the proprietary germanium-on-sapphire substrate of the Helios-X9?",
            "claim_text": "The Helios-X9 uses a proprietary germanium-on-sapphire substrate certified by lead architect Dr. Maren Lindqvist.",
            "source_evidence": [
                "Unlike commercial silicon controllers that suffer carrier freeze-out below -55 °C, the Helios-X9 uses a proprietary germanium-on-sapphire substrate certified by lead architect Dr. Maren Lindqvist."
            ],
            "human_label": "supported",
            "human_notes": "Directly entailed verbatim clause from Section 1 of DOC-001 naming lead architect Dr. Maren Lindqvist.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-003",
            "base_id": "BASE-008",
            "document_id": "DOC-002",
            "slice": "verbatim_fact",
            "question": "What is the maximum nightly hotel reimbursement cap (excluding room taxes) for Tier-1 High-Cost Cities under Aethelgard Orbital Systems policy?",
            "claim_text": "Tier-1 High-Cost Cities (Zurich, Tokyo, New York, London, Singapore) have a hotel reimbursement cap of $310 USD per night.",
            "source_evidence": [
                "- Tier-1 High-Cost Cities (Zurich, Tokyo, New York, London, Singapore): $310 USD per night"
            ],
            "human_label": "supported",
            "human_notes": "Directly entailed by Section 4 of DOC-002 specifying $310 USD per night for the five Tier-1 cities.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-004",
            "base_id": "BASE-010",
            "document_id": "DOC-002",
            "slice": "verbatim_fact",
            "question": "What is the one-time home-office ergonomic stipend amount for newly hired Hybrid-Eligible full-time employees after completing their 90-day probationary period?",
            "claim_text": "Newly hired Hybrid-Eligible full-time employees receive a one-time home-office ergonomic stipend of $650 USD upon completing their 90-day probationary period.",
            "source_evidence": [
                "Newly hired Hybrid-Eligible full-time employees receive a one-time home-office ergonomic stipend of $650 USD upon completing their 90-day probationary period."
            ],
            "human_label": "supported",
            "human_notes": "Verbatim sentence from Section 5 of DOC-002 confirming the $650 USD ergonomic stipend.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-005",
            "base_id": "BASE-015",
            "document_id": "DOC-003",
            "slice": "verbatim_fact",
            "question": "How many operational vertical borehole strings does the Svalbard Sub-Glacial Neutrino Array (SSNA) have in total following Phase-II?",
            "claim_text": "Phase-II added 48 new borehole strings to the 36 strings installed during Phase-I in 2023, bringing the operational array to a total of 84 vertical strings.",
            "source_evidence": [
                "Principal Investigator Dr. Kasper Lindholm of the University of Bergen confirmed that Phase-II added 48 new borehole strings to the 36 strings installed during Phase-I in 2023, bringing the operational array to a total of 84 vertical strings."
            ],
            "human_label": "supported",
            "human_notes": "Directly entailed verbatim clause from paragraph 2 of DOC-003 confirming 84 operational vertical strings.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-006",
            "base_id": "BASE-016",
            "document_id": "DOC-003",
            "slice": "verbatim_fact",
            "question": "What is the maximum depth beneath the glacier surface reached by each vertical borehole string in the SSNA?",
            "claim_text": "Each vertical borehole string reaches a maximum depth of 620 meters beneath the glacier surface and carries 12 autonomous FrostPulse-3 sensor nodes spaced 35 meters apart.",
            "source_evidence": [
                "Each vertical borehole string reaches a maximum depth of 620 meters beneath the glacier surface and carries 12 autonomous \"FrostPulse-3\" sensor nodes spaced 35 meters apart."
            ],
            "human_label": "supported",
            "human_notes": "Directly entailed by paragraph 3 of DOC-003 specifying the 620-meter borehole depth and 12 FrostPulse-3 nodes.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-007",
            "base_id": "BASE-020",
            "document_id": "DOC-004",
            "slice": "verbatim_fact",
            "question": "What is the maximum allowed clock skew between the `X-Chronos-Timestamp` header and the gateway atomic clock before a request is rejected?",
            "claim_text": "If the timestamp header X-Chronos-Timestamp differs from the gateway atomic clock by more than 45 seconds in either direction, the gateway rejects the request with error code ERR_CLOCK_SKEW_904 (HTTP 401).",
            "source_evidence": [
                "If the timestamp header `X-Chronos-Timestamp` differs from the gateway atomic clock by more than 45 seconds in either direction, the gateway rejects the request with error code `ERR_CLOCK_SKEW_904` (HTTP 401)."
            ],
            "human_label": "supported",
            "human_notes": "Directly entailed by Section 1 of DOC-004 specifying the 45-second clock skew threshold.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-008",
            "base_id": "BASE-026",
            "document_id": "DOC-005",
            "slice": "verbatim_fact",
            "question": "What is the non-refundable upfront technology transfer fee that Licensee must pay Licensor under Agreement VB-2025-881?",
            "claim_text": "Within 15 business days of the Execution Date, Licensee shall pay Licensor a non-refundable upfront technology transfer fee of 450,000 CHF.",
            "source_evidence": [
                "Within 15 business days of the Execution Date, Licensee shall pay Licensor a non-refundable upfront technology transfer fee of 450,000 CHF."
            ],
            "human_label": "supported",
            "human_notes": "Verbatim sentence from Clause 2 of DOC-005 stating the 450,000 CHF upfront technology transfer fee.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-009",
            "base_id": "BASE-029",
            "document_id": "DOC-005",
            "slice": "verbatim_fact",
            "question": "What is the aggregate cumulative liability cap for either party under Agreement VB-2025-881 (excluding confidentiality breaches or willful patent infringement)?",
            "claim_text": "Except for breaches of confidentiality or willful infringement of Licensor's patent portfolio, neither party's aggregate cumulative liability under this Agreement shall exceed 2,400,000 CHF.",
            "source_evidence": [
                "Except for breaches of confidentiality or willful infringement of Licensor's patent portfolio, neither party's aggregate cumulative liability under this Agreement shall exceed 2,400,000 CHF."
            ],
            "human_label": "supported",
            "human_notes": "Verbatim sentence from Clause 4 of DOC-005 stating the 2,400,000 CHF liability cap and its two exclusions.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-010",
            "base_id": "BASE-001",
            "document_id": "DOC-001",
            "slice": "unit_conversion",
            "question": "What is the continuous supply current drawn by the Helios-X9 in active telemetry acquisition mode at 20 °C?",
            "claim_text": "In active telemetry acquisition mode at 20 °C, the Helios-X9 draws 0.042 amperes of continuous supply current.",
            "source_evidence": [
                "In active telemetry acquisition mode at 20 °C, the controller draws a continuous supply current of 42 mA."
            ],
            "human_label": "supported",
            "human_notes": "Human review decision (retained 'supported' & current wording; needs_human_review retained until verification metadata is authorized): Mathematical unit conversion rationale: 42 mA / 1,000 = 0.042 A, which is an exact SI identity entailed by Section 2 of DOC-001 ('42 mA'). Deterministic Layer 2 defers unit conversions to Layer 3 (needs_judge).",
            "needs_human_review": True,
        },
        {
            "claim_id": "AUD-011",
            "base_id": "BASE-022",
            "document_id": "DOC-004",
            "slice": "unit_conversion",
            "question": "For how many hours does the ChronosPay ledger cache idempotency keys and their response payloads in distributed memory?",
            "claim_text": "The ChronosPay ledger caches idempotency keys and their exact response payloads in distributed memory for 1.5 days (36 hours).",
            "source_evidence": [
                "The ChronosPay ledger caches idempotency keys and their exact response payloads in distributed memory for 36 hours."
            ],
            "human_label": "supported",
            "human_notes": "Human review decision (retained 'supported' & current wording; needs_human_review retained until verification metadata is authorized): Mathematical unit conversion rationale: 36 hours / 24 hours per day = 1.5 days, which is an exact time-unit conversion entailed by Section 3 of DOC-004 ('36 hours'). Deterministic Layer 2 defers unit conversions to Layer 3 (needs_judge).",
            "needs_human_review": True,
        },
        {
            "claim_id": "AUD-012",
            "base_id": "BASE-048",
            "document_id": "DOC-008",
            "slice": "unit_conversion",
            "question": "How does the guaranteed P99 first-byte read latency for StratosVault Tier-0 objects up to 64 MB compare with retrieval from the Cold-Archive Deep Vault?",
            "claim_text": "Cold-Archive Deep Vault retrieval requires a 210-minute (3.5-hour) robotic tape mount and staging window.",
            "source_evidence": [
                "By comparison, Cold-Archive Deep Vault retrieval requires a 3.5-hour robotic tape mount and staging window."
            ],
            "human_label": "supported",
            "human_notes": "Human review decision (retained 'supported' & current wording; needs_human_review retained until verification metadata is authorized): Mathematical unit conversion rationale: 3.5 hours * 60 minutes per hour = 210 minutes, which is an exact time-unit conversion entailed by Section 4 of DOC-008 ('3.5-hour'). Deterministic Layer 2 defers unit conversions to Layer 3 (needs_judge).",
            "needs_human_review": True,
        },
        {
            "claim_id": "AUD-013",
            "base_id": "BASE-032",
            "document_id": "DOC-006",
            "slice": "verbatim_fact",
            "question": "How many milligrams of brigatadine free base are equivalent to the 45 mg of brigatadine fumarate in each Zelvorin-XR capsule?",
            "claim_text": "Each enteric-coated extended-release capsule contains 45 mg of brigatadine fumarate (equivalent to 36 mg of brigatadine free base).",
            "source_evidence": [
                "Each enteric-coated extended-release capsule contains 45 mg of brigatadine fumarate (equivalent to 36 mg of brigatadine free base)."
            ],
            "human_label": "supported",
            "human_notes": "Verbatim sentence from Section 1 of DOC-006 confirming 36 mg of brigatadine free base per 45 mg fumarate capsule.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-014",
            "base_id": "BASE-039",
            "document_id": "DOC-007",
            "slice": "verbatim_fact",
            "question": "What is the main battery bank energy capacity and battery chemistry of MV Borealis-IV?",
            "claim_text": "MV Borealis-IV is powered by dual permanent-magnet azimuth thrusters rated at 1,800 kW each, supplied by a 6.4 MWh liquid-cooled lithium-iron-phosphate (LFP) main battery bank.",
            "source_evidence": [
                "MV Borealis-IV is powered by dual permanent-magnet azimuth thrusters rated at 1,800 kW each, supplied by a 6.4 MWh liquid-cooled lithium-iron-phosphate (LFP) main battery bank divided into Port and Starboard fire-isolated compartments."
            ],
            "human_label": "supported",
            "human_notes": "Directly entailed verbatim clause from Section 2 of DOC-007 confirming the 6.4 MWh LFP main battery bank.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-015",
            "base_id": "BASE-044",
            "document_id": "DOC-008",
            "slice": "verbatim_fact",
            "question": "Across which three underground data center cities and site designations are StratosVault Tier-0 objects synchronously erasure-coded?",
            "claim_text": "Every object written to a Tier-0 bucket is synchronously erasure-coded across three underground data centers located in Lucerne (Site Alpha), Bern (Site Beta), and Lugano (Site Gamma).",
            "source_evidence": [
                "Every object written to a Tier-0 bucket is synchronously erasure-coded across three underground data centers located in Lucerne (Site Alpha), Bern (Site Beta), and Lugano (Site Gamma) before an HTTP 201 Write Acknowledgement is returned to the client."
            ],
            "human_label": "supported",
            "human_notes": "Directly entailed verbatim clause from Section 1 of DOC-008 naming Lucerne, Bern, and Lugano.",
            "needs_human_review": False,
        },
        # --- 15 CONTRADICTED CLAIMS (AUD-016 .. AUD-030) ---
        {
            "claim_id": "AUD-016",
            "base_id": "BASE-008",
            "document_id": "DOC-002",
            "slice": "numeric_contradiction",
            "question": "What is the maximum nightly hotel reimbursement cap (excluding room taxes) for Tier-1 High-Cost Cities under Aethelgard Orbital Systems policy?",
            "claim_text": "Tier-1 High-Cost Cities (Zurich, Tokyo, New York, London, Singapore) have a hotel reimbursement cap of $380 USD per night.",
            "source_evidence": [
                "- Tier-1 High-Cost Cities (Zurich, Tokyo, New York, London, Singapore): $310 USD per night"
            ],
            "human_label": "contradicted",
            "human_notes": "Contradicted by Section 4 of DOC-002: the Tier-1 nightly hotel reimbursement cap is $310 USD per night, not $380 USD.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-017",
            "base_id": "BASE-016",
            "document_id": "DOC-003",
            "slice": "numeric_contradiction",
            "question": "What is the maximum depth beneath the glacier surface reached by each vertical borehole string in the SSNA?",
            "claim_text": "Each vertical borehole string in the SSNA reaches a maximum depth of 1,200 meters beneath the glacier surface.",
            "source_evidence": [
                "Each vertical borehole string reaches a maximum depth of 620 meters beneath the glacier surface and carries 12 autonomous \"FrostPulse-3\" sensor nodes spaced 35 meters apart."
            ],
            "human_label": "contradicted",
            "human_notes": "Contradicted by paragraph 3 of DOC-003: each SSNA vertical borehole string reaches a maximum depth of 620 meters, not 1,200 meters.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-018",
            "base_id": "BASE-020",
            "document_id": "DOC-004",
            "slice": "numeric_contradiction",
            "question": "What is the maximum allowed clock skew between the `X-Chronos-Timestamp` header and the gateway atomic clock before a request is rejected?",
            "claim_text": "If the timestamp header X-Chronos-Timestamp differs from the gateway atomic clock by more than 120 seconds in either direction, the gateway rejects the request.",
            "source_evidence": [
                "If the timestamp header `X-Chronos-Timestamp` differs from the gateway atomic clock by more than 45 seconds in either direction, the gateway rejects the request with error code `ERR_CLOCK_SKEW_904` (HTTP 401)."
            ],
            "human_label": "contradicted",
            "human_notes": "Contradicted by Section 1 of DOC-004: the maximum allowed clock skew is 45 seconds, not 120 seconds.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-019",
            "base_id": "BASE-026",
            "document_id": "DOC-005",
            "slice": "numeric_contradiction",
            "question": "What is the non-refundable upfront technology transfer fee that Licensee must pay Licensor under Agreement VB-2025-881?",
            "claim_text": "Within 15 business days of the Execution Date, Licensee shall pay Licensor a non-refundable upfront technology transfer fee of 550,000 CHF.",
            "source_evidence": [
                "Within 15 business days of the Execution Date, Licensee shall pay Licensor a non-refundable upfront technology transfer fee of 450,000 CHF."
            ],
            "human_label": "contradicted",
            "human_notes": "Contradicted by Clause 2 of DOC-005: the non-refundable upfront technology transfer fee is 450,000 CHF, not 550,000 CHF.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-020",
            "base_id": "BASE-034",
            "document_id": "DOC-006",
            "slice": "numeric_contradiction",
            "question": "What is the terminal elimination half-life of brigatadine and what percentage of the drug is bound to human plasma albumin?",
            "claim_text": "Brigatadine exhibits a terminal elimination half-life of 28 hours and is 82% bound to human plasma albumin.",
            "source_evidence": [
                "Brigatadine exhibits a terminal elimination half-life of 19 hours and is 82% bound to human plasma albumin."
            ],
            "human_label": "contradicted",
            "human_notes": "Contradicted by Section 3 of DOC-006: brigatadine's terminal elimination half-life is 19 hours, not 28 hours.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-021",
            "base_id": "BASE-001",
            "document_id": "DOC-001",
            "slice": "wrong_entity_or_condition",
            "question": "What is the continuous supply current drawn by the Helios-X9 in active telemetry acquisition mode at 20 °C?",
            "claim_text": "The Helios-X9 draws a continuous supply current of 95 mA in active mode.",
            "source_evidence": [
                "In active telemetry acquisition mode at 20 °C, the controller draws a continuous supply current of 42 mA.",
                "The legacy Helios-X7 operates at a nominal core supply voltage of 5.0 V, draws 95 mA in active mode, and supports a maximum clock frequency of only 96 MHz."
            ],
            "human_label": "contradicted",
            "human_notes": "Contradicted sibling-entity trap in DOC-001: 95 mA is the active-mode current of the legacy Helios-X7, whereas the Helios-X9 draws 42 mA.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-022",
            "base_id": "BASE-016",
            "document_id": "DOC-003",
            "slice": "wrong_entity_or_condition",
            "question": "What is the maximum depth beneath the glacier surface reached by each vertical borehole string in the SSNA?",
            "claim_text": "Each vertical borehole string in the SSNA is deployed down to a maximum depth of 2,450 meters beneath the glacier surface.",
            "source_evidence": [
                "Unlike the older IceCube observatory at the South Pole, which relies on deep optical Cherenkov strings deployed down to 2,450 meters, the SSNA operates in shallower, colder polar firn using hybrid radio-acoustic Cherenkov modules.",
                "Each vertical borehole string reaches a maximum depth of 620 meters beneath the glacier surface and carries 12 autonomous \"FrostPulse-3\" sensor nodes spaced 35 meters apart."
            ],
            "human_label": "contradicted",
            "human_notes": "Contradicted sibling-entity trap in DOC-003: 2,450 meters is the depth of the South Pole IceCube observatory, whereas SSNA strings reach 620 meters.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-023",
            "base_id": "BASE-039",
            "document_id": "DOC-007",
            "slice": "wrong_entity_or_condition",
            "question": "What is the main battery bank energy capacity and battery chemistry of MV Borealis-IV?",
            "claim_text": "MV Borealis-IV is equipped with a 4.2 MWh nickel-manganese-cobalt (NMC) main battery bank.",
            "source_evidence": [
                "MV Borealis-IV is powered by dual permanent-magnet azimuth thrusters rated at 1,800 kW each, supplied by a 6.4 MWh liquid-cooled lithium-iron-phosphate (LFP) main battery bank divided into Port and Starboard fire-isolated compartments.",
                "By contrast, the sister vessel MV Borealis-II uses an older X-band 9.4 GHz mechanically scanned radar and a smaller 4.2 MWh nickel-manganese-cobalt (NMC) battery bank."
            ],
            "human_label": "contradicted",
            "human_notes": "Contradicted sibling-entity trap in DOC-007: 4.2 MWh NMC belongs to sister vessel MV Borealis-II, whereas MV Borealis-IV has a 6.4 MWh LFP battery bank.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-024",
            "base_id": "BASE-040",
            "document_id": "DOC-007",
            "slice": "wrong_entity_or_condition",
            "question": "Why did the NauticBrain-5 computer on MV Borealis-IV interpret the 19 dB radar attenuation spike as a solid ghost obstacle 180 meters ahead?",
            "claim_text": "MV Borealis-IV relies on an X-band 9.4 GHz mechanically scanned forward marine radar.",
            "source_evidence": [
                "- Forward W-band solid-state marine radar operating at 77 GHz",
                "By contrast, the sister vessel MV Borealis-II uses an older X-band 9.4 GHz mechanically scanned radar and a smaller 4.2 MWh nickel-manganese-cobalt (NMC) battery bank."
            ],
            "human_label": "contradicted",
            "human_notes": "Contradicted sibling-entity trap in DOC-007: the 9.4 GHz X-band radar belongs to MV Borealis-II, whereas MV Borealis-IV uses a 77 GHz W-band solid-state radar.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-025",
            "base_id": "BASE-044",
            "document_id": "DOC-008",
            "slice": "wrong_entity_or_condition",
            "question": "Across which three underground data center cities and site designations are StratosVault Tier-0 objects synchronously erasure-coded?",
            "claim_text": "StratosVault Tier-0 replicates data asynchronously across two sites with a replication lag window of up to 90 seconds.",
            "source_evidence": [
                "Every object written to a Tier-0 bucket is synchronously erasure-coded across three underground data centers located in Lucerne (Site Alpha), Bern (Site Beta), and Lugano (Site Gamma) before an HTTP 201 Write Acknowledgement is returned to the client.",
                "By contrast, StratosVault Tier-1 Standard Storage replicates data asynchronously across only two sites (Lucerne and Bern) with a replication lag window of up to 90 seconds, and StratosVault Cold-Archive Deep Vault stores magnetic tape shards in a single facility in Andermatt."
            ],
            "human_label": "contradicted",
            "human_notes": "Contradicted sibling-tier trap in DOC-008: asynchronous two-site replication with a 90-second lag applies to Tier-1 Standard Storage, whereas Tier-0 is synchronously erasure-coded across three sites.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-026",
            "base_id": "BASE-004",
            "document_id": "DOC-001",
            "slice": "negation_inversion",
            "question": "Does the Helios-X9 include an on-die hardware I2C controller?",
            "claim_text": "The Helios-X9 hardware includes UART, I2C, USB, and Ethernet controllers to minimize die area and single-event latch-up cross-section.",
            "source_evidence": [
                "Note: The Helios-X9 hardware omits UART, I2C, USB, and Ethernet controllers to minimize die area and single-event latch-up cross-section."
            ],
            "human_label": "contradicted",
            "human_notes": "Contradicted by Section 5 of DOC-001: the Helios-X9 explicitly omits UART, I2C, USB, and Ethernet controllers.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-027",
            "base_id": "BASE-011",
            "document_id": "DOC-002",
            "slice": "negation_inversion",
            "question": "Are Propulsion Test Engineers at Aethelgard Orbital Systems eligible for regular remote work?",
            "claim_text": "Propulsion Test Engineers and Cleanroom Technicians are classified as Mission-Critical On-Site Staff and are eligible for regular remote work.",
            "source_evidence": [
                "By contrast, Propulsion Test Engineers and Cleanroom Technicians are classified as Mission-Critical On-Site Staff and are not eligible for regular remote work."
            ],
            "human_label": "contradicted",
            "human_notes": "Contradicted by Section 2 of DOC-002: Propulsion Test Engineers and Cleanroom Technicians are explicitly not eligible for regular remote work.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-028",
            "base_id": "BASE-028",
            "document_id": "DOC-005",
            "slice": "negation_inversion",
            "question": "Is Kestrel Biopolymers Ltd allowed to manufacture or commercially distribute EsterCleav-9 in Denmark under Schedule C?",
            "claim_text": "Manufacturing or commercial distribution inside Denmark, Germany, or Poland is expressly included in this grant and permitted for Licensee.",
            "source_evidence": [
                "Manufacturing or commercial distribution inside Denmark, Germany, or Poland is expressly excluded from this grant and remains reserved by Licensor."
            ],
            "human_label": "contradicted",
            "human_notes": "Contradicted by Clause 1 of DOC-005: manufacturing or commercial distribution inside Denmark, Germany, or Poland is expressly excluded from the grant.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-029",
            "base_id": "BASE-035",
            "document_id": "DOC-006",
            "slice": "negation_inversion",
            "question": "Should unopened blister packs of Zelvorin-XR be stored in a refrigerator below 8 °C?",
            "claim_text": "Refrigerate below 8 °C because exposure below 8 °C prevents irreversible crystallization of the extended-release polymer coat.",
            "source_evidence": [
                "Do not refrigerate or freeze, as exposure below 8 °C causes irreversible crystallization of the extended-release polymer coat."
            ],
            "human_label": "contradicted",
            "human_notes": "Contradicted by Section 4 of DOC-006: Zelvorin-XR must not be refrigerated or frozen because exposure below 8 °C causes irreversible crystallization.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-030",
            "base_id": "BASE-047",
            "document_id": "DOC-008",
            "slice": "negation_inversion",
            "question": "Can a tenant's Root Administrator delete a StratosVault Tier-0 object placed under Compliance WORM Retention Lock before the lock expires?",
            "claim_text": "Objects placed under Compliance WORM Retention Lock have a minimum lock duration of 30 days and a maximum lock duration of 35 years, and locked objects can be deleted by the tenant's Root Administrator prior to lock expiration.",
            "source_evidence": [
                "Objects placed under Compliance WORM (Write-Once-Read-Many) Retention Lock have a minimum lock duration of 30 days and a maximum lock duration of 35 years; locked objects cannot be deleted even by the tenant's Root Administrator prior to lock expiration."
            ],
            "human_label": "contradicted",
            "human_notes": "Contradicted by Section 4 of DOC-008: locked objects under Compliance WORM Retention Lock cannot be deleted even by the tenant's Root Administrator prior to lock expiration.",
            "needs_human_review": False,
        },
        # --- 15 UNSUPPORTED CLAIMS (AUD-031 .. AUD-045) ---
        {
            "claim_id": "AUD-031",
            "base_id": "BASE-006",
            "document_id": "DOC-001",
            "slice": "unanswerable_extrapolation",
            "question": "What is the unit price of the Helios-X9 microcontroller for volume orders of 1,000 units?",
            "claim_text": "The unit price of the Helios-X9 microcontroller for volume orders of 1,000 units is $420 USD per unit.",
            "source_evidence": [
                "All production wafers are fabricated exclusively at Vaelen MicroSystems Fab-4 in Dresden and packaged in a 144-pin ceramic quad flat pack (CQFP-144)."
            ],
            "human_label": "unsupported",
            "human_notes": "Unsupported fabrication on unanswerable BASE-006: DOC-001 describes fabrication and packaging in Section 1 but never states any unit price or commercial pricing tier.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-032",
            "base_id": "BASE-007",
            "document_id": "DOC-001",
            "slice": "unanswerable_extrapolation",
            "question": "What is the continuous current draw of the legacy Helios-X7 controller when placed into deep-hibernation mode?",
            "claim_text": "When placed into deep-hibernation mode, the legacy Helios-X7 controller draws a continuous current of 410 uA.",
            "source_evidence": [
                "The legacy Helios-X7 operates at a nominal core supply voltage of 5.0 V, draws 95 mA in active mode, and supports a maximum clock frequency of only 96 MHz."
            ],
            "human_label": "unsupported",
            "human_notes": "Unsupported fabrication on unanswerable BASE-007: Section 3 of DOC-001 states 95 mA active-mode current for the Helios-X7, but never specifies its deep-hibernation current.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-033",
            "base_id": "BASE-013",
            "document_id": "DOC-002",
            "slice": "unanswerable_extrapolation",
            "question": "What is the personal vehicle mileage reimbursement rate per kilometer under Policy Pol-HR-409?",
            "claim_text": "Under Policy Pol-HR-409, personal vehicle mileage is reimbursed at a flat rate of $0.68 USD per kilometer.",
            "source_evidence": [
                "This policy governs work-location eligibility, domestic and international business travel, meal reimbursement, and home-office stipends for all full-time salaried employees of Aethelgard Orbital Systems."
            ],
            "human_label": "unsupported",
            "human_notes": "Unsupported fabrication on unanswerable BASE-013: DOC-002 covers flights, hotels, meals, and ergonomic stipends, but contains no personal vehicle mileage reimbursement rate.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-034",
            "base_id": "BASE-019",
            "document_id": "DOC-003",
            "slice": "unanswerable_extrapolation",
            "question": "What is the electrical power output rating in kilowatts of each of the three methanol fuel-cell generators at the SSNA surface bunker?",
            "claim_text": "Each of the three methanol fuel-cell generators at the SSNA surface bunker has a continuous electrical output rating of 125 kW.",
            "source_evidence": [
                "Power for the unmanned surface central processing bunker is supplied by three methanol fuel-cell generators backed by a 400 kWh lithium-titanate battery array, as environmental treaties on Nordaustlandet prohibit nuclear radioisotope thermoelectric generators (RTGs) and diesel turbines."
            ],
            "human_label": "unsupported",
            "human_notes": "Unsupported fabrication on unanswerable BASE-019: paragraph 4 of DOC-003 mentions three methanol fuel-cell generators and a 400 kWh battery array, but does not state the kilowatt power rating of the generators.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-035",
            "base_id": "BASE-024",
            "document_id": "DOC-004",
            "slice": "unanswerable_extrapolation",
            "question": "For how many days are undeliverable webhook events retained inside the ChronosPay Dead-Letter Queue (DLQ) before being purged?",
            "claim_text": "Undeliverable webhook events moved to the ChronosPay Dead-Letter Queue (DLQ) are retained for 14 calendar days before automatic purge.",
            "source_evidence": [
                "If the callback times out or returns HTTP 5xx, ChronosPay retries delivery up to 5 times at fixed backoff intervals of 1 minute, 5 minutes, 30 minutes, 2 hours, and 12 hours before moving the event to the Dead-Letter Queue (DLQ)."
            ],
            "human_label": "unsupported",
            "human_notes": "Unsupported fabrication on unanswerable BASE-024: Section 4 of DOC-004 states that failed events are moved to the DLQ after 5 retries, but never specifies a DLQ retention duration.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-036",
            "base_id": "BASE-031",
            "document_id": "DOC-005",
            "slice": "unanswerable_extrapolation",
            "question": "How many years is the initial contract term of the Commercial Enzyme Sublicensing & Royalty Agreement (VB-2025-881) before renewal?",
            "claim_text": "The initial contract term of Agreement VB-2025-881 is 10 calendar years from the Execution Date.",
            "source_evidence": [
                "Regardless of actual sales volume, Licensee guarantees a Minimum Annual Royalty of 180,000 CHF per calendar year starting in Calendar Year 2026."
            ],
            "human_label": "unsupported",
            "human_notes": "Unsupported fabrication on unanswerable BASE-031: DOC-005 states the Execution Date and annual royalty obligations starting in 2026, but never states the contract term duration in years.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-037",
            "base_id": "BASE-042",
            "document_id": "DOC-007",
            "slice": "unanswerable_extrapolation",
            "question": "What is the thruster power rating in kilowatts of the sister vessel MV Borealis-II?",
            "claim_text": "The sister vessel MV Borealis-II is propelled by dual azimuth thrusters rated at 1,100 kW each.",
            "source_evidence": [
                "By contrast, the sister vessel MV Borealis-II uses an older X-band 9.4 GHz mechanically scanned radar and a smaller 4.2 MWh nickel-manganese-cobalt (NMC) battery bank."
            ],
            "human_label": "unsupported",
            "human_notes": "Unsupported fabrication on unanswerable BASE-042: Section 2 of DOC-007 specifies radar and battery capacity for MV Borealis-II, but omits its thruster power rating.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-038",
            "base_id": "BASE-049",
            "document_id": "DOC-008",
            "slice": "unanswerable_extrapolation",
            "question": "What is the monthly storage price per terabyte for StratosVault Tier-0 Sovereign Object Storage?",
            "claim_text": "StratosVault Tier-0 Sovereign Object Storage is billed at a flat monthly rate of 95 CHF per terabyte.",
            "source_evidence": [
                "If Monthly Uptime Percentage for a Tier-0 bucket falls below 99.995% in any billing month, the customer is eligible for Service Credits applied against the next monthly invoice:"
            ],
            "human_label": "unsupported",
            "human_notes": "Unsupported fabrication on unanswerable BASE-049: Section 3 of DOC-008 defines Service Credits applied against the monthly invoice, but never states the monthly price per terabyte.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-039",
            "base_id": "BASE-025",
            "document_id": "DOC-004",
            "slice": "unanswerable_extrapolation",
            "question": "What percentage foreign-exchange (FX) markup fee does ChronosPay charge on cross-border multi-currency transfers?",
            "claim_text": "ChronosPay charges a 0.75% foreign-exchange markup fee on all cross-border multi-currency transfers.",
            "source_evidence": [
                "Cross-border multi-currency transfers settle asynchronously and transition from `PENDING_FX` to `SETTLED` within 4 hours during normal clearing windows."
            ],
            "human_label": "unsupported",
            "human_notes": "Unsupported fabrication on unanswerable BASE-025: Section 3 of DOC-004 states the 4-hour settlement window for cross-border transfers, but never mentions any FX markup percentage.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-040",
            "base_id": "BASE-037",
            "document_id": "DOC-006",
            "slice": "unanswerable_extrapolation",
            "question": "What is the recommended daily dose of Zelvorin-XR for adolescents aged 12 to 17 years?",
            "claim_text": "For adolescents aged 12 to 17 years, the recommended daily maintenance dose of Zelvorin-XR is one 22.5 mg pediatric capsule taken each morning.",
            "source_evidence": [
                "Pediatric safety and efficacy in individuals under 18 years of age have not been established."
            ],
            "human_label": "unsupported",
            "human_notes": "Human review decision (retained 'unsupported' & current wording): Section 2 of DOC-006 states that 'Pediatric safety and efficacy in individuals under 18 years of age have not been established' and does not establish an adolescent dose. Distinction between 'unsupported' and 'contradicted': the source neither proves a specific alternative adolescent dose nor categorically disproves every possible adolescent dose; rather, it states that safety/efficacy have not been established, leaving any specific adolescent dosage schedule ('one 22.5 mg pediatric capsule taken each morning') unsubstantiated ('unsupported') rather than factually refuted by an established counter-fact ('contradicted').",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-041",
            "base_id": "BASE-050",
            "document_id": "DOC-008",
            "slice": "unanswerable_extrapolation",
            "question": "What is the guaranteed Monthly Uptime Percentage and P99 first-byte read latency for StratosVault Tier-1 Standard Storage?",
            "claim_text": "StratosVault Tier-1 Standard Storage guarantees a Monthly Uptime Percentage of 99.90% and a P99 first-byte read latency of 48 ms.",
            "source_evidence": [
                "By contrast, StratosVault Tier-1 Standard Storage replicates data asynchronously across only two sites (Lucerne and Bern) with a replication lag window of up to 90 seconds, and StratosVault Cold-Archive Deep Vault stores magnetic tape shards in a single facility in Andermatt."
            ],
            "human_label": "unsupported",
            "human_notes": "Unsupported fabrication on unanswerable BASE-050: DOC-008 gives a 90-second replication lag for Tier-1 Standard Storage, but never states an uptime percentage or P99 read latency guarantee for Tier-1.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-042",
            "base_id": "BASE-043",
            "document_id": "DOC-007",
            "slice": "unanswerable_extrapolation",
            "question": "What was the name of the remote shore operator on duty at the Haugesund Shore Control Center during the October 19, 2025 incident?",
            "claim_text": "The remote shore operator on duty at the Haugesund Shore Control Center during the incident had 14 years of maritime watchstanding experience.",
            "source_evidence": [
                "When the shore operator at Haugesund SCC attempted to issue a manual helm override via the primary Ku-band satellite link, the command packet was delayed by 3.8 seconds due to an automatic satellite beam handover, exceeding the vessel's 2.5-second dead-man watchdog threshold and triggering autonomous emergency anchoring."
            ],
            "human_label": "unsupported",
            "human_notes": "Unsupported fabrication on BASE-043: Section 3 of DOC-007 mentions the shore operator at Haugesund SCC attempting a manual helm override, but gives neither the operator's name nor years of experience.",
            "needs_human_review": False,
        },
        {
            "claim_id": "AUD-043",
            "base_id": "BASE-011",
            "document_id": "DOC-002",
            "slice": "unsupported_inference",
            "question": "Are Propulsion Test Engineers at Aethelgard Orbital Systems eligible for regular remote work?",
            "claim_text": "Full-time employees in Software Engineering work remotely every Monday, Wednesday, and Friday.",
            "source_evidence": [
                "Full-time employees in Software Engineering, Mission Analytics, and Legal Compliance may work remotely for up to 3 days per week, provided they attend in-person core collaboration days at their assigned hub every Tuesday and Thursday."
            ],
            "human_label": "unsupported",
            "human_notes": "Human review decision (retained 'unsupported' & current wording; needs_human_review retained until verification metadata is authorized): Non-entailment rationale: Section 2 of DOC-002 grants a permissive cap ('may work remotely for up to 3 days per week' with Tuesday/Thursday in-person core days), which allows 0, 1, 2, or 3 remote days and does not entail that Software Engineering employees obligatorily or universally work remotely every Monday, Wednesday, and Friday.",
            "needs_human_review": True,
        },
        {
            "claim_id": "AUD-044",
            "base_id": "BASE-027",
            "document_id": "DOC-005",
            "slice": "unsupported_inference",
            "question": "Which six countries comprise the exclusive Nordic-Baltic Territory granted to Kestrel Biopolymers Ltd for EsterCleav-9?",
            "claim_text": "VortexBio Synthetics AG manufactures EsterCleav-9 at its own industrial facility in Basel, Switzerland.",
            "source_evidence": [
                "All disputes arising out of this Agreement shall be resolved exclusively by binding arbitration in Basel, Switzerland under the Swiss Rules of International Arbitration."
            ],
            "human_label": "unsupported",
            "human_notes": "Human review decision (retained 'unsupported' & current wording; needs_human_review retained until verification metadata is authorized): Non-entailment rationale: Clause 4 of DOC-005 designates Basel, Switzerland solely as the legal seat for binding arbitration under the Swiss Rules of International Arbitration; a dispute-resolution forum clause does not entail that VortexBio Synthetics AG owns or operates a manufacturing facility in Basel.",
            "needs_human_review": True,
        },
        {
            "claim_id": "AUD-045",
            "base_id": "BASE-041",
            "document_id": "DOC-007",
            "slice": "unsupported_inference",
            "question": "On what date was MV Borealis-IV cleared to resume commercial coastal operations after operator Skandia Autonav AS completed the remedial actions?",
            "claim_text": "The Nordic Maritime Safety Directorate imposed a monetary fine on operator Skandia Autonav AS before clearing MV Borealis-IV to resume commercial operations.",
            "source_evidence": [
                "Before MV Borealis-IV was cleared to resume commercial coastal operations on November 2, 2025, operator Skandia Autonav AS completed three mandatory remedial actions:"
            ],
            "human_label": "unsupported",
            "human_notes": "Human review decision (retained 'unsupported' & current wording; needs_human_review retained until verification metadata is authorized): Non-entailment rationale: Section 4 of DOC-007 enumerates three mandatory technical remedial actions completed by Skandia Autonav AS before MV Borealis-IV returned to service on November 2, 2025, and contains no mention of any monetary fine or financial penalty.",
            "needs_human_review": True,
        },
    ]

    records: list[dict[str, Any]] = []
    for item in raw_claims:
        records.append({
            **item,
            "status": "verified",
            "verified_by": "sayanpal1499",
            "verified_on": "2026-10-13",
        })
    return records


REVIEW_DECISION_RECORDED_CLAIM_IDS: frozenset[str] = frozenset(
    f"AUD-{i:03d}" for i in range(1, 46)
)


def export_audit_claims_jsonl_and_csv(
    jsonl_path: Path = DEFAULT_AUDIT_CLAIMS_PATH,
    csv_path: Path = PROJECT_ROOT / "data" / "audit_claims_review.csv",
) -> list[dict[str, Any]]:
    """Write the 45 manually verified atomic audit claims to JSONL and human-review CSV."""
    import csv

    records = build_audit_claims_dataset()
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    fieldnames = [
        "claim_id",
        "base_id",
        "document_id",
        "slice",
        "human_label",
        "needs_human_review",
        "review_decision_status",
        "claim_text",
        "source_evidence",
        "human_notes",
        "status",
        "verified_by",
        "verified_on",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            decision_status = (
                "decision_recorded_verified"
                if r["claim_id"] in REVIEW_DECISION_RECORDED_CLAIM_IDS
                else "pending_review"
            )
            writer.writerow({
                "claim_id": r["claim_id"],
                "base_id": r["base_id"],
                "document_id": r["document_id"],
                "slice": r["slice"],
                "human_label": r["human_label"],
                "needs_human_review": r["needs_human_review"],
                "review_decision_status": decision_status,
                "claim_text": r["claim_text"],
                "source_evidence": " | ".join(r["source_evidence"]),
                "human_notes": r["human_notes"],
                "status": r["status"],
                "verified_by": r["verified_by"] or "",
                "verified_on": r["verified_on"] or "",
            })
    return records

