"""
services/document_classifier.py — Document Classification

Classifies uploaded documents into domain categories to drive
downstream chunking strategy selection.

Classification approach:
  1. Rule-based (file extension + keyword scoring) — fast, no API call
  2. LLM-assisted (Gemini Flash on first 2000 chars) — higher accuracy,
     only used when CLASSIFIER_USE_LLM=True and rule-based confidence is low

DocumentClass → ChunkingStrategy mapping:
  LEGAL / CONTRACT / INSURANCE / POLICY → CLAUSE_AWARE
  RESEARCH                               → SEMANTIC
  MANUAL / SOP                           → HIERARCHICAL
  EMAIL                                  → STRUCTURE_PRESERVE
  SPREADSHEET                            → ROW_BASED
  PRESENTATION                           → HIERARCHICAL
  FINANCE / INVOICE / HR / REPORT        → PARENT_CHILD (default hybrid)
  GENERAL                                → PARENT_CHILD
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from models.domain import DocumentClass, ChunkingStrategy
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    document_class: DocumentClass
    confidence: float
    strategy: ChunkingStrategy
    method: str   # "rule_based" or "llm_assisted"


# ─────────────────────────────────────────────────────────────────────────────
# Keyword scoring rules
# Each class has a list of (pattern, weight) pairs.
# Score = sum(weight for each match). Normalized to [0, 1] by dividing by max.
# ─────────────────────────────────────────────────────────────────────────────

_KEYWORD_RULES: Dict[DocumentClass, List[Tuple[re.Pattern, float]]] = {
    DocumentClass.LEGAL: [
        (re.compile(r"\b(whereas|witnesseth|indemnif|arbitration|jurisdiction|governing\s+law)\b", re.I), 3.0),
        (re.compile(r"\b(plaintiff|defendant|court|statute|clause|whereas|hereinafter|hereby)\b", re.I), 2.0),
        (re.compile(r"\b(legal|law|litigation|lawsuit|judgment|motion|pleading)\b", re.I), 1.5),
    ],
    DocumentClass.CONTRACT: [
        (re.compile(r"\b(this\s+agreement|terms\s+and\s+conditions|effective\s+date|party\s+of)\b", re.I), 3.0),
        (re.compile(r"\b(obligations|representations|warranties|termination|renewal|breach)\b", re.I), 2.0),
        (re.compile(r"\b(contract|agreement|signed|signatory|counterpart)\b", re.I), 1.5),
    ],
    DocumentClass.POLICY: [
        (re.compile(r"\b(policy\s+number|policy\s+period|policyholder|insured|deductible)\b", re.I), 3.0),
        (re.compile(r"\b(coverage|exclusion|premium|claim|endorsement|rider)\b", re.I), 2.0),
        (re.compile(r"\b(policy|rule|procedure|guideline|standard|compliance)\b", re.I), 1.0),
    ],
    DocumentClass.INSURANCE: [
        (re.compile(r"\b(insured|insurer|deductible|premium|liability|bodily\s+injury|comprehensive|collision)\b", re.I), 3.0),
        (re.compile(r"\b(coverage|claim|underwriter|policyholder|beneficiary|adjuster)\b", re.I), 2.0),
    ],
    DocumentClass.MEDICAL: [
        (re.compile(r"\b(patient|diagnosis|treatment|medication|dosage|physician|clinical|symptom)\b", re.I), 3.0),
        (re.compile(r"\b(icd|cpt|ehr|emr|fda|hipaa|lab\s+result|prescription)\b", re.I), 2.0),
        (re.compile(r"\b(medical|health|hospital|surgery|therapy|nursing|care)\b", re.I), 1.5),
    ],
    DocumentClass.FINANCE: [
        (re.compile(r"\b(balance\s+sheet|income\s+statement|cash\s+flow|ebitda|gaap|ifrs|audited)\b", re.I), 3.0),
        (re.compile(r"\b(revenue|earnings|profit|loss|assets|liabilities|equity|dividend|fiscal)\b", re.I), 2.0),
        (re.compile(r"\b(financial|accounting|budget|forecast|quarter|annual\s+report)\b", re.I), 1.5),
    ],
    DocumentClass.INVOICE: [
        (re.compile(r"\b(invoice\s+(number|#|no)|bill\s+to|ship\s+to|payment\s+due|total\s+due)\b", re.I), 3.0),
        (re.compile(r"\b(qty|quantity|unit\s+price|subtotal|tax|discount|amount\s+due)\b", re.I), 2.0),
        (re.compile(r"\b(invoice|receipt|billing|purchase\s+order|po\s+number)\b", re.I), 1.5),
    ],
    DocumentClass.HR: [
        (re.compile(r"\b(employee|employer|onboarding|performance\s+review|salary|compensation|benefit)\b", re.I), 2.5),
        (re.compile(r"\b(vacation|pto|sick\s+leave|termination|severance|job\s+description|kpi)\b", re.I), 2.0),
        (re.compile(r"\b(hr|human\s+resources|payroll|recruit|hiring|workforce)\b", re.I), 1.5),
    ],
    DocumentClass.RESEARCH: [
        (re.compile(r"\b(abstract|introduction|methodology|results|discussion|conclusion|references)\b", re.I), 3.0),
        (re.compile(r"\b(hypothesis|dataset|experiment|statistical|significance|p-value|confidence\s+interval)\b", re.I), 2.5),
        (re.compile(r"\b(paper|study|research|journal|doi|citation|literature\s+review)\b", re.I), 2.0),
    ],
    DocumentClass.MANUAL: [
        (re.compile(r"\b(table\s+of\s+contents|chapter|section|step\s+\d+|figure\s+\d+)\b", re.I), 2.5),
        (re.compile(r"\b(installation|configuration|troubleshoot|maintenance|operation|specification)\b", re.I), 2.0),
        (re.compile(r"\b(manual|guide|handbook|reference|documentation|user\s+guide)\b", re.I), 1.5),
    ],
    DocumentClass.SOP: [
        (re.compile(r"\b(standard\s+operating\s+procedure|sop|work\s+instruction|process\s+flow)\b", re.I), 3.0),
        (re.compile(r"\b(step\s+\d+|do\s+not|warning|caution|note:|prerequisite|checklist)\b", re.I), 2.0),
        (re.compile(r"\b(procedure|process|workflow|flowchart|responsibility|approval)\b", re.I), 1.5),
    ],
    DocumentClass.EMAIL: [
        (re.compile(r"\b(from:|to:|cc:|bcc:|subject:|reply-to:|sent:|received:)\b", re.I), 3.0),
        (re.compile(r"\b(dear|sincerely|regards|best|forwarded|original\s+message|replied)\b", re.I), 2.0),
        (re.compile(r"\b(email|message|thread|inbox|outlook|gmail|exchange)\b", re.I), 1.5),
    ],
    DocumentClass.REPORT: [
        (re.compile(r"\b(executive\s+summary|key\s+findings|recommendations|appendix)\b", re.I), 2.5),
        (re.compile(r"\b(quarterly|annual|monthly|weekly|dashboard|kpi|metric|trend)\b", re.I), 2.0),
        (re.compile(r"\b(report|analysis|overview|summary|performance)\b", re.I), 1.5),
    ],
}

# DocumentClass → ChunkingStrategy mapping
_CLASS_TO_STRATEGY: Dict[DocumentClass, ChunkingStrategy] = {
    DocumentClass.LEGAL:        ChunkingStrategy.CLAUSE_AWARE,
    DocumentClass.CONTRACT:     ChunkingStrategy.CLAUSE_AWARE,
    DocumentClass.POLICY:       ChunkingStrategy.CLAUSE_AWARE,
    DocumentClass.INSURANCE:    ChunkingStrategy.CLAUSE_AWARE,
    DocumentClass.MEDICAL:      ChunkingStrategy.PARENT_CHILD,
    DocumentClass.FINANCE:      ChunkingStrategy.PARENT_CHILD,
    DocumentClass.INVOICE:      ChunkingStrategy.ROW_BASED,
    DocumentClass.HR:           ChunkingStrategy.PARENT_CHILD,
    DocumentClass.RESEARCH:     ChunkingStrategy.SEMANTIC,
    DocumentClass.MANUAL:       ChunkingStrategy.HIERARCHICAL,
    DocumentClass.SOP:          ChunkingStrategy.HIERARCHICAL,
    DocumentClass.EMAIL:        ChunkingStrategy.STRUCTURE_PRESERVE,
    DocumentClass.REPORT:       ChunkingStrategy.PARENT_CHILD,
    DocumentClass.SPREADSHEET:  ChunkingStrategy.ROW_BASED,
    DocumentClass.PRESENTATION: ChunkingStrategy.HIERARCHICAL,
    DocumentClass.GENERAL:      ChunkingStrategy.PARENT_CHILD,
}

# File extension → strong class hints
_EXTENSION_HINTS: Dict[str, DocumentClass] = {
    "xlsx": DocumentClass.SPREADSHEET,
    "csv":  DocumentClass.SPREADSHEET,
    "pptx": DocumentClass.PRESENTATION,
    "eml":  DocumentClass.EMAIL,
}


def _rule_based_classify(
    text: str,
    file_extension: str,
) -> Tuple[DocumentClass, float]:
    """
    Score the text against all keyword rules and return the
    highest-scoring class with its normalized confidence.
    """
    # Extension hint overrides with high confidence for unambiguous types
    if file_extension in _EXTENSION_HINTS:
        return _EXTENSION_HINTS[file_extension], 0.95

    scores: Dict[DocumentClass, float] = {}

    for doc_class, rules in _KEYWORD_RULES.items():
        score = 0.0
        for pattern, weight in rules:
            matches = len(pattern.findall(text))
            score += matches * weight
        if score > 0:
            scores[doc_class] = score

    if not scores:
        return DocumentClass.GENERAL, 0.3

    max_score = max(scores.values())
    best_class = max(scores, key=scores.get)

    # Normalize confidence: cap at 0.95 to prevent overconfidence
    # Rough heuristic: a score of 20+ is very high confidence
    confidence = min(0.95, max_score / 20.0)

    return best_class, confidence


async def _llm_classify(text: str, rule_class: DocumentClass) -> Tuple[DocumentClass, float]:
    """
    Use Gemini Flash to improve classification when rule confidence is low.
    Returns (class, confidence) or falls back to rule_class on failure.
    """
    from services.model_router import generate

    sample = text[:2000].strip()
    prompt = f"""Classify the following document excerpt into exactly ONE of these categories:
legal, contract, policy, insurance, medical, finance, invoice, hr, research, manual, sop, email, report, spreadsheet, presentation, general

Return ONLY a JSON object with two fields: "class" (string) and "confidence" (float 0.0-1.0).
No explanations. No markdown. Just the JSON.

DOCUMENT EXCERPT:
{sample}
"""
    try:
        raw, _ = await generate(prompt, use_json_mode=True, allow_fallback=False)
        import json
        data = json.loads(raw)
        cls_str = data.get("class", "").lower().strip()
        confidence = float(data.get("confidence", 0.5))
        doc_class = DocumentClass(cls_str)
        return doc_class, confidence
    except Exception as exc:
        logger.warning(
            "LLM classification failed — using rule-based result",
            extra={"error": str(exc)[:200]},
        )
        return rule_class, 0.5


async def classify_document(
    text: str,
    file_extension: str,
    document_id: str,
) -> ClassificationResult:
    """
    Classify a document and return its class and chunking strategy.

    Args:
        text:           Extracted text (first 5000 chars are sufficient).
        file_extension: Lowercase file extension without dot.
        document_id:    For logging.

    Returns:
        ClassificationResult with class, confidence, and strategy.
    """
    # Use a sample for classification — first 5000 chars is enough
    sample = text[:5000]

    rule_class, rule_confidence = _rule_based_classify(sample, file_extension)

    final_class = rule_class
    final_confidence = rule_confidence
    method = "rule_based"

    # Use LLM if enabled and confidence is below threshold
    if (
        settings.CLASSIFIER_USE_LLM
        and rule_confidence < settings.CLASSIFIER_LLM_CONFIDENCE_THRESHOLD
    ):
        llm_class, llm_confidence = await _llm_classify(sample, rule_class)
        if llm_confidence > rule_confidence:
            final_class = llm_class
            final_confidence = llm_confidence
            method = "llm_assisted"

    strategy = _CLASS_TO_STRATEGY.get(final_class, ChunkingStrategy.PARENT_CHILD)

    logger.info(
        "Document classified",
        extra={
            "document_id": document_id,
            "class": final_class.value,
            "confidence": round(final_confidence, 3),
            "strategy": strategy.value,
            "method": method,
        },
    )

    return ClassificationResult(
        document_class=final_class,
        confidence=final_confidence,
        strategy=strategy,
        method=method,
    )
