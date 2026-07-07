import os
import sys
import asyncio
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Set
from config import settings

# Setup Gemini Client if API key is present
client = None
if settings.GOOGLE_API_KEY:
    try:
        from google import genai
        from google.genai import types as genai_types
        client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    except Exception as e:
        print(f"Warning: Failed to initialize Gemini client: {e}")

# ─── Namespace Constants ───────────────────────────────────────────────────────
# Baseline eval: ONLY policy.txt — completely isolated from multidoc docs
EVAL_NAMESPACE          = "eval_baseline_v3"
# Hard benchmark: ONLY policy_a.txt + policy_b.txt — separate namespace
EVAL_NAMESPACE_MULTIDOC = "eval_multidoc_v3"

# Authoritative document sets per namespace — used by purity checks
_ALLOWED_DOCS: Dict[str, Set[str]] = {
    EVAL_NAMESPACE:          {"policy.txt"},
    EVAL_NAMESPACE_MULTIDOC: {"policy_a.txt", "policy_b.txt"},
}

SAMPLE_DOC_PATH = "evaluation/sample_docs/policy.txt"
SAMPLE_DOC_ID   = "policy.txt"


def check_namespace_purity(
    namespace: str,
    allowed_docs: Optional[Set[str]] = None,
    force_clean: bool = False,
) -> bool:
    """
    Verify that a Pinecone namespace contains ONLY the expected document IDs.

    Strategy: fetch a sample of vectors (up to 100) via a zero-vector query and
    inspect the `source` / `document_id` metadata field. Any document name not
    in `allowed_docs` is flagged as a contaminant.

    Args:
        namespace:    The Pinecone namespace to inspect.
        allowed_docs: Set of document IDs that are permitted in this namespace.
                      Defaults to _ALLOWED_DOCS[namespace] if registered.
        force_clean:  If True, delete contaminating vectors and return True.
                      If False, raise RuntimeError so evaluation cannot proceed.

    Returns:
        True  — namespace is clean (or was cleaned).
        False — contamination detected and force_clean=False (will raise).

    Raises:
        RuntimeError: when contamination is found and force_clean=False.
    """
    from services.vectordb import index as pinecone_index
    from services.embedding import embed_texts

    if allowed_docs is None:
        allowed_docs = _ALLOWED_DOCS.get(namespace)
    if not allowed_docs:
        print(f"  [PurityCheck] No allowed-doc list for namespace '{namespace}' -- skipping.")
        return True

    print(f"  [PurityCheck] Inspecting namespace '{namespace}'...")
    print(f"  [PurityCheck] Allowed documents: {sorted(allowed_docs)}")

    # Use a dummy zero-vector to sample metadata — we only care about metadata
    try:
        dummy_emb = embed_texts(["namespace purity check"])[0]
        res = pinecone_index.query(
            vector=dummy_emb,
            top_k=100,
            include_metadata=True,
            namespace=namespace,
        )
        matches = res.get("matches", []) or []
    except Exception as exc:
        print(f"  [PurityCheck] WARNING: Could not query namespace '{namespace}': {exc}")
        return True  # Cannot verify — let evaluation proceed with warning

    if not matches:
        print(f"  [PurityCheck] Namespace '{namespace}' is empty or inaccessible -- OK.")
        return True

    # Collect all unique source names found
    found_docs: Dict[str, List[str]] = {}  # {source_name: [vector_id, ...]}
    for m in matches:
        meta = m.get("metadata") or {}
        source = meta.get("source") or meta.get("document_id", "unknown")
        vid = m.get("id", "?")
        found_docs.setdefault(source, []).append(vid)

    contaminants = {doc: ids for doc, ids in found_docs.items() if doc not in allowed_docs}

    if not contaminants:
        print(f"  [PurityCheck] [OK] Namespace '{namespace}' is clean. "
              f"Found: {sorted(found_docs.keys())}")
        return True

    # ── Contamination detected ────────────────────────────────────────────────
    contaminating_docs = sorted(contaminants.keys())
    total_contaminating = sum(len(v) for v in contaminants.values())

    print(f"  [PurityCheck] [FAIL] CONTAMINATION DETECTED in namespace '{namespace}'!")
    print(f"  [PurityCheck]   Expected: {sorted(allowed_docs)}")
    print(f"  [PurityCheck]   Found:    {sorted(found_docs.keys())}")
    print(f"  [PurityCheck]   Offenders ({total_contaminating} sampled vectors): "
          f"{contaminating_docs}")

    if not force_clean:
        raise RuntimeError(
            f"Namespace '{namespace}' is contaminated with unexpected documents: "
            f"{contaminating_docs}. "
            f"Re-run with --force-clean to purge contaminating vectors, or "
            f"manually delete the namespace in the Pinecone console."
        )

    # ── force_clean=True: delete all vectors in the namespace and re-ingest ──
    print(f"  [PurityCheck] --force-clean active: deleting ALL vectors in "
          f"namespace '{namespace}' for clean re-ingestion...")
    try:
        pinecone_index.delete(delete_all=True, namespace=namespace)
        print(f"  [PurityCheck] [OK] Namespace '{namespace}' purged. "
              f"Documents will be re-ingested on next run.")
        return True
    except Exception as exc:
        print(f"  [PurityCheck] ERROR during purge: {exc}")
        raise


async def ensure_sample_ingested(namespace: str = EVAL_NAMESPACE, force_clean: bool = False):
    """Ensure the sample policy document is ingested in the given namespace."""
    from services.db import documents_collection
    from services.ingestion import ingest_text
    from services.vectordb import index as pinecone_index

    # ── Purity check before touching the namespace ────────────────────────────
    check_namespace_purity(
        namespace=namespace,
        allowed_docs=_ALLOWED_DOCS.get(namespace, {SAMPLE_DOC_ID}),
        force_clean=force_clean,
    )

    doc = documents_collection.find_one(
        {"document_id": SAMPLE_DOC_ID, "namespace": namespace}
    )
    stats    = pinecone_index.describe_index_stats()
    ns_stats = stats.get("namespaces", {}).get(namespace, {})
    vector_count = ns_stats.get("vector_count", 0)

    if doc and vector_count > 0:
        print(
            f"  '{SAMPLE_DOC_ID}' already indexed in namespace "
            f"'{namespace}' ({vector_count} vectors)."
        )
        return

    print(f"  Ingesting '{SAMPLE_DOC_ID}' into namespace '{namespace}'...")
    if not os.path.exists(SAMPLE_DOC_PATH):
        raise FileNotFoundError(f"Sample document not found at {SAMPLE_DOC_PATH}")

    await ingest_text(
        file_path=SAMPLE_DOC_PATH,
        file_type="txt",
        namespace=namespace,
        use_semantic=True,
        original_filename=SAMPLE_DOC_ID,
    )
    print(f"  OK  '{SAMPLE_DOC_ID}' ingested into '{namespace}'.")


def estimate_tokens(prompt_text: str, answer_text: str) -> Dict[str, int]:
    """Estimate token usage based on character length (approx 4 chars per token)."""
    return {
        "prompt_tokens":     int(len(prompt_text) / 4),
        "completion_tokens": int(len(answer_text) / 4),
        "total_tokens":      int((len(prompt_text) + len(answer_text)) / 4),
    }


# ─── Gemini Judge ─────────────────────────────────────────────────────────────

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_similarity":          {"type": "number"},
        "faithfulness":               {"type": "number"},
        "answer_completeness":        {"type": "number"},
        "citation_correctness":       {"type": "number"},
        "hallucination_rate_raw":     {"type": "number"},
        "cross_document_consistency": {"type": "number"},
        "explanation":                {"type": "string"},
    },
    "required": [
        "answer_similarity", "faithfulness", "answer_completeness",
        "citation_correctness", "hallucination_rate_raw",
        "cross_document_consistency", "explanation",
    ],
}


def judge_with_gemini(
    question: str,
    expected_answer: str,
    generated_answer: str,
    context_chunks: List[str],
    is_multi_doc: bool = False,
) -> Dict[str, Any]:
    """
    Use Gemini to evaluate generation metrics.

    Scores returned (all 0.0–1.0):
      - answer_similarity        : semantic match to expected answer
      - faithfulness             : every claim grounded in context
      - answer_completeness      : fraction of expected key points covered
      - cross_document_consistency: cited facts internally consistent (N/A for single-doc → 0.5)
      - hallucination_rate       : 1 - faithfulness
      - citation_correctness     : correct attribution to source documents
    """
    if not client:
        print("  [Judge] No Gemini client -- using local fallback.")
        return judge_locally(expected_answer, generated_answer)

    combined_context = "\n\n".join(context_chunks) if context_chunks else "(no context provided)"

    multi_doc_instruction = (
        '\n7. "cross_document_consistency": When the answer draws on multiple documents, '
        "are the facts internally consistent and do conflicts get correctly reported? "
        "(1.0 = fully consistent or conflicts properly flagged; 0.0 = contradictions ignored; "
        "use 0.5 if only one document is cited)"
    ) if is_multi_doc else (
        '\n7. "cross_document_consistency": 0.5 (single-document query, metric N/A)'
    )

    prompt = f"""You are an objective AI judge evaluating a RAG (Retrieval-Augmented Generation) system.
Evaluate the generated answer against the expected ground-truth answer and the retrieved context.

QUESTION:
{question}

EXPECTED GROUND-TRUTH ANSWER:
{expected_answer}

GENERATED ANSWER TO EVALUATE:
{generated_answer}

RETRIEVED CONTEXT CHUNKS:
{combined_context}

Score each metric from 0.0 to 1.0:
1. "answer_similarity": Semantic and factual alignment with the expected answer.
2. "faithfulness": Every claim in the generated answer is supported by the context. No hallucinations.
3. "answer_completeness": The generated answer covers all key points from the expected answer.
4. "citation_correctness": The answer correctly attributes information to source documents.
5. "hallucination_rate_raw": Rate of claims NOT supported by the context (inverse of faithfulness).{multi_doc_instruction}
7. "explanation": Brief explanation of your scores.

Respond ONLY with valid JSON matching the schema. No markdown. No extra text."""

    try:
        # Fix 2: Use JSON response_mime_type to guarantee parseable output
        response = client.models.generate_content(
            model=settings.GEMINI_GENERATION_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.0,
            },
        )
        text = response.text.strip()
        # Belt-and-suspenders strip of any accidental fences
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0].strip()

        data = json.loads(text)
        faithfulness = float(data.get("faithfulness", 1.0))
        return {
            "answer_similarity":          float(data.get("answer_similarity", 0.0)),
            "faithfulness":               faithfulness,
            "answer_completeness":        float(data.get("answer_completeness", 0.0)),
            "citation_correctness":       float(data.get("citation_correctness", 0.0)),
            "cross_document_consistency": float(data.get("cross_document_consistency", 0.5)),
            "hallucination_rate":         round(1.0 - faithfulness, 4),
            "explanation":                data.get("explanation", ""),
        }

    except json.JSONDecodeError as exc:
        # Fix 3: detailed error logging — print actual response text for diagnosis
        print(f"  [Judge] JSON parse error: {exc}")
        print(f"  [Judge] Raw response was: {text[:300]!r}")
        return judge_locally(expected_answer, generated_answer)

    except Exception as exc:
        err_str = str(exc).lower()
        if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
            raise
        # Log full error type + message — no silent swallowing
        print(f"  [Judge] Gemini call failed ({type(exc).__name__}): {exc}")
        return judge_locally(expected_answer, generated_answer)


def judge_locally(expected_answer: str, generated_answer: str) -> Dict[str, Any]:
    """Fallback local evaluation using Jaccard similarity and basic heuristics."""
    words_expected  = set(expected_answer.lower().split())
    words_generated = set(generated_answer.lower().split())

    if not words_expected:
        similarity = 0.0
    else:
        intersection = words_expected & words_generated
        union        = words_expected | words_generated
        similarity   = len(intersection) / len(union) if union else 0.0

    if "could not find this information" in generated_answer.lower():
        faithfulness  = 0.0
        completeness  = 0.0
        hallucination = 0.0
        citations     = 0.0
    else:
        faithfulness  = min(1.0, similarity + 0.1) if similarity > 0.3 else 0.5
        completeness  = similarity
        hallucination = round(1.0 - faithfulness, 4)
        citations     = 0.8 if similarity > 0.5 else 0.3

    return {
        "answer_similarity":          round(similarity, 4),
        "faithfulness":               round(faithfulness, 4),
        "answer_completeness":        round(completeness, 4),
        "citation_correctness":       citations,
        "cross_document_consistency": 0.5,
        "hallucination_rate":         hallucination,
        "explanation":                "Local Jaccard similarity fallback.",
    }
