"""
evaluation/evaluate_phase3.py

Phase 3 combined evaluation script.

Runs all 8 target metrics across two benchmark sets:
  - benchmark_questions.json  : original 8 single-doc questions (baseline)
  - benchmark_hard.json       : 40-question hard set (multi-doc, conflicting, follow-up, ambiguous)

Reports:
  Recall@5, Precision@5, MRR
  Answer Similarity
  Faithfulness
  Hallucination Rate
  Citation Correctness
  Answer Completeness
  Cross-Document Consistency

CLI flags:
  --force-clean   Purge contaminated namespaces before evaluation (deletes all
                  vectors in the affected namespace and re-ingests).
  --quick         Run only the first 10 hard-benchmark questions (sanity check).
"""
from __future__ import annotations
import argparse

import os
import sys
import time
import json
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.answer import answer_single_question
from services.cache import invalidate_namespace
from services.retrieval import retrieve_contexts
from evaluation.utils import (
    ensure_sample_ingested,
    judge_with_gemini,
    judge_locally,
    estimate_tokens,
    EVAL_NAMESPACE,
    EVAL_NAMESPACE_MULTIDOC,
    SAMPLE_DOC_ID,
)
from evaluation.evaluate_retrieval import calculate_metrics

# ─── Constants ────────────────────────────────────────────────────────────────

EVAL_DOCS     = ["policy_a.txt", "policy_b.txt"]
# 35s between synthesis calls: synthesis (1 call) + judge (1 call) = 2 calls per question
REQUEST_DELAY = 35.0
JUDGE_DELAY   = 10.0

# ─── Ingestion Helpers ────────────────────────────────────────────────────────

async def ensure_hard_docs_ingested(force_clean: bool = False):
    """
    Ensure policy_a.txt and policy_b.txt are indexed in the multidoc eval namespace.

    Runs a purity check FIRST. If unexpected documents are found:
      - force_clean=False  → raises RuntimeError (evaluation blocked)
      - force_clean=True   → purges the namespace, then re-ingests all expected docs
    """
    from services.ingestion import ingest_text
    from services.vectordb import index as pinecone_index
    from evaluation.utils import check_namespace_purity, _ALLOWED_DOCS

    # ── Purity check BEFORE checking vector count ────────────────────────────
    # This may purge the namespace if force_clean=True, resetting vector_count to 0
    check_namespace_purity(
        namespace=EVAL_NAMESPACE_MULTIDOC,
        allowed_docs=_ALLOWED_DOCS[EVAL_NAMESPACE_MULTIDOC],
        force_clean=force_clean,
    )

    stats    = pinecone_index.describe_index_stats()
    ns_stats = stats.get("namespaces", {}).get(EVAL_NAMESPACE_MULTIDOC, {})
    vector_count = ns_stats.get("vector_count", 0)

    if vector_count >= 10:
        print(f"  Hard-benchmark docs already indexed ({vector_count} vectors in '{EVAL_NAMESPACE_MULTIDOC}').")
        return

    base_dir = os.path.join(os.path.dirname(__file__), "sample_docs")
    for doc_name in EVAL_DOCS:
        doc_path = os.path.join(base_dir, doc_name)
        if not os.path.exists(doc_path):
            print(f"  WARNING: {doc_name} not found at {doc_path}. Skipping.")
            continue
        ext = doc_name.rsplit(".", 1)[-1].lower()
        print(f"  Ingesting {doc_name} into namespace '{EVAL_NAMESPACE_MULTIDOC}'...")
        try:
            await ingest_text(
                file_path=doc_path,
                file_type=ext,
                namespace=EVAL_NAMESPACE_MULTIDOC,
                use_semantic=True,
                original_filename=doc_name,
            )
            print(f"  OK  {doc_name} ingested.")
        except Exception as e:
            print(f"  FAIL  Failed to ingest {doc_name}: {e}")


# ─── Pipeline with Retry ──────────────────────────────────────────────────────

async def run_with_retry(
    question: str,
    namespace: str,
    retries: int = 3,
    backoff: int = 70,
) -> Dict:
    for attempt in range(retries):
        try:
            return await answer_single_question(question, namespace=namespace)
        except Exception as e:
            err = str(e).lower()
            if "resource_exhausted" in err or "429" in err or "quota" in err:
                wait = backoff * (attempt + 1)
                print(f"\n  [Rate Limit] Sleeping {wait}s (attempt {attempt+1}/{retries})...")
                await asyncio.sleep(wait)
            else:
                # Non-rate-limit errors: log and re-raise immediately
                print(f"\n  [Pipeline Error] {type(e).__name__}: {e}")
                raise
    return await answer_single_question(question, namespace=namespace)


# --- Retrieval Evaluation -----------------------------------------------------

def run_retrieval_eval(questions: List[Dict], label: str, namespace: str) -> Dict:
    print(f"\n{'-'*60}")
    print(f"RETRIEVAL EVAL -- {label} (ns={namespace})")
    print(f"{'-'*60}")

    latencies, precisions, recalls, rrs = [], [], [], []

    for q_item in questions:
        question       = q_item["question"]
        expected_pages = q_item.get("expected_pages", [])
        expected_docs  = q_item.get("expected_documents", [SAMPLE_DOC_ID])
        primary_doc    = expected_docs[0] if expected_docs else SAMPLE_DOC_ID

        t0 = time.time()
        chunks = retrieve_contexts(question, top_k=5, namespace=namespace)
        latencies.append((time.time() - t0) * 1000)

        m = calculate_metrics(chunks, primary_doc, expected_pages, k=5)
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        rrs.append(m["rr"])
        time.sleep(3) # Avoid embedding API rate limit

    avg_recall    = float(np.mean(recalls))
    avg_precision = float(np.mean(precisions))
    mrr           = float(np.mean(rrs))
    avg_latency   = float(np.mean(latencies))
    p95_latency   = float(np.percentile(latencies, 95))

    print(f"  Recall@5:    {avg_recall:.4f}")
    print(f"  Precision@5: {avg_precision:.4f}")
    print(f"  MRR:         {mrr:.4f}")
    print(f"  Avg Latency: {avg_latency:.1f} ms")

    return {
        "recall_at_5":    round(avg_recall, 4),
        "precision_at_5": round(avg_precision, 4),
        "mrr":            round(mrr, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
    }


# --- Generation Evaluation ----------------------------------------------------

async def run_generation_eval(questions: List[Dict], label: str, namespace: str) -> Dict:
    print(f"\n{'-'*60}")
    print(f"GENERATION EVAL -- {label} (ns={namespace})")
    print(f"{'-'*60}")

    all_runs: List[Dict] = []

    for idx, q_item in enumerate(questions):
        question        = q_item["question"]
        expected_answer = q_item.get("expected_answer", "")
        expected_pages  = q_item.get("expected_pages", [])
        expected_docs   = q_item.get("expected_documents", [SAMPLE_DOC_ID])
        is_multi_doc    = len(expected_docs) > 1
        category        = q_item.get("category", "standard")

        if idx > 0:
            await asyncio.sleep(REQUEST_DELAY)

        t0 = time.time()
        try:
            result = await run_with_retry(question, namespace=namespace)
        except Exception as e:
            print(f"  ERROR on Q{idx+1}: {e}")
            result = {
                "answer": "I could not find this information in the document.",
                "citations": [], "sources": [],
                "evidence_by_document": {}, "conflicts": [],
                "synthesis_mode": "single_doc",
            }
        latency_ms = (time.time() - t0) * 1000

        # Extract cited pages
        citations = result.get("citations", [])
        cited_pages = {str(c.get("page")) for c in citations if c.get("page") is not None}
        expected_pages_str = {str(p) for p in expected_pages}
        cit_precision = (
            len(cited_pages & expected_pages_str) / len(cited_pages)
            if cited_pages else 0.0
        )

        # Context chunks for judge
        context_texts = [c.get("highlight", "") for c in citations]

        await asyncio.sleep(JUDGE_DELAY)

        scores = None
        for attempt in range(3):
            try:
                scores = judge_with_gemini(
                    question=question,
                    expected_answer=expected_answer,
                    generated_answer=result.get("answer", ""),
                    context_chunks=context_texts,
                    is_multi_doc=is_multi_doc,
                )
                break
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                    wait = 70 * (attempt + 1)
                    print(f"  [Judge Rate Limit] Sleeping {wait}s (attempt {attempt+1}/3)...")
                    await asyncio.sleep(wait)
                else:
                    print(f"  Judge failed: {e}. Using local fallback.")
                    scores = judge_locally(expected_answer, result.get("answer", ""))
                    break
        
        if scores is None:
            scores = judge_locally(expected_answer, result.get("answer", ""))

        # Runtime generation scores from the pipeline
        runtime_scores = result.get("generation_scores") or {}
        runtime_faithfulness    = float(runtime_scores.get("faithfulness",    1.0))
        runtime_completeness    = float(runtime_scores.get("completeness",    1.0))
        runtime_cross_doc       = float(runtime_scores.get("cross_doc_consistency", 1.0))

        conflicts_detected = len(result.get("conflicts", [])) > 0
        synthesis_mode     = result.get("synthesis_mode", "single_doc")

        run = {
            "id":                          q_item.get("id", f"q{idx+1}"),
            "category":                    category,
            "question":                    question[:60],
            "answer":                      result.get("answer", "")[:200],
            "synthesis_mode":              synthesis_mode,
            "document_count":              result.get("document_count", 1),
            "conflicts_detected":          conflicts_detected,
            "expected_conflict":           q_item.get("requires_conflict_detection", False),
            "structured_sources_count":    len(result.get("structured_sources") or []),
            "latency_ms":                  round(latency_ms, 1),
            "citation_page_precision":     round(cit_precision, 3),
            # LLM judge scores
            "answer_similarity":           scores.get("answer_similarity", 0.0),
            "faithfulness":                scores.get("faithfulness", 0.0),
            "answer_completeness":         scores.get("answer_completeness", 0.0),
            "citation_correctness":        scores.get("citation_correctness", 0.0),
            "cross_doc_consistency":       scores.get("cross_document_consistency", 0.5),
            "hallucination_rate":          scores.get("hallucination_rate", 0.0),
            # Runtime heuristic scores
            "runtime_faithfulness":        round(runtime_faithfulness, 4),
            "runtime_completeness":        round(runtime_completeness, 4),
            "runtime_cross_doc":           round(runtime_cross_doc, 4),
        }
        all_runs.append(run)

        mode_tag = f"[{synthesis_mode}]"
        conflict_tag = "!" if conflicts_detected else " "
        src_tag = f"+{run['structured_sources_count']}src" if run["structured_sources_count"] else ""
        print(
            f"  {conflict_tag} Q{idx+1:02d} {mode_tag:<12} "
            f"Sim:{scores.get('answer_similarity',0):.2f} "
            f"Faith:{scores.get('faithfulness',0):.2f}(r{runtime_faithfulness:.2f}) "
            f"Compl:{scores.get('answer_completeness',0):.2f}(r{runtime_completeness:.2f}) "
            f"XDoc:{scores.get('cross_document_consistency',0.5):.2f} "
            f"{src_tag} | {latency_ms:.0f}ms"
        )

    # Aggregate
    def _avg(key): return round(float(np.mean([r[key] for r in all_runs])), 4)

    return {
        "avg_answer_similarity":          _avg("answer_similarity"),
        "avg_faithfulness":               _avg("faithfulness"),
        "avg_answer_completeness":        _avg("answer_completeness"),
        "avg_citation_correctness":       _avg("citation_correctness"),
        "avg_cross_doc_consistency":      _avg("cross_doc_consistency"),
        "avg_hallucination_rate":         _avg("hallucination_rate"),
        "avg_latency_ms":                 _avg("latency_ms"),
        "avg_runtime_faithfulness":       _avg("runtime_faithfulness"),
        "avg_runtime_completeness":       _avg("runtime_completeness"),
        "avg_runtime_cross_doc":          _avg("runtime_cross_doc"),
        "avg_structured_sources_count":   _avg("structured_sources_count"),
        "p95_latency_ms":                 round(float(np.percentile([r["latency_ms"] for r in all_runs], 95)), 2),
        "multi_doc_activation_rate":      round(sum(1 for r in all_runs if r["synthesis_mode"] == "multi_doc") / len(all_runs), 4),
        "conflict_detection_rate":        round(sum(1 for r in all_runs if r["conflicts_detected"]) / len(all_runs), 4),
        "details":                        all_runs,
    }


# ─── Category Breakdown ───────────────────────────────────────────────────────

def print_category_breakdown(details: List[Dict]):
    from collections import defaultdict
    by_cat: Dict[str, List] = defaultdict(list)
    for r in details:
        by_cat[r["category"]].append(r)

    print("\n  Category Breakdown:")
    header = f"    {'Category':<22} {'n':>3}  {'Sim':>5} {'Faith':>6} {'Compl':>6} {'XDoc':>5} {'CitPrec':>8} {'CDR%':>6} {'SrcCount':>9}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for cat, items in sorted(by_cat.items()):
        sim   = np.mean([i["answer_similarity"]   for i in items])
        faith = np.mean([i["faithfulness"]         for i in items])
        compl = np.mean([i["answer_completeness"]  for i in items])
        xdoc  = np.mean([i["cross_doc_consistency"] for i in items])
        cit   = np.mean([i["citation_page_precision"] for i in items])
        cdr   = sum(1 for i in items if i["conflicts_detected"]) / len(items)
        src   = np.mean([i.get("structured_sources_count", 0) for i in items])
        print(
            f"    {cat:<22} {len(items):>3}  "
            f"{sim:>5.2f} {faith:>6.2f} {compl:>6.2f} {xdoc:>5.2f} "
            f"{cit:>8.3f} {cdr:>6.1%} {src:>9.1f}"
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

async def run_evaluation(force_clean: bool = False, quick: bool = False):
    print("=" * 65)
    print("PHASE 3 EVALUATION -- Answer Quality & Multi-Document Reasoning")
    if force_clean:
        print("  MODE: --force-clean  (contaminated namespaces will be purged)")
    if quick:
        print("  MODE: --quick  (first 10 hard questions only)")
    print("=" * 65)

    # 1. Setup ─ namespace purity is enforced inside each ensure_* call
    print("\n[1/5] Ensuring evaluation documents are indexed (with purity checks)...")
    await ensure_sample_ingested(force_clean=force_clean)
    await ensure_hard_docs_ingested(force_clean=force_clean)

    # 2. Invalidate cache for both namespaces
    print("\n[2/5] Invalidating semantic cache...")
    invalidate_namespace(EVAL_NAMESPACE)
    invalidate_namespace(EVAL_NAMESPACE_MULTIDOC)

    # 3. Load benchmarks
    eval_dir = os.path.dirname(__file__)
    with open(os.path.join(eval_dir, "benchmark_questions.json"), encoding="utf-8") as f:
        baseline_qs = json.load(f)
    with open(os.path.join(eval_dir, "benchmark_hard.json"), encoding="utf-8") as f:
        hard_qs_full = json.load(f)

    # Apply --quick mode: first 10 hard questions only
    hard_qs = hard_qs_full[:10] if quick else hard_qs_full
    mode_tag = f"QUICK ({len(hard_qs)})" if quick else f"FULL ({len(hard_qs)})"

    print(f"\n[3/5] Running retrieval evaluation "
          f"({len(baseline_qs)} baseline + {len(hard_qs)} hard [{mode_tag}])...")
    # Baseline uses isolated single-doc namespace; hard uses multidoc namespace
    retrieval_baseline = run_retrieval_eval(baseline_qs, "Baseline (8 questions)", namespace=EVAL_NAMESPACE)
    retrieval_hard     = run_retrieval_eval(hard_qs,     f"Hard ({mode_tag})",    namespace=EVAL_NAMESPACE_MULTIDOC)

    print(f"\n[4/5] Running generation evaluation...")
    print(f"  Note: {REQUEST_DELAY}s delay between queries (synthesis + judge = 2 API calls).")
    print(f"  Baseline ({len(baseline_qs)} questions, namespace={EVAL_NAMESPACE})...")
    gen_baseline = await run_generation_eval(baseline_qs, "Baseline", namespace=EVAL_NAMESPACE)

    est_min = int(len(hard_qs) * REQUEST_DELAY / 60) + 1
    print(f"\n  Hard benchmark ({len(hard_qs)} questions [{mode_tag}], "
          f"namespace={EVAL_NAMESPACE_MULTIDOC})...")
    print(f"  (Estimated time: ~{est_min} minutes due to rate limiting)")
    gen_hard = await run_generation_eval(hard_qs, f"Hard Benchmark {mode_tag}", namespace=EVAL_NAMESPACE_MULTIDOC)
    print_category_breakdown(gen_hard["details"])

    # 5. Final report
    print(f"\n[5/5] Generating report...")

    report = {
        "phase": 3,
        "retrieval": {
            "baseline": retrieval_baseline,
            "hard":     retrieval_hard,
        },
        "generation": {
            "baseline": {k: v for k, v in gen_baseline.items() if k != "details"},
            "hard":     {k: v for k, v in gen_hard.items() if k != "details"},
        },
        "details": {
            "baseline": gen_baseline.get("details", []),
            "hard":     gen_hard.get("details", []),
        },
    }

    report_path = os.path.join(eval_dir, "phase3_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ── Print Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 3 EVALUATION SUMMARY")
    print("=" * 65)

    print("\n  RETRIEVAL (Baseline -- 8 questions)")
    print(f"    Recall@5:          {retrieval_baseline['recall_at_5']:.4f}")
    print(f"    Precision@5:       {retrieval_baseline['precision_at_5']:.4f}")
    print(f"    MRR:               {retrieval_baseline['mrr']:.4f}")
    print(f"    Avg Latency:       {retrieval_baseline['avg_latency_ms']:.1f} ms")

    print("\n  RETRIEVAL (Hard -- 40 questions)")
    print(f"    Recall@5:          {retrieval_hard['recall_at_5']:.4f}")
    print(f"    Precision@5:       {retrieval_hard['precision_at_5']:.4f}")
    print(f"    MRR:               {retrieval_hard['mrr']:.4f}")
    print(f"    Avg Latency:       {retrieval_hard['avg_latency_ms']:.1f} ms")

    print("\n  GENERATION (Baseline -- 8 questions)")
    gb = gen_baseline
    print(f"    Answer Similarity:          {gb['avg_answer_similarity']:.4f}")
    print(f"    Faithfulness:               {gb['avg_faithfulness']:.4f}")
    print(f"    Answer Completeness:        {gb['avg_answer_completeness']:.4f}")
    print(f"    Hallucination Rate:         {gb['avg_hallucination_rate']:.4f}")
    print(f"    Citation Correctness:       {gb['avg_citation_correctness']:.4f}")
    print(f"    Cross-Doc Consistency:      {gb['avg_cross_doc_consistency']:.4f}")

    print("\n  GENERATION (Hard -- 40 questions)")
    gh = gen_hard
    print(f"    Answer Similarity:           {gh['avg_answer_similarity']:.4f}")
    print(f"    Faithfulness (LLM judge):    {gh['avg_faithfulness']:.4f}")
    print(f"    Faithfulness (runtime):      {gh['avg_runtime_faithfulness']:.4f}")
    print(f"    Answer Completeness (LLM):   {gh['avg_answer_completeness']:.4f}")
    print(f"    Completeness (runtime):      {gh['avg_runtime_completeness']:.4f}")
    print(f"    Hallucination Rate:          {gh['avg_hallucination_rate']:.4f}")
    print(f"    Citation Correctness:        {gh['avg_citation_correctness']:.4f}")
    print(f"    Cross-Doc Consistency:       {gh['avg_cross_doc_consistency']:.4f}")
    print(f"    Runtime Cross-Doc:           {gh['avg_runtime_cross_doc']:.4f}")
    print(f"    Multi-Doc Activation Rate:   {gh['multi_doc_activation_rate']:.4f}")
    print(f"    Conflict Detection Rate:     {gh['conflict_detection_rate']:.4f}")
    print(f"    Avg Structured Sources:      {gh['avg_structured_sources_count']:.2f} per answer")
    print(f"    Avg Latency:                 {gh['avg_latency_ms']:.1f} ms")

    print(f"\n  Report saved to: {report_path}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3 RAG Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m evaluation.evaluate_phase3              # full run\n"
            "  python -m evaluation.evaluate_phase3 --quick     # 10-question sanity check\n"
            "  python -m evaluation.evaluate_phase3 --force-clean        # purge dirty namespaces\n"
            "  python -m evaluation.evaluate_phase3 --force-clean --quick  # clean + quick\n"
        ),
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        default=False,
        help=(
            "Purge ALL vectors from contaminated evaluation namespaces before running. "
            "Use when a namespace contains unexpected documents from a previous test run."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Run only the first 10 hard-benchmark questions (fast sanity check, ~6 min).",
    )
    args = parser.parse_args()
    asyncio.run(run_evaluation(force_clean=args.force_clean, quick=args.quick))
