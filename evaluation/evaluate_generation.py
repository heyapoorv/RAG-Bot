import os
import sys
import time
import json
import asyncio
import numpy as np

# Ensure root directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.answer import answer_single_question
from services.cache import invalidate_namespace
from config import settings
from evaluation.utils import (
    ensure_sample_ingested,
    estimate_tokens,
    judge_with_gemini,
    judge_locally,
    EVAL_NAMESPACE,
    SAMPLE_DOC_ID
)

# Configurable delay between requests to respect Gemini 5 RPM rate limit
REQUEST_DELAY_SECONDS = 15.0

def extract_cited_pages(result: dict) -> set:
    """Extract the set of page numbers cited in the RAG result."""
    citations = result.get("citations", [])
    pages = set()
    for cit in citations:
        page = cit.get("page")
        if page is not None:
            pages.add(str(page))
    return pages

def calculate_citation_correctness(cited_pages: set, expected_pages: list) -> float:
    """Calculate the overlap between cited pages and expected pages."""
    expected_pages_str = {str(p) for p in expected_pages}
    if not expected_pages_str:
        return 1.0
    if not cited_pages:
        return 0.0
    
    # Precision of citations: what fraction of cited pages are expected
    precision = len(intersection) / len(cited_pages) if (intersection := cited_pages.intersection(expected_pages_str)) else 0.0
    return precision

def count_gemini_calls(result: dict, highest_score: float) -> int:
    """Estimate number of Gemini API calls made in the RAG pipeline request."""
    if result.get("cache_hit", False):
        return 0
        
    calls = 0
    # 1. Query rewrite call
    if result.get("rewritten_query") and result["rewritten_query"] != result["question"]:
        calls += 1
        
    # 2. Multi-query expansion call
    if highest_score < 0.65:
        calls += 1
        
    # 3. Generation call
    calls += 1
    
    # 4. Verification call
    if settings.VERIFICATION_ENABLED and result.get("verified") is not None:
        calls += 1
        
    return calls

async def run_pipeline_with_retry(question: str, namespace: str, retries: int = 5, backoff_seconds: int = 60) -> dict:
    """Run RAG pipeline with automatic retries on rate limits (429)."""
    for attempt in range(retries):
        try:
            result = await answer_single_question(question, namespace=namespace)
            # If answer contains the fallback but did not raise an exception, or if it succeeded
            # Check if it hit local Ollama fallback. If so, let's see if we should consider that a success or retry.
            # Usually, if Gemini is rate limited, answer_single_question catches it internally and calls Ollama.
            # To get a true Gemini baseline, we should try to avoid hitting Ollama if we can retry and get Gemini to answer!
            # But since answer_single_question catches the error, we check the logs or if it fell back to Ollama.
            # In our evaluation, if the answer text is the local fallback or if we want to ensure Gemini,
            # we can run it. For now, since the pipeline catches it, let's run it.
            return result
        except Exception as e:
            err_msg = str(e).lower()
            if "resource_exhausted" in err_msg or "429" in err_msg or "quota" in err_msg:
                print(f"\n[Rate Limit] Hit Gemini rate limit. Sleeping for {backoff_seconds} seconds (Attempt {attempt+1}/{retries})...")
                await asyncio.sleep(backoff_seconds)
            else:
                raise e
    # Fallback return
    return await answer_single_question(question, namespace=namespace)

async def run_evaluation():
    # 1. Ingest sample document if needed
    await ensure_sample_ingested()
    
    # 2. Invalidate cache to establish clean baseline
    print("Invalidating semantic cache...")
    invalidate_namespace(EVAL_NAMESPACE)
    
    # 3. Load benchmark questions
    benchmark_path = os.path.join(os.path.dirname(__file__), "benchmark_questions.json")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    print(f"\nStarting generation evaluation on {len(questions)} queries...")
    print(f"Adding a {REQUEST_DELAY_SECONDS}s delay between queries to respect API rate limits.")
    print("-" * 75)
    
    all_runs = []
    
    # Pass 1: Cache Misses
    print("\n--- Pass 1: Cache Misses ---")
    for idx, q_item in enumerate(questions):
        question = q_item["question"]
        expected_answer = q_item["expected_answer"]
        expected_pages = q_item["expected_pages"]
        
        # Sleep to pace API calls
        if idx > 0:
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            
        start_time = time.time()
        result = await run_pipeline_with_retry(question, namespace=EVAL_NAMESPACE)
        latency_ms = (time.time() - start_time) * 1000
        
        # Extract citations
        cited_pages = extract_cited_pages(result)
        cit_correctness = calculate_citation_correctness(cited_pages, expected_pages)
        
        # Get highest score
        citations = result.get("citations", [])
        highest_score = max([c.get("score") or 0.0 for c in citations]) if citations else 0.0
        gemini_calls = count_gemini_calls(result, highest_score)
        
        # Estimate tokens
        context_texts = [c.get("highlight", "") for c in result.get("citations", [])]
        combined_context = "\n".join(context_texts)
        tokens = estimate_tokens(combined_context + question, result.get("answer", ""))
        
        # Short sleep before calling judge to avoid rate limits
        await asyncio.sleep(5.0)
        
        # LLM-as-a-judge scoring
        judge_scores = judge_with_gemini(
            question=question,
            expected_answer=expected_answer,
            generated_answer=result.get("answer", ""),
            context_chunks=context_texts
        )
        
        run_data = {
            "question": question,
            "expected_pages": expected_pages,
            "answer": result.get("answer", ""),
            "cache_hit": False,
            "latency_ms": latency_ms,
            "cit_correctness": cit_correctness,
            "gemini_calls": gemini_calls,
            "tokens": tokens["total_tokens"],
            "judge_similarity": judge_scores["answer_similarity"],
            "judge_hallucination_rate": judge_scores["hallucination_rate"],
            "judge_citation_correctness": judge_scores["citation_correctness"]
        }
        all_runs.append(run_data)
        
        print(f"Q: {question[:35]}... | Similarity: {judge_scores['answer_similarity']:.2f} | Citations: {cit_correctness:.2f} | Latency: {latency_ms:.1f}ms")

    # Pass 2: Cache Hits (Run first 4 questions again)
    print("\n--- Pass 2: Cache Hits (Duplicate Queries) ---")
    for idx, q_item in enumerate(questions[:4]):
        question = q_item["question"]
        expected_answer = q_item["expected_answer"]
        expected_pages = q_item["expected_pages"]
        
        if idx > 0:
            await asyncio.sleep(2.0) # Faster sleep since cache hits don't call external APIs
            
        start_time = time.time()
        result = await answer_single_question(question, namespace=EVAL_NAMESPACE)
        latency_ms = (time.time() - start_time) * 1000
        
        cited_pages = extract_cited_pages(result)
        cit_correctness = calculate_citation_correctness(cited_pages, expected_pages)
        
        tokens = estimate_tokens(question, result.get("answer", ""))
        
        # Retrieve previous judge scores for this question to keep metrics accurate
        prev_run = [r for r in all_runs if r["question"] == question and not r["cache_hit"]][0]
        
        run_data = {
            "question": question,
            "expected_pages": expected_pages,
            "answer": result.get("answer", ""),
            "cache_hit": True,
            "latency_ms": latency_ms,
            "cit_correctness": cit_correctness,
            "gemini_calls": 0,
            "tokens": tokens["total_tokens"],
            "judge_similarity": prev_run["judge_similarity"],
            "judge_hallucination_rate": prev_run["judge_hallucination_rate"],
            "judge_citation_correctness": prev_run["judge_citation_correctness"]
        }
        all_runs.append(run_data)
        print(f"Q (Cached): {question[:25]}... | Latency: {latency_ms:.1f}ms (Cache Hit: {result.get('cache_hit')})")

    print("-" * 75)
    
    # Aggregate Metrics
    latencies = [r["latency_ms"] for r in all_runs]
    cache_hits = [r["cache_hit"] for r in all_runs]
    gemini_calls_list = [r["gemini_calls"] for r in all_runs]
    tokens_list = [r["tokens"] for r in all_runs]
    
    miss_runs = [r for r in all_runs if not r["cache_hit"]]
    similarities = [r["judge_similarity"] for r in miss_runs]
    hallucination_rates = [r["judge_hallucination_rate"] for r in miss_runs]
    citation_correctness_scores = [r["cit_correctness"] for r in miss_runs]
    
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    cache_hit_ratio = sum(cache_hits) / len(all_runs)
    avg_gemini_calls = np.mean(gemini_calls_list)
    avg_tokens = np.mean(tokens_list)
    
    avg_similarity = np.mean(similarities)
    avg_hallucination = np.mean(hallucination_rates)
    avg_citation = np.mean(citation_correctness_scores)
    
    report = {
        "avg_latency_ms": round(float(avg_latency), 2),
        "p95_latency_ms": round(float(p95_latency), 2),
        "cache_hit_ratio": round(float(cache_hit_ratio), 4),
        "avg_gemini_calls_per_request": round(float(avg_gemini_calls), 2),
        "avg_tokens_per_request": round(float(avg_tokens), 1),
        "avg_answer_similarity": round(float(avg_similarity), 4),
        "avg_hallucination_rate": round(float(avg_hallucination), 4),
        "avg_citation_correctness": round(float(avg_citation), 4),
        "total_requests": len(all_runs),
        "details": all_runs
    }
    
    # Save Report
    report_path = os.path.join(os.path.dirname(__file__), "generation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("\nCURRENT SYSTEM BASELINE")
    print("=" * 35)
    print("Retrieval:")
    print("  Recall@5:            [Run evaluate_retrieval.py to generate]")
    print("  Precision@5:         [Run evaluate_retrieval.py to generate]")
    print("  MRR:                 [Run evaluate_retrieval.py to generate]")
    print("\nGeneration:")
    print(f"  Answer similarity:   {avg_similarity:.4f}")
    print(f"  Hallucination rate:  {avg_hallucination:.4f}")
    print(f"  Citation correctness:{avg_citation:.4f}")
    print("\nPerformance:")
    print(f"  Average latency:     {avg_latency:.2f} ms")
    print(f"  P95 latency:         {p95_latency:.2f} ms")
    print(f"  Gemini calls/request:{avg_gemini_calls:.2f}")
    print(f"  Token usage/request: {avg_tokens:.1f}")
    print(f"  Cache hit ratio:     {cache_hit_ratio:.4f}")
    print("=" * 35)
    print(f"Report saved to: {report_path}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
