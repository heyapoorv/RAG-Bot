import os
import sys
import time
import json
import asyncio
import numpy as np

# Ensure root directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.retrieval import retrieve_contexts
from evaluation.utils import ensure_sample_ingested, EVAL_NAMESPACE, SAMPLE_DOC_ID

def is_chunk_a_hit(chunk: dict, expected_doc: str, expected_pages: list) -> bool:
    """Determine if a retrieved chunk is a hit for the expected document and pages."""
    chunk_source = chunk.get("source", "").lower()
    chunk_doc_id = chunk.get("document_id", "").lower()
    
    # Check document match
    doc_match = (expected_doc.lower() in chunk_source) or (expected_doc.lower() in chunk_doc_id)
    if not doc_match:
        return False
        
    # Check page match
    chunk_page = chunk.get("page")
    if chunk_page is None:
        return False
        
    # Convert pages to strings to avoid type issues
    chunk_page_str = str(chunk_page)
    expected_pages_str = [str(p) for p in expected_pages]
    
    return chunk_page_str in expected_pages_str

def calculate_metrics(retrieved_chunks: list, expected_doc: str, expected_pages: list, k: int = 5) -> dict:
    """Calculate Recall@k, Precision@k, and Reciprocal Rank."""
    # Truncate retrieved chunks to k
    chunks = retrieved_chunks[:k]
    
    hits = []
    retrieved_pages = set()
    
    for idx, chunk in enumerate(chunks):
        hit = is_chunk_a_hit(chunk, expected_doc, expected_pages)
        hits.append(hit)
        if hit:
            retrieved_pages.add(str(chunk.get("page")))
            
    # Precision@k = hits / k
    precision = sum(hits) / k if k > 0 else 0.0
    
    # Recall@k = retrieved expected pages / total expected pages
    expected_pages_str = [str(p) for p in expected_pages]
    retrieved_expected_pages = [p for p in expected_pages_str if p in retrieved_pages]
    recall = len(retrieved_expected_pages) / len(expected_pages_str) if expected_pages_str else 0.0
    
    # MRR (Reciprocal Rank)
    mrr = 0.0
    for idx, hit in enumerate(hits):
        if hit:
            mrr = 1.0 / (idx + 1)
            break
            
    return {
        "precision": precision,
        "recall": recall,
        "rr": mrr,
        "hit_count": sum(hits)
    }

async def run_evaluation():
    # 1. Ingest sample document if needed
    await ensure_sample_ingested()
    
    # 2. Load benchmark questions
    benchmark_path = os.path.join(os.path.dirname(__file__), "benchmark_questions.json")
    with open(benchmark_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    print(f"\nStarting retrieval evaluation on {len(questions)} queries...")
    print("-" * 60)
    
    latencies = []
    precisions = []
    recalls = []
    rrs = []
    
    results_detail = []
    
    for q_item in questions:
        q_id = q_item["id"]
        question = q_item["question"]
        expected_pages = q_item["expected_pages"]
        
        # Measure latency
        start_time = time.time()
        # Retrieve chunks
        retrieved_chunks = retrieve_contexts(question, top_k=5, namespace=EVAL_NAMESPACE)
        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)
        
        # Calculate metrics
        metrics = calculate_metrics(retrieved_chunks, SAMPLE_DOC_ID, expected_pages, k=5)
        
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        rrs.append(metrics["rr"])
        
        results_detail.append({
            "id": q_id,
            "question": question,
            "expected_pages": expected_pages,
            "latency_ms": round(latency_ms, 2),
            "recall": round(metrics["recall"], 4),
            "precision": round(metrics["precision"], 4),
            "reciprocal_rank": round(metrics["rr"], 4),
            "retrieved_pages": [chunk.get("page") for chunk in retrieved_chunks]
        })
        
        print(f"Q: {question[:40]}... | Recall@5: {metrics['recall']:.2f} | MRR: {metrics['rr']:.2f} | Latency: {latency_ms:.1f}ms")
        
    print("-" * 60)
    
    # Aggregate Metrics
    avg_recall = np.mean(recalls)
    avg_precision = np.mean(precisions)
    mrr = np.mean(rrs)
    
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    report = {
        "avg_recall_at_5": round(float(avg_recall), 4),
        "avg_precision_at_5": round(float(avg_precision), 4),
        "mrr": round(float(mrr), 4),
        "avg_latency_ms": round(float(avg_latency), 2),
        "p95_latency_ms": round(float(p95_latency), 2),
        "queries_evaluated": len(questions),
        "details": results_detail
    }
    
    # Save Report
    report_path = os.path.join(os.path.dirname(__file__), "retrieval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    # Print Summary
    print("\nRETRIEVAL EVALUATION SUMMARY")
    print("=" * 30)
    print(f"Recall@5:      {avg_recall:.4f}")
    print(f"Precision@5:   {avg_precision:.4f}")
    print(f"MRR:           {mrr:.4f}")
    print(f"Avg Latency:   {avg_latency:.2f} ms")
    print(f"P95 Latency:   {p95_latency:.2f} ms")
    print(f"Report saved to: {report_path}")
    print("=" * 30)

if __name__ == "__main__":
    asyncio.run(run_evaluation())
