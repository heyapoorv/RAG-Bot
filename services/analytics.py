import time
from datetime import datetime
from services.db import analytics_collection


def log_query_event(
    question: str,
    answer: str,
    namespace: str,
    latency_ms: float,
    verified: bool = None,
    cache_hit: bool = False,
    retrieved_chunks: list = None,
    reranked_chunks: list = None,
    final_context: str = "",
    model: str = "gemini-2.5-flash"
):
    try:
        analytics_collection.insert_one({
            "question": question,
            "answer": answer,
            "namespace": namespace,
            "latency_ms": latency_ms,
            "verified": verified,
            "cache_hit": cache_hit,
            "timestamp": datetime.utcnow().timestamp(),
            "retrieved_chunks": retrieved_chunks or [],
            "reranked_chunks": reranked_chunks or [],
            "final_context": final_context,
            "model": model
        })
    except Exception as e:
        # Never break RAG flow due to logging failure
        import logging
        logging.error(f"[Analytics Error] {e}")