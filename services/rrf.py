from collections import defaultdict

def rrf_fusion(rankings: list[list[dict]], k: int = 60):
    """
    rankings = list of retrieval result lists
    each list = [{"source":..., "chunk_text":..., "score":...}]
    """

    scores = defaultdict(float)
    docs = {}

    for rank_list in rankings:
        for rank, doc in enumerate(rank_list):
            key = (doc.get("source"), doc.get("chunk_text", "")[:120])

            # store doc once
            docs[key] = doc

            # RRF score
            scores[key] += 1 / (k + rank + 1)
            
            # Preserve highest constituent scores if multiple matches
            if doc.get("dense_score") is not None:
                docs[key]["dense_score"] = max(docs[key].get("dense_score") or 0.0, doc["dense_score"])
            if doc.get("bm25_score") is not None:
                docs[key]["bm25_score"] = max(docs[key].get("bm25_score") or 0.0, doc["bm25_score"])

    # sort by fused score
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    result = []
    for key, score in fused:
        docs[key]["rrf_score"] = score
        docs[key]["score"] = score  # For backward compatibility if anything relies on 'score'
        result.append(docs[key])

    return result