def compute_dynamic_topk(question: str) -> int:
    """
    Dynamic retrieval depth based on query complexity.
    """

    q = question.lower()

    if any(word in q for word in [
        "compare",
        "difference",
        "all",
        "list",
        "summarize",
        "coverage",
        "benefits",
        "conditions"
    ]):
        return 10

    if len(question.split()) > 18:
        return 8

    return 5