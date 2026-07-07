def is_valid_rag_query(query: str) -> bool:
    q = query.lower().strip()

    # too short
    if len(q) < 3:
        return False

    # garbage/meta queries
    bad_patterns = [
        "compare all",
        "tell everything",
        "do all",
        "explain all",
        "what about it",
        "and this",
        "this one",
        "everything"
    ]

    if any(p in q for p in bad_patterns):
        return False

    return True