import re
from services.memory import get_history

async def rewrite_query(question: str, session_id: str | None = None) -> str:
    """
    Rewrites follow-up / ambiguous user query deterministically.
    """
    if not session_id:
        return question

    history = get_history(session_id)
    if not history:
        return question

    stop_words = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
        "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
        "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", 
        "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", 
        "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", 
        "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", 
        "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", 
        "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", 
        "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "yes", "could", "would"
    }

    history_words = []
    for msg in history[-4:]:
        content = msg.get("content", "")
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        for w in words:
            if w not in stop_words:
                history_words.append(w)
                
    seen = set()
    unique_history_words = []
    for w in reversed(history_words):
        if w not in seen:
            seen.add(w)
            unique_history_words.append(w)
    
    unique_history_words.reverse()

    q_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
    words_to_add = [w for w in unique_history_words if w not in q_words][-5:]
    
    if words_to_add:
        return f"{question} {' '.join(words_to_add)}"
    return question