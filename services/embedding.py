import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

from config import settings

def embed_texts(texts):
    """
    ALWAYS returns List[List[float]]
    Sends texts in batches to reduce API calls.
    """
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return []

    embed_dim = settings.GEMINI_EMBED_DIM
    results = [[0.0] * embed_dim] * len(texts)
    
    # Map valid texts to their original indices
    valid_texts = []
    valid_indices = []
    
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts.append(text)
            valid_indices.append(i)

    if not valid_texts:
        return results

    # Batch process in chunks of 100 to avoid API limits
    batch_size = 100
    for i in range(0, len(valid_texts), batch_size):
        batch_texts = valid_texts[i:i + batch_size]
        batch_indices = valid_indices[i:i + batch_size]

        try:
            res = client.models.embed_content(
                model=settings.GEMINI_EMBEDDING_MODEL,
                contents=batch_texts,
                config={
                    "output_dimensionality": embed_dim
                }
            )

            for j, emb in enumerate(res.embeddings):
                vector = emb.values
                if len(vector) != embed_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: got {len(vector)} (expected {embed_dim})"
                    )
                results[batch_indices[j]] = vector

        except Exception as e:
            # If batch fails, we log and return 0s for this batch (or raise depending on policy)
            # We'll re-raise so the ingestion job catches it and can fail gracefully
            raise RuntimeError(f"Batch embedding failed: {e}")

    return results