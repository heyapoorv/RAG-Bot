# services/llm.py
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_answer(question: str, contexts: list):
    """
    Generate an answer strictly from retrieved contexts.
    If answer not found in context, returns "NOT_FOUND_IN_DOCS".
    """
    # Combine all chunk texts into context
    context_text = "\n".join(
        [f"[{c['source']}] {c['chunk_text']}" for c in contexts]
    )

    prompt = f"""
You are a strict AI assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not in the context, respond: "NOT_FOUND_IN_DOCS".
- Do NOT hallucinate or make up information.
- Keep answers concise and factual.

Context:
{context_text}

Question: {question}
Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt
    )

    return response.text.strip()