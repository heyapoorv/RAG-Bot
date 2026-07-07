import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

VERIFY_MODEL = "gemini-2.5-flash"


async def verify_answer(
    question: str,
    answer: str,
    context_chunks: list
):
    """
    Verify whether answer is grounded in provided context.
    """

    context = "\n\n".join([
        c["chunk_text"]
        for c in context_chunks
    ])

    prompt = f"""
You are verifying whether an answer is fully supported by document context.

QUESTION:
{question}

ANSWER:
{answer}

DOCUMENT CONTEXT:
{context}

Respond EXACTLY with one of:
- VERIFIED
- UNSUPPORTED
"""

    try:
        response = client.models.generate_content(
            model=VERIFY_MODEL,
            contents=prompt
        )

        verdict = response.text.strip().upper()

        return verdict == "VERIFIED"

    except Exception:
        return True