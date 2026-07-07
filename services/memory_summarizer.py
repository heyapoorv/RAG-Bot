from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def summarize_history(history: list) -> str:
    if not history:
        return ""

    text = "\n".join([
        f"{m['role']}: {m['content']}"
        for m in history[-8:]
    ])

    prompt = f"""
Summarize this conversation for a retrieval system.

Keep ONLY key entities, topics, insurance terms.

Conversation:
{text}

Summary:
"""

    try:
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return res.text.strip()
    except:
        return ""