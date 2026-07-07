import os
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")

# Initialize client
client = genai.Client(api_key=api_key)

# ----------- TEST 1: Text Generation -----------
print("🔹 Testing Gemini text generation...\n")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in 2 lines"
)

print("✅ Response:")
print(response.text)


# ----------- TEST 2: Embeddings -----------
print("\n🔹 Testing Embeddings...\n")

emb = client.models.embed_content(
    model="gemini-embedding-001",
    contents=["Hello world"]
)

print("✅ Embedding length:", len(emb.embeddings[0].values))