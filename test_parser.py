import asyncio
from utils.parser import parse_file
from utils.chunking import clause_aware_chunk

chunks = parse_file("evaluation/sample_docs/policy.txt", "txt")
print(f"Number of chunks: {len(chunks)}")
if chunks:
    print(chunks[0])
