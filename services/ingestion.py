# # services/ingestion.py
# import asyncio
# import logging
# import os
# from dotenv import load_dotenv
# from utils.parser import parse_file
# from services.embedding import embed_texts  # Gemini 2048-d
# from pinecone import Pinecone

# load_dotenv()
# # Configure Pinecone
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# INDEX_NAME = "newrag"

# pc = Pinecone(api_key=PINECONE_API_KEY)
# index = pc.Index(INDEX_NAME)

# # ---------- Async embedding ----------
# async def embed_chunk(chunk: dict) -> dict:
#     """
#     Embeds a single chunk, returns enriched dict with embedding.
#     """
#     text = chunk.get("text", "")
#     if not text.strip():
#         logging.warning("Skipping empty chunk")
#         return None

#     try:
#         embedding = embed_texts(text)
#         chunk["embedding"] = embedding
#         return chunk
#     except Exception as e:
#         logging.error(f"Embedding failed for chunk: {e}")
#         return None

# # ---------- Ingest all chunks ----------
# async def ingest_text(file_path: str, file_type: str, use_semantic: bool = True):
#     """
#     Parses, chunks, embeds, and upserts to Pinecone.
#     """
#     # 1️⃣ Parse file
#     chunks = parse_file(file_path, file_type, use_semantic=use_semantic)
#     if not chunks:
#         logging.warning("No chunks extracted from file")
#         return

#     logging.info(f"Extracted {len(chunks)} chunks")

#     # 2️⃣ Async embed all chunks
#     tasks = [embed_chunk(c) for c in chunks]
#     results = await asyncio.gather(*tasks)

#     # 3️⃣ Filter out None embeddings
#     vectors = []
#     for i, chunk in enumerate(results):
#         if chunk and "embedding" in chunk:
#             vectors.append({
#                 "id": f"{file_path}-{i}",
#                 "values": chunk["embedding"],
#                 "metadata": {
#                     "page": chunk.get("page"),
#                     "section": chunk.get("section"),
#                     "cluster_id": chunk.get("cluster_id"),
#                     "source": os.path.basename(file_path),
#                     "chunk_text": chunk["text"]
#                 }
#             })

#     if not vectors:
#         logging.warning("No valid embeddings to upsert")
#         return

#     # 4️⃣ Upsert to Pinecone
#     try:
#         index.upsert(vectors=vectors)
#         logging.info(f"Successfully upserted {len(vectors)} vectors")
#     except Exception as e:
#         logging.error(f"Pinecone upsert failed: {e}")



import asyncio
import logging
import os
import uuid

from dotenv import load_dotenv
from pinecone import Pinecone

from utils.parser import parse_file
from services.embedding import embed_texts
from services.bm25 import update_corpus_stats

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "newrag"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


# ---------------------------------------------------
# 🔥 METADATA CLEANER (CRITICAL FIX)
# ---------------------------------------------------
def clean_metadata(metadata: dict) -> dict:
    """
    Ensures Pinecone-compatible metadata:
    - No None values
    - Only allowed types
    """
    cleaned = {}

    for k, v in metadata.items():
        if v is None:
            continue  # ❌ remove nulls

        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v

        elif isinstance(v, list):
            cleaned[k] = [str(x) for x in v]

        else:
            cleaned[k] = str(v)

    return cleaned


# Removed single-chunk embedding function to force batching


# ---------------------------------------------------
# Main Ingestion Pipeline
# ---------------------------------------------------
async def ingest_text(
    file_path: str,
    file_type: str,
    namespace: str,
    use_semantic: bool = True,
    original_filename: str = None,
    job_id: str = None,
    collection_id: str = "default",
) -> int:
    """
    Parse, chunk, embed, and upsert a document to Pinecone.

    Args:
        file_path:         Absolute path to the temp file on disk.
        file_type:         Extension string (pdf, docx, txt, eml).
        namespace:         Pinecone namespace (== authenticated user's workspace_id).
        use_semantic:      Route to semantic/hybrid chunking if True.
        original_filename: The real filename to use as document_id.
        job_id:            Background ingestion job ID.
        collection_id:     Sub-namespace grouping identifier.

    Returns:
        Number of vectors upserted (0 on failure).
    """
    from services.db import ingestion_jobs_collection
    import time
    
    if job_id:
        ingestion_jobs_collection.update_one(
            {"job_id": job_id},
            {"$set": {"status": "processing"}}
        )

    chunks = []
    try:
        chunks = parse_file(
            file_path,
            file_type,
            use_semantic=use_semantic
        )
    except Exception as e:
        logging.error(f"Parsing failed for {file_path}: {e}")
        if job_id:
            ingestion_jobs_collection.update_one(
                {"job_id": job_id},
                {"$set": {"status": "failed", "error": f"Parsing failed: {e}", "completed_at": time.time()}}
            )
        return 0

    if not chunks:
        logging.warning("No chunks found during parsing.")
        if job_id:
            ingestion_jobs_collection.update_one(
                {"job_id": job_id},
                {"$set": {"status": "failed", "error": "No chunks found", "completed_at": time.time()}}
            )
        return 0

    logging.info(f"Extracted {len(chunks)} chunks")

    if job_id:
        ingestion_jobs_collection.update_one(
            {"job_id": job_id},
            {"$set": {"chunks_total": len(chunks)}}
        )

    # -----------------------------------------
    # Embed All Chunks (Batched)
    # -----------------------------------------
    chunk_texts = [c.get("text", "") for c in chunks]
    try:
        embeddings = embed_texts(chunk_texts)
    except Exception as e:
        logging.error(f"Batch embedding failed: {e}")
        if job_id:
            ingestion_jobs_collection.update_one(
                {"job_id": job_id},
                {"$set": {"status": "failed", "error": f"Embedding failed: {e}", "completed_at": time.time()}}
            )
        return 0

    vectors = []
    # Use original_filename as document identity so UUID-temp-prefix doesn't leak
    source_name = original_filename or os.path.basename(file_path)

    # -----------------------------------------
    # Build Vectors
    # -----------------------------------------
    for idx, chunk in enumerate(chunks):
        embedding = embeddings[idx]

        # Ignore empty/zero embeddings
        if not embedding or sum(abs(x) for x in embedding) < 1e-5:
            continue

        # ✅ stable + traceable id
        chunk_id = f"{source_name}-{idx}-{uuid.uuid4().hex[:8]}"

        # ✅ SAFE linking (no None)
        prev_id = f"{source_name}-{idx-1}" if idx > 0 else "START"
        next_id = f"{source_name}-{idx+1}" if idx < len(chunks)-1 else "END"

        metadata = {
            "source": source_name,
            "document_id": source_name,
            "workspace_id": namespace,
            "collection_id": collection_id,

            # TEXT
            "chunk_text": chunk.get("text", ""),
            "parent_text": chunk.get("parent_text"),

            # LINKING
            "chunk_id": chunk_id,
            "prev_chunk_id": prev_id,
            "next_chunk_id": next_id,

            # STRUCTURE
            "page": chunk.get("page"),
            "section": chunk.get("section"),

            # STATS
            "chunk_length": len(chunk["text"]),
            "word_count": len(chunk["text"].split()),

            # FLAGS
            "compressed": False,
            "importance_score": chunk.get("importance_score", 0.0),
        }

        # 🔥 CLEAN metadata (CRITICAL)
        metadata = clean_metadata(metadata)

        vectors.append({
            "id": chunk_id,
            "values": embedding,
            "metadata": metadata
        })

    if not vectors:
        logging.warning("No valid vectors to upsert.")
        if job_id:
            ingestion_jobs_collection.update_one(
                {"job_id": job_id},
                {"$set": {"status": "failed", "error": "No valid vectors", "completed_at": time.time()}}
            )
        return 0

    # -----------------------------------------
    # Upsert to Pinecone
    # -----------------------------------------
    try:
        index.upsert(
            vectors=vectors,
            namespace=namespace
        )

        # ✅ Track document record in MongoDB for UI listings and intelligence jobs
        from services.db import documents_collection
        from datetime import datetime, timezone
        documents_collection.update_one(
            {"document_id": source_name, "namespace": namespace},
            {
                "$set": {
                    "document_id": source_name,
                    "filename": source_name,
                    "namespace": namespace,
                    "workspace_id": namespace,
                    "collection_id": collection_id,
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                    "chunk_count": len(vectors),
                }
            },
            upsert=True
        )

        if job_id:
            ingestion_jobs_collection.update_one(
                {"job_id": job_id},
                {
                    "$set": {
                        "status": "completed",
                        "chunks_processed": len(vectors),
                        "completed_at": time.time()
                    }
                }
            )

        logging.info(
            f"✅ Upserted {len(vectors)} vectors to namespace={namespace} and registered in MongoDB."
        )

        # Update BM25 corpus stats so future queries have accurate IDF
        update_corpus_stats(chunks, namespace)

        return len(vectors)

    except Exception as e:
        logging.error(
            f"❌ Pinecone upsert failed: {e}",
            exc_info=True
        )
        if job_id:
            ingestion_jobs_collection.update_one(
                {"job_id": job_id},
                {"$set": {"status": "failed", "error": str(e), "completed_at": time.time()}}
            )
        return 0