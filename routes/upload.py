"""
Upload routes.
Handles secure document uploading, validation, parsing, chunking, embedding,
and ingestion into Pinecone namespace, scoped per tenant/user.
"""
from __future__ import annotations

import os
import shutil
import time
import uuid
from fastapi import APIRouter, UploadFile, HTTPException, Form, Depends, status, BackgroundTasks

from services.auth_service import get_current_user, validate_namespace_access
from services.ingestion import ingest_text
from models.schemas import UploadResponse, IngestionJobStatus
from config import settings
from utils.logger import get_logger
from services.metrics import metrics
from services.db import ingestion_jobs_collection

router = APIRouter()
logger = get_logger(__name__)


async def _background_ingest_wrapper(
    file_path: str,
    ext: str,
    session_id: str,
    safe_name: str,
    job_id: str,
    collection_id: str
):
    """
    Wrapper for ingest_text to ensure the temporary file is deleted
    after the background job finishes (success or fail).
    """
    try:
        chunks_ingested = await ingest_text(
            file_path=file_path,
            file_type=ext,
            namespace=session_id,
            use_semantic=True,
            original_filename=safe_name,
            job_id=job_id,
            collection_id=collection_id,
        )
        if metrics:
            metrics.ingestion_total.labels(file_type=ext).inc()
            if chunks_ingested:
                metrics.ingestion_chunks_total.inc(chunks_ingested)
    except Exception as e:
        logger.error(f"Background ingestion wrapper failed: {e}", exc_info=True)
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as exc:
            logger.warning(
                "Failed to remove temporary file after ingestion",
                extra={"path": file_path, "error": str(exc)},
            )


@router.post("/", response_model=UploadResponse)
async def upload_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    session_id: str = Form(..., description="The user's own namespace (must equal your username)"),
    collection_id: str = Form("default", description="The collection this document belongs to"),
    user: dict = Depends(get_current_user),
):
    """
    Upload and queue a document for ingestion.
    Returns a job_id immediately while ingestion runs in the background.
    """
    start_time = time.time()

    # ── 1. Namespace ownership check ───────────────────────────────────────────
    validate_namespace_access(session_id, user, require_write=True)

    logger.info(
        "Received file upload request",
        extra={
            "filename": file.filename,
            "session_id": session_id,
            "collection_id": collection_id,
            "username": user["username"],
        },
    )

    # ── 2. Validate File Extension ─────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing.",
        )

    # Strip any directory components from the original filename (path traversal prevention)
    safe_name = os.path.basename(file.filename)
    if not safe_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    ext = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    allowed_types = {"pdf", "docx", "txt", "eml"}

    if ext not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: '{ext}'. Allowed: {', '.join(sorted(allowed_types))}",
        )

    # ── 3. Save to UUID-prefixed temp path (prevents collisions and traversal) ─
    temp_dir = os.path.abspath("temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # UUID prefix: no two concurrent uploads of the same filename will collide
    temp_filename = f"{uuid.uuid4().hex}_{safe_name}"
    temp_path = os.path.join(temp_dir, temp_filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        logger.error("Failed to write temporary upload file", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to receive uploaded file.",
        )

    # ── 4. Create Job & Queue Background Task ──────────────────────────────────
    job_id = str(uuid.uuid4())
    
    job_record = {
        "job_id": job_id,
        "status": "pending",
        "filename": safe_name,
        "namespace": session_id,
        "collection_id": collection_id,
        "chunks_total": 0,
        "chunks_processed": 0,
        "error": None,
        "created_at": time.time(),
        "completed_at": None,
    }
    
    try:
        ingestion_jobs_collection.insert_one(job_record)
    except Exception as e:
        logger.error(f"Failed to create ingestion job record: {e}")
        # Clean up temp file immediately if we fail to queue
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize ingestion job.",
        )

    background_tasks.add_task(
        _background_ingest_wrapper,
        temp_path,
        ext,
        session_id,
        safe_name,
        job_id,
        collection_id
    )

    return UploadResponse(
        status="success",
        file=safe_name,
        namespace=session_id,
        job_id=job_id,
        message="Document uploaded and queued for ingestion.",
    )


@router.get("/status/{job_id}", response_model=IngestionJobStatus)
async def get_upload_status(
    job_id: str,
    user: dict = Depends(get_current_user),
):
    """
    Get the status of a background ingestion job.
    Users can only query jobs that belong to their namespace.
    """
    job = ingestion_jobs_collection.find_one({"job_id": job_id})
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found.",
        )
        
    # Security: Ensure user owns the namespace of this job
    validate_namespace_access(job["namespace"], user, require_write=False)

    return IngestionJobStatus(
        job_id=job["job_id"],
        status=job["status"],
        filename=job["filename"],
        namespace=job["namespace"],
        collection_id=job.get("collection_id", "default"),
        chunks_total=job.get("chunks_total", 0),
        chunks_processed=job.get("chunks_processed", 0),
        error=job.get("error"),
        created_at=job.get("created_at", 0.0),
        completed_at=job.get("completed_at"),
    )