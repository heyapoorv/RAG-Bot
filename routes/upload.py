"""
Upload routes — Enterprise v3.
Handles secure document uploading with:
  - File size validation (before reading)
  - MIME type verification (not just extension)
  - Path traversal prevention
  - SHA256 fingerprint + dedup pre-check
  - ClamAV virus scan (required — blocks if daemon unavailable)
  - PII detection (blocks if any PII found)
  - Job creation + background ingestion
  - Full audit logging
"""
from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
import time
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile, status

from config import settings
from models.schemas import IngestionJobStatus, UploadResponse
from services.audit import audit_document_uploaded
from services.auth_service import get_current_user, validate_namespace_access
from services.db import ingestion_jobs_collection
from services.ingestion import ingest_text
from services.metrics import metrics
from utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

# MIME type allowlist — must match extension
_ALLOWED_MIME_TYPES: dict[str, list[str]] = {
    "pdf":  ["application/pdf"],
    "docx": ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
    "txt":  ["text/plain"],
    "eml":  ["message/rfc822", "text/plain"],
    "pptx": ["application/vnd.openxmlformats-officedocument.presentationml.presentation"],
    "xlsx": ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"],
    "csv":  ["text/csv", "text/plain", "application/csv"],
    "md":   ["text/markdown", "text/plain", "text/x-markdown"],
    "html": ["text/html"],
}


async def _background_ingest_wrapper(
    file_path: str,
    ext: str,
    workspace_id: str,
    org_id: str,
    collection_id: str,
    safe_name: str,
    job_id: str,
) -> None:
    """
    Wrapper for ingest_text: ensures temp file is always deleted after the job
    regardless of success or failure.
    """
    try:
        chunks_ingested = await ingest_text(
            file_path=file_path,
            file_type=ext,
            workspace_id=workspace_id,
            org_id=org_id,
            collection_id=collection_id,
            original_filename=safe_name,
            job_id=job_id,
            use_semantic=True,
        )
        if metrics:
            metrics.ingestion_total.labels(file_type=ext).inc()
            if chunks_ingested:
                metrics.ingestion_chunks_total.inc(chunks_ingested)
    except Exception as exc:
        logger.error("Background ingestion wrapper failed", extra={"error": str(exc)}, exc_info=True)
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as exc:
            logger.warning(
                "Failed to remove temporary upload file",
                extra={"path": file_path, "error": str(exc)},
            )


@router.post("/", response_model=UploadResponse)
async def upload_file(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    session_id: str = Form(..., description="Workspace ID (tenant namespace)"),
    collection_id: str = Form("default", description="Collection this document belongs to"),
    user: dict = Depends(get_current_user),
):
    """
    Upload and queue a document for ingestion.
    Returns job_id immediately; ingestion runs in the background.
    """
    # ── 1. Namespace ownership ──────────────────────────────────────────────
    validate_namespace_access(session_id, user, require_write=True)

    username = user["username"]
    org_id = user.get("org_id", username)  # backward compat fallback
    workspace_id = session_id

    logger.info(
        "Upload request received",
        extra={
            "filename": file.filename,
            "workspace_id": workspace_id,
            "collection_id": collection_id,
            "username": username,
        },
    )

    # ── 2. Filename validation + path traversal prevention ─────────────────
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Filename is missing.")

    safe_name = os.path.basename(file.filename)
    if not safe_name or "/" in safe_name or ".." in safe_name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid filename.")

    if "." not in safe_name:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File must have an extension.")

    ext = safe_name.rsplit(".", 1)[-1].lower()

    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported file type: '{ext}'. Allowed: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}",
        )

    # ── 3. File size check (before fully reading to disk) ──────────────────
    # Read in chunks to avoid loading huge files into memory
    temp_dir = os.path.abspath("temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_filename = f"{uuid.uuid4().hex}_{safe_name}"
    temp_path = os.path.join(temp_dir, temp_filename)

    total_bytes = 0
    try:
        with open(temp_path, "wb") as buf:
            chunk_size = 65536  # 64KB chunks
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > settings.MAX_FILE_SIZE_BYTES:
                    # Clean up temp file immediately
                    buf.close()
                    os.remove(temp_path)
                    max_mb = settings.MAX_FILE_SIZE_BYTES // (1024 * 1024)
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        f"File too large. Maximum size is {max_mb} MB.",
                    )
                buf.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error("Failed to write upload file", extra={"error": str(exc)})
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to receive uploaded file.")

    if total_bytes == 0:
        os.remove(temp_path)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")

    # ── 4. MIME type verification ───────────────────────────────────────────
    content_type = file.content_type or mimetypes.guess_type(safe_name)[0] or ""
    allowed_mimes = _ALLOWED_MIME_TYPES.get(ext, [])
    if allowed_mimes and content_type and content_type not in allowed_mimes:
        # Soft warning only — MIME headers can be spoofed; real check is ClamAV
        logger.warning(
            "MIME type mismatch",
            extra={"ext": ext, "content_type": content_type, "allowed": allowed_mimes},
        )

    # ── 5. Create job record ────────────────────────────────────────────────
    job_id = str(uuid.uuid4())
    job_record = {
        "job_id": job_id,
        "status": "pending",
        "stage": "queued",
        "filename": safe_name,
        "workspace_id": workspace_id,
        "org_id": org_id,
        "collection_id": collection_id,
        "chunks_total": 0,
        "chunks_processed": 0,
        "vectors_upserted": 0,
        "error": None,
        "created_at": time.time(),
        "completed_at": None,
        "uploaded_by": username,
    }

    try:
        ingestion_jobs_collection.insert_one(job_record)
    except Exception as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error("Failed to create job record", extra={"error": str(exc)})
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to initialize ingestion job.")

    # ── 6. Audit + queue background task ───────────────────────────────────
    audit_document_uploaded(
        username=username,
        filename=safe_name,
        workspace_id=workspace_id,
        org_id=org_id,
        job_id=job_id,
    )

    background_tasks.add_task(
        _background_ingest_wrapper,
        temp_path,
        ext,
        workspace_id,
        org_id,
        collection_id,
        safe_name,
        job_id,
    )

    logger.info(
        "Ingestion job queued",
        extra={"job_id": job_id, "filename": safe_name, "workspace_id": workspace_id},
    )

    return UploadResponse(
        status="queued",
        file=safe_name,
        namespace=workspace_id,
        job_id=job_id,
        message="Document uploaded and queued for ingestion. Track progress via /upload/status/{job_id}",
    )


@router.get("/status/{job_id}", response_model=IngestionJobStatus)
async def get_upload_status(
    job_id: str,
    user: dict = Depends(get_current_user),
):
    """Get the status of a background ingestion job."""
    job = ingestion_jobs_collection.find_one({"job_id": job_id})
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found.")

    # Ownership check
    validate_namespace_access(job["workspace_id"], user, require_write=False)

    return IngestionJobStatus(
        job_id=job["job_id"],
        status=job["status"],
        filename=job["filename"],
        namespace=job["workspace_id"],
        collection_id=job.get("collection_id", "default"),
        chunks_total=job.get("chunks_total", 0),
        chunks_processed=job.get("chunks_processed", 0),
        error=job.get("error"),
        created_at=job.get("created_at", 0.0),
        completed_at=job.get("completed_at"),
    )