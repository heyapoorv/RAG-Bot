import re
import email
from email import policy
from typing import List, Tuple
import fitz
from docx import Document

from utils.chunking import (
    chunk_text,
    clause_aware_chunk,
    parent_child_chunk
)

# ---------------------------------------------------
# DOCUMENT CLASSIFIER
# ---------------------------------------------------
def classify_document(text: str) -> str:
    text_lower = text.lower()
    legal_keywords = {"contract", "liability", "termination", "policy", "agreement", "clause", "indemnity", "warranties", "deductible"}
    research_keywords = {"abstract", "methodology", "conclusion", "references", "et al", "figure", "literature", "hypothesis"}
    
    legal_score = sum(1 for w in legal_keywords if w in text_lower)
    research_score = sum(1 for w in research_keywords if w in text_lower)
    
    if legal_score > 1 and legal_score >= research_score:
        return "legal"
    elif research_score > 1 and research_score > legal_score:
        return "research"
    else:
        return "general"

# ---------------------------------------------------
# PDF PARSER
# ---------------------------------------------------
def parse_pdf(file_path: str) -> List[Tuple[str, int, str]]:
    """
    Extract full PDF text using PyMuPDF (fitz).
    """
    pages = []
    with fitz.open(file_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages.append((text, page_num + 1, f"Page {page_num + 1}"))
    return pages


# ---------------------------------------------------
# DOCX PARSER
# ---------------------------------------------------
def parse_docx(file_path: str) -> List[Tuple[str, int, str]]:
    doc = Document(file_path)

    full_text = "\n".join(
        para.text.strip()
        for para in doc.paragraphs
        if para.text.strip()
    )

    return [
        (full_text, 1, "Full Document")
    ]


# ---------------------------------------------------
# TXT PARSER
# ---------------------------------------------------
def parse_text(file_path: str) -> List[Tuple[str, int, str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    pages = []
    # Split text by "Page \d+:"
    parts = re.split(r'(?i)page\s+(\d+):', text)
    if len(parts) > 1:
        if parts[0].strip():
            pages.append((parts[0].strip(), 1, "Preamble"))
        for i in range(1, len(parts), 2):
            page_num = int(parts[i])
            page_text = parts[i+1].strip()
            if page_text:
                pages.append((page_text, page_num, f"Page {page_num}"))
    else:
        pages.append((text, 1, "Full Document"))
        
    return pages


# ---------------------------------------------------
# EML PARSER
# ---------------------------------------------------
def parse_eml(file_path: str) -> List[Tuple[str, int, str]]:
    with open(file_path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    subject = msg.get("subject", "No Subject")
    sender = msg.get("from", "Unknown Sender")
    recipient = msg.get("to", "Unknown Recipient")
    date = msg.get("date", "Unknown Date")

    body = ""
    # Get plain text body if possible
    body_part = msg.get_body(preferencelist=('plain', 'html'))
    if body_part:
        body = body_part.get_content()
    else:
        # Fallback if no body part
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                body = part.get_content()
                break

    full_text = f"Subject: {subject}\nFrom: {sender}\nTo: {recipient}\nDate: {date}\n\n{body}".strip()

    return [
        (full_text, 1, "Full Email")
    ]


# ---------------------------------------------------
# UNIFIED PARSER
# ---------------------------------------------------
def parse_file(
    file_path: str,
    file_type: str,
    use_semantic: bool = True
) -> List[dict]:

    if file_type.lower() == "pdf":
        pages = parse_pdf(file_path)
    elif file_type.lower() == "docx":
        pages = parse_docx(file_path)
    elif file_type.lower() == "txt":
        pages = parse_text(file_path)
    elif file_type.lower() == "eml":
        pages = parse_eml(file_path)
    else:
        raise ValueError("Unsupported file type")

    chunks = []
    
    # Classify based on first page to determine global routing
    doc_type = "general"
    if pages:
        doc_type = classify_document(pages[0][0])

    for text, page, section in pages:
        
        # -----------------------------------------
        # Adaptive Chunk Routing
        # -----------------------------------------
        if doc_type == "legal":
            raw_chunks = clause_aware_chunk(text, max_tokens=250, overlap_sentences=2)
            for i, chunk in enumerate(raw_chunks):
                chunk_id = f"legal_{page}_chunk_{i}"
                chunks.append({
                    "text": chunk,
                    "parent_text": chunk,
                    "parent_id": f"legal_parent_{page}_{i}",
                    "chunk_id": chunk_id,
                    "prev_chunk_id": f"legal_{page}_chunk_{i-1}" if i > 0 else None,
                    "next_chunk_id": f"legal_{page}_chunk_{i+1}" if i < len(raw_chunks)-1 else None,
                    "page": page,
                    "section": section
                })
                
        elif doc_type == "research":
            structured_chunks = parent_child_chunk(text)
            for i, item in enumerate(structured_chunks):
                chunk_id = f"{item['parent_id']}_child_{i}"
                chunks.append({
                    "text": item["child_text"],
                    "parent_text": item["parent_text"],
                    "parent_id": item["parent_id"],
                    "chunk_id": chunk_id,
                    "prev_chunk_id": f"{item['parent_id']}_child_{i-1}" if i > 0 else None,
                    "next_chunk_id": f"{item['parent_id']}_child_{i+1}" if i < len(structured_chunks)-1 else None,
                    "page": page,
                    "section": section
                })
                
        else:
            # general -> hybrid (using basic chunking with parent-child overlay or just basic chunking)
            # We'll use parent_child_chunk but with smaller parent_size to represent hybrid
            structured_chunks = parent_child_chunk(text, parent_size=350, child_size=150)
            if not structured_chunks:
                basic_chunks = chunk_text(text)
                for i, chunk in enumerate(basic_chunks):
                    chunk_id = f"basic_{page}_chunk_{i}"
                    chunks.append({
                        "text": chunk,
                        "parent_text": chunk,
                        "parent_id": f"basic_parent_{page}_{i}",
                        "chunk_id": chunk_id,
                        "prev_chunk_id": f"basic_{page}_chunk_{i-1}" if i > 0 else None,
                        "next_chunk_id": f"basic_{page}_chunk_{i+1}" if i < len(basic_chunks)-1 else None,
                        "page": page,
                        "section": section
                    })
            else:
                for i, item in enumerate(structured_chunks):
                    chunk_id = f"hybrid_{item['parent_id']}_child_{i}"
                    chunks.append({
                        "text": item["child_text"],
                        "parent_text": item["parent_text"],
                        "parent_id": item["parent_id"],
                        "chunk_id": chunk_id,
                        "prev_chunk_id": f"hybrid_{item['parent_id']}_child_{i-1}" if i > 0 else None,
                        "next_chunk_id": f"hybrid_{item['parent_id']}_child_{i+1}" if i < len(structured_chunks)-1 else None,
                        "page": page,
                        "section": section
                    })

    return chunks