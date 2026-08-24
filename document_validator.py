"""
document_validator.py
=====================
Pre-indexing validation for uploaded PDF documents.

Two validation stages:
  1. validate_file   — format, size, readability, title/hash deduplication
  2. validate_topic  — 3-way LLM classification (ML | math | neither)
                       with smart redirect logic for ML<->Math mismatches

Topic decision table:
  Detected = ML,   Selected = ML   -> Accept, save to Data/ML/
  Detected = math, Selected = math -> Accept, save to Data/math/
  Detected = ML,   Selected = math -> Redirect to Data/ML/  + notify user
  Detected = math, Selected = ML   -> Redirect to Data/math/ + notify user
  Detected = neither               -> Reject with explanation
"""

import io
import os
import json
import glob
import hashlib

from pypdf import PdfReader
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
CONFIDENCE_THRESHOLD = 0.55          # min score to be classified as ML or math

CACHE_DIR = "faiss_index_cache"
HASH_FILE = os.path.join(CACHE_DIR, "file_hashes.json")
DATA_DIRS = {
    "ML":   os.path.join("Data", "ML"),
    "math": os.path.join("Data", "math"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_hash(file_bytes: bytes) -> str:
    """Return the SHA-256 hex digest of raw file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def _load_cached_hashes() -> dict:
    """Load the persisted hash manifest (path -> sha256)."""
    if not os.path.exists(HASH_FILE):
        return {}
    with open(HASH_FILE, "r") as fh:
        return json.load(fh)


def _scan_existing_filenames() -> set:
    """Return the set of filenames (lowercased) already in Data/ML/ and Data/math/."""
    names = set()
    for folder in DATA_DIRS.values():
        if os.path.isdir(folder):
            for p in glob.glob(os.path.join(folder, "*.pdf")):
                names.add(os.path.basename(p).lower())
    return names


def _extract_pdf_text_head(file_bytes: bytes, max_chars: int = 1800):
    """
    Extract the PDF title (from metadata or first heading) and the
    first `max_chars` characters of body text for topic sniffing.

    Returns:
        (title, excerpt)
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    meta = reader.metadata or {}
    title = str(meta.get("/Title", "")).strip()

    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
        if sum(len(t) for t in text_parts) >= max_chars:
            break

    full_text = "\n".join(text_parts)[:max_chars]

    # Fallback: use first non-empty line as title
    if not title:
        for line in full_text.splitlines():
            line = line.strip()
            if len(line) > 5:
                title = line
                break

    return title, full_text


# ── Stage 1: File Validation ──────────────────────────────────────────────────

def validate_file(file_bytes: bytes, filename: str):
    """
    Validate the uploaded file before topic checking.

    Checks (in order):
      1. Non-empty file
      2. Valid PDF magic bytes (%PDF-)
      3. File size within limit
      4. PDF is readable (pypdf can open it)
      5. Has extractable text (not a scanned image-only PDF)
      6. Filename not already in Data/ (name dedup)
      7. Content hash not already in index (content dedup)

    Returns:
        (True, "")              — all checks passed
        (False, error_message)  — first failing check with explanation
    """
    # 1. Non-empty
    if not file_bytes:
        return False, "The uploaded file is empty."

    # 2. PDF magic bytes
    if not file_bytes[:5] == b"%PDF-":
        return False, "The uploaded file does not appear to be a valid PDF (missing PDF header)."

    # 3. Size limit
    size_mb = len(file_bytes) / (1024 * 1024)
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return False, f"File size ({size_mb:.1f} MB) exceeds the {MAX_FILE_SIZE_MB} MB limit."

    # 4. Readability
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            return False, "The PDF is password-protected and cannot be read."
        page_count = len(reader.pages)
        if page_count == 0:
            return False, "The PDF has no pages."
    except Exception as exc:
        return False, f"The PDF appears to be corrupted or unreadable: {exc}"

    # 5. Extractable text
    text_sample = ""
    for page in reader.pages[:3]:
        text_sample += (page.extract_text() or "")
        if len(text_sample.strip()) > 50:
            break
    if len(text_sample.strip()) < 50:
        return False, (
            "Could not extract readable text from the PDF. "
            "It may be a scanned image-only document with no OCR layer."
        )

    # 6. Filename deduplication
    existing_names = _scan_existing_filenames()
    if filename.lower() in existing_names:
        return False, (
            f"A file named **{filename}** already exists in the knowledge base. "
            "Please rename the file or check if it was already uploaded."
        )

    # 7. Content-hash deduplication
    new_hash = _compute_hash(file_bytes)
    cached = _load_cached_hashes()
    for cached_path, cached_hash in cached.items():
        if cached_hash == new_hash:
            cached_name = os.path.basename(cached_path)
            return False, (
                f"This document's content is identical to **{cached_name}**, "
                "which is already in the knowledge base. Skipping to avoid duplication."
            )

    return True, ""


# ── Stage 2: Topic Validation (3-way LLM Classification) ────────────────────

def validate_topic(
    file_bytes: bytes,
    filename: str,
    selected_category: str,
    groq_api_key: str,
) -> dict:
    """
    Use the Groq LLM to classify the document as 'ML', 'math', or 'neither'.

    Args:
        file_bytes:        Raw PDF bytes.
        filename:          Original filename (for context).
        selected_category: The category the user chose ("ML" or "math").
        groq_api_key:      Groq API key string.

    Returns a dict:
        {
            "accepted":           bool,
            "detected_category":  str,   "ML" | "math" | "neither"
            "redirected":         bool,  True if detected != selected
            "actual_category":    str,   where the file WILL be saved
            "confidence":         float,
            "reason":             str,
        }
    """
    title, excerpt = _extract_pdf_text_head(file_bytes, max_chars=1800)

    prompt = (
        "You are an academic document classifier. Your job is to classify a research paper "
        "or textbook excerpt into exactly one of three categories:\n"
        '- "ML"      : Machine Learning, Deep Learning, NLP, Reinforcement Learning, LLMs, '
        "fine-tuning, transformers, neural networks, etc.\n"
        '- "math"    : Pure or Applied Mathematics, Linear Algebra, Calculus, Statistics, '
        "Probability Theory, Optimization, Topology, etc.\n"
        '- "neither" : Anything that does not clearly belong to either of the above two domains.\n\n'
        f"Document title : {title or '(not found)'}\n"
        f"Filename       : {filename}\n"
        f'Excerpt (first ~1800 chars):\n"""\n{excerpt}\n"""\n\n'
        "Respond with ONLY a JSON object with these exact keys:\n"
        '{\n'
        '  "detected": "<ML | math | neither>",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "reason": "<one concise sentence explaining your decision>"\n'
        "}\n\n"
        "No additional text, no markdown, only raw JSON."
    )

    llm = ChatGroq(
        model_name="qwen/qwen3.6-27b",
        temperature=0.0,
        api_key=groq_api_key,
    )

    try:
        raw = llm.invoke(prompt).content.strip()
        # Strip potential markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        detected   = str(parsed.get("detected", "neither")).strip()
        confidence = float(parsed.get("confidence", 0.0))
        reason     = str(parsed.get("reason", "No reason provided."))
    except Exception as exc:
        return {
            "accepted": False,
            "detected_category": "neither",
            "redirected": False,
            "actual_category": selected_category,
            "confidence": 0.0,
            "reason": f"Classification service encountered an error: {exc}",
        }

    # Normalise detected label (handle case variations)
    detected_lower = detected.lower()
    if detected_lower == "ml":
        detected = "ML"
    elif detected_lower == "math":
        detected = "math"
    else:
        detected = "neither"

    # Below confidence threshold -> treat as 'neither'
    if detected != "neither" and confidence < CONFIDENCE_THRESHOLD:
        return {
            "accepted": False,
            "detected_category": "neither",
            "redirected": False,
            "actual_category": selected_category,
            "confidence": confidence,
            "reason": (
                f"The document could not be confidently classified into ML or Math "
                f"(confidence {confidence:.0%} < required {CONFIDENCE_THRESHOLD:.0%}). "
                f"LLM note: {reason}"
            ),
        }

    # ── Decision table ────────────────────────────────────────────────────────
    if detected == "neither":
        return {
            "accepted": False,
            "detected_category": "neither",
            "redirected": False,
            "actual_category": selected_category,
            "confidence": confidence,
            "reason": reason,
        }

    # Detected is "ML" or "math" with sufficient confidence
    redirected    = (detected != selected_category)
    actual_folder = detected   # where it will actually be saved

    return {
        "accepted":          True,
        "detected_category": detected,
        "redirected":        redirected,
        "actual_category":   actual_folder,
        "confidence":        confidence,
        "reason":            reason,
    }
