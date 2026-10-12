import os
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    JSONLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ═══════════════════════════════════════════════════════════
#   ⚙️ Configuration
# ═══════════════════════════════════════════════════════════

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# Language-aware separators (Persian + English + punctuation)
SEPARATORS = [
    "\n\n",      # Paragraphs
    "\n",        # Lines
    "。",        # Japanese/Chinese period
    "؟",         # Persian question mark
    "?",         # English question mark
    "！",        # Japanese/Chinese exclamation
    "!",         # English exclamation
    "؛",         # Persian semicolon
    ";",         # English semicolon
    ".",         # English period
    "،",         # Persian comma
    ",",         # English comma
    " ",         # Spaces
    "",          # Fallback (character-level)
]

SUPPORTED_EXTENSIONS = {
    ".pdf":  "pdf",
    ".txt":  "text",
    ".md":   "markdown",
    ".docx": "docx",
    ".csv":  "csv",
    ".json": "json",
}


# ═══════════════════════════════════════════════════════════
#   🧹 Text Cleaning
# ═══════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Clean extracted text:
      - Normalize whitespace
      - Fix broken line breaks
      - Remove control characters
      - Normalize Persian/Arabic characters
    """
    if not text:
        return ""

    # Normalize Arabic/Persian variants
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = text.replace("ﻻ", "لا")

    # Remove zero-width characters
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)

    # Fix hyphenated line breaks (English: "exam-\nple" → "example")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse multiple newlines (>2 → 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ═══════════════════════════════════════════════════════════
#   📄 Document Loaders
# ═══════════════════════════════════════════════════════════

def _load_pdf(file_path: str) -> List[Document]:
    """Load a PDF file."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def _load_text(file_path: str) -> List[Document]:
    """Load a plain text file (UTF-8)."""
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def _load_markdown(file_path: str) -> List[Document]:
    """Load a Markdown file."""
    loader = UnstructuredMarkdownLoader(file_path, mode="single")
    return loader.load()


def _load_docx(file_path: str) -> List[Document]:
    """Load a Word document."""
    loader = Docx2txtLoader(file_path)
    return loader.load()


def _load_csv(file_path: str) -> List[Document]:
    """Load a CSV file."""
    loader = CSVLoader(file_path, encoding="utf-8")
    return loader.load()


def _load_json(file_path: str) -> List[Document]:
    """Load a JSON file as plain text (simple approach)."""
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


LOADER_MAP = {
    "pdf":      _load_pdf,
    "text":     _load_text,
    "markdown": _load_markdown,
    "docx":     _load_docx,
    "csv":      _load_csv,
    "json":     _load_json,
}


# ═══════════════════════════════════════════════════════════
#   📥 Load Document
# ═══════════════════════════════════════════════════════════

def load_document(file_path: str) -> List[Document]:
    """
    Load a document from any supported format.

    Supported: PDF, TXT, MD, DOCX, CSV, JSON

    Args:
        file_path: Path to the file

    Returns:
        List of Document objects with metadata

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is unsupported
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"❌ Unsupported format: {ext}\n"
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS.keys())}"
        )

    file_type = SUPPORTED_EXTENSIONS[ext]
    loader_fn = LOADER_MAP[file_type]

    # Load
    documents = loader_fn(file_path)

    # Enrich metadata + clean text
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    file_hash = _file_hash(file_path)

    for i, doc in enumerate(documents):
        doc.page_content = clean_text(doc.page_content)
        doc.metadata.update({
            "source": file_path,
            "file_name": file_name,
            "file_size": file_size,
            "file_hash": file_hash,
            "file_type": file_type,
            "doc_index": i,
        })

        # pypdf gives page 0-based → make it 1-based
        if "page" in doc.metadata and isinstance(doc.metadata["page"], int):
            doc.metadata["page"] = doc.metadata["page"] + 1

    # Remove empty docs
    documents = [d for d in documents if d.page_content.strip()]

    return documents


# ═══════════════════════════════════════════════════════════
#   ✂️ Split Documents
# ═══════════════════════════════════════════════════════════

def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into chunks with rich metadata.

    Args:
        documents: List of Document objects
        chunk_size: Max characters per chunk
        chunk_overlap: Overlap between chunks (for context continuity)

    Returns:
        List of chunked Documents with extra metadata
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,           # character-based (good for Persian)
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    # Enrich chunks with metadata
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_index": i,
            "chunk_total": total,
            "chunk_length": len(chunk.page_content),
            "chunk_id": _chunk_id(chunk, i),
        })

    # Remove tiny chunks (< 30 chars usually noise)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 30]

    return chunks


# ═══════════════════════════════════════════════════════════
#   📊 Statistics
# ═══════════════════════════════════════════════════════════

def get_stats(
    documents: List[Document],
    chunks: List[Document],
) -> Dict[str, Any]:
    """
    Compute statistics for loaded documents and chunks.
    """
    total_chars = sum(len(d.page_content) for d in documents)
    total_chunks = len(chunks)
    chunk_lengths = [len(c.page_content) for c in chunks] if chunks else [0]

    return {
        "num_documents": len(documents),
        "num_chunks": total_chunks,
        "total_characters": total_chars,
        "avg_chunk_size": round(sum(chunk_lengths) / len(chunk_lengths), 1),
        "min_chunk_size": min(chunk_lengths),
        "max_chunk_size": max(chunk_lengths),
        "total_pages": sum(
            1 for d in documents if "page" in d.metadata
        ),
    }


# ═══════════════════════════════════════════════════════════
#   🛠️ Helpers
# ═══════════════════════════════════════════════════════════

def _file_hash(file_path: str, block_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file (for caching)."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _chunk_id(chunk: Document, index: int) -> str:
    """Generate a short unique ID for a chunk."""
    raw = f"{chunk.metadata.get('source', '')}_{index}_{chunk.page_content[:50]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════
#   🧪 Self-Test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing ingest.py...")

    # Text cleaning test
    dirty = "این   یه   تست\n\n\n\nبا   فاصله   های   زیاد   است"
    clean = clean_text(dirty)
    assert "  " not in clean, "Whitespace not cleaned"
    print(f"✅ clean_text OK → {clean[:50]}")

    # Split test with a fake document
    from langchain_core.documents import Document as Doc
    fake_docs = [Doc(page_content="سلام. " * 200, metadata={"page": 1})]
    chunks = split_documents(fake_docs, chunk_size=200, chunk_overlap=20)
    print(f"✅ split_documents OK → {len(chunks)} chunks created")

    stats = get_stats(fake_docs, chunks)
    print(f"✅ get_stats OK → {stats}")

    print("Done.")