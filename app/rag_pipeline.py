import sys
import os
import time
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple, Callable, Iterator

# ═══════════════════════════════════════════════════════════
#   🔧 Add project root to sys.path
# ═══════════════════════════════════════════════════════════
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.ingest import load_document, split_documents, get_stats


# ═══════════════════════════════════════════════════════════
#   ⚙️ Configuration
# ═══════════════════════════════════════════════════════════

EMBEDDING_MODELS = {
    "intfloat/multilingual-e5-large": {
        "desc": "Best overall quality",
        "dim": 1024,
        "size_mb": 2200,
        "speed": "slow",
    },
    "intfloat/multilingual-e5-base": {
        "desc": "Balanced quality & speed",
        "dim": 768,
        "size_mb": 1100,
        "speed": "medium",
    },
    "BAAI/bge-m3": {
        "desc": "Excellent multilingual, dense+sparse",
        "dim": 1024,
        "size_mb": 2300,
        "speed": "slow",
    },
    "sentence-transformers/LaBSE": {
        "desc": "Best for 100+ languages",
        "dim": 768,
        "size_mb": 1800,
        "speed": "medium",
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "desc": "Fastest, good for prototyping",
        "dim": 384,
        "size_mb": 470,
        "speed": "fast",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "desc": "English-only, very fast",
        "dim": 384,
        "size_mb": 90,
        "speed": "very fast",
    },
}

DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_COLLECTION = "knowledge_base"


# ═══════════════════════════════════════════════════════════
#   🧠 Device Detection
# ═══════════════════════════════════════════════════════════

def _detect_device() -> str:
    """
    Auto-detect the best available device.
    Returns 'cuda' for NVIDIA, 'mps' for Apple Silicon, else 'cpu'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ═══════════════════════════════════════════════════════════
#   💾 Embedding Model Cache (Singleton)
# ═══════════════════════════════════════════════════════════

_embedding_cache: Dict[str, HuggingFaceEmbeddings] = {}


def get_embedding_model(
    model_name: str = DEFAULT_EMBED_MODEL,
    device: Optional[str] = None,
    normalize: bool = True,
) -> HuggingFaceEmbeddings:
    """
    Load (or retrieve from cache) a HuggingFace embedding model.

    Args:
        model_name: HuggingFace model ID
        device: 'cpu', 'cuda', 'mps' (auto-detected if None)
        normalize: L2-normalize embeddings (recommended)

    Returns:
        Cached HuggingFaceEmbeddings instance
    """
    if device is None:
        device = _detect_device()

    cache_key = f"{model_name}_{device}_{normalize}"
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    print(f"🔤 Loading embedding model: {model_name} on {device}...")
    start = time.time()

    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize},
    )

    elapsed = time.time() - start
    print(f"✅ Embedding model loaded in {elapsed:.2f}s")

    _embedding_cache[cache_key] = model
    return model


def clear_embedding_cache() -> None:
    """Free all cached embedding models from memory."""
    global _embedding_cache
    _embedding_cache.clear()
    print("🧹 Embedding cache cleared")


# ═══════════════════════════════════════════════════════════
#   🏗️ Build Vectorstore
# ═══════════════════════════════════════════════════════════

def build_vectorstore(
    file_path: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    device: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Chroma, Dict[str, Any]]:
    """
    Build a Chroma vector store from a document.

    Args:
        file_path: Path to source document
        persist_dir: Chroma persistence directory
        collection_name: Collection name
        embedding_model: HuggingFace model ID
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks
        device: 'cpu', 'cuda', or None (auto)
        progress_callback: Optional fn(percent, message)
        metadata: Optional extra metadata to attach

    Returns:
        (vectorstore, stats_dict)
    """
    def report(pct: int, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)
        print(f"[{pct:3d}%] {msg}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    start_total = time.time()

    # ---------- 1. Load document ----------
    report(5, f"Loading document: {os.path.basename(file_path)}")
    t0 = time.time()
    documents = load_document(file_path)
    load_time = time.time() - t0

    if not documents:
        raise ValueError(f"❌ No content extracted from {file_path}")

    report(20, f"Loaded {len(documents)} pages in {load_time:.2f}s")

    # ---------- 2. Split into chunks ----------
    report(25, f"Splitting (size={chunk_size}, overlap={chunk_overlap})...")
    t0 = time.time()
    chunks = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_time = time.time() - t0

    if not chunks:
        raise ValueError("❌ Splitting produced no chunks")

    report(40, f"Created {len(chunks)} chunks in {split_time:.2f}s")

    # ---------- 3. Load embedding model ----------
    report(45, f"Loading embedding model: {embedding_model.split('/')[-1]}")
    t0 = time.time()
    embeddings = get_embedding_model(embedding_model, device=device)
    embed_time = time.time() - t0
    report(55, f"Embedding model ready in {embed_time:.2f}s")

    # ---------- 4. Store in Chroma ----------
    report(60, f"Storing in Chroma at {persist_dir}...")
    t0 = time.time()
    os.makedirs(persist_dir, exist_ok=True)

    # Clean up existing collection to avoid mixing
    _delete_collection(persist_dir, collection_name)

    # Attach run metadata to each chunk
    run_meta = {
        "ingested_at": datetime.now().isoformat(),
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "source_file": os.path.basename(file_path),
    }
    if metadata:
        run_meta.update(metadata)

    for c in chunks:
        c.metadata.update(run_meta)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    # Persist explicitly (older versions require it)
    try:
        vectorstore.persist()
    except Exception:
        pass

    store_time = time.time() - t0
    report(90, f"Stored in {store_time:.2f}s")

    total_time = time.time() - start_total

    # ---------- 5. Collect stats ----------
    stats = get_stats(documents, chunks)
    stats.update({
        "load_time": round(load_time, 2),
        "split_time": round(split_time, 2),
        "embed_time": round(embed_time, 2),
        "store_time": round(store_time, 2),
        "total_time": round(total_time, 2),
        "embedding_model": embedding_model,
        "embedding_dim": EMBEDDING_MODELS.get(embedding_model, {}).get("dim", "?"),
        "device": device or _detect_device(),
        "collection": collection_name,
        "persist_dir": persist_dir,
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    })

    report(100, f"✅ Done in {total_time:.2f}s")

    return vectorstore, stats


# ═══════════════════════════════════════════════════════════
#   ➕ Add Documents to Existing Collection (Incremental)
# ═══════════════════════════════════════════════════════════

def add_to_vectorstore(
    vectorstore: Chroma,
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Dict[str, Any]:
    """
    Add a document to an existing vectorstore (incremental ingestion).

    Args:
        vectorstore: Existing Chroma instance
        file_path: Document to add
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap

    Returns:
        Stats dict for the newly added document
    """
    t0 = time.time()

    documents = load_document(file_path)
    if not documents:
        raise ValueError(f"No content extracted from {file_path}")

    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("No chunks produced")

    run_meta = {
        "ingested_at": datetime.now().isoformat(),
        "source_file": os.path.basename(file_path),
    }
    for c in chunks:
        c.metadata.update(run_meta)

    vectorstore.add_documents(chunks)

    try:
        vectorstore.persist()
    except Exception:
        pass

    elapsed = time.time() - t0

    stats = get_stats(documents, chunks)
    stats.update({
        "total_time": round(elapsed, 2),
        "file_name": os.path.basename(file_path),
        "mode": "incremental",
    })
    return stats


# ═══════════════════════════════════════════════════════════
#   🔍 Build Retriever
# ═══════════════════════════════════════════════════════════

def get_retriever(
    vectorstore: Chroma,
    k: int = 3,
    search_type: str = "similarity",
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    score_threshold: float = 0.3,
    filter: Optional[Dict[str, Any]] = None,
) -> VectorStoreRetriever:
    """
    Build a configurable retriever.

    Args:
        vectorstore: Chroma instance
        k: Number of results
        search_type:
            - "similarity": cosine similarity
            - "mmr": max marginal relevance (diverse)
            - "similarity_score_threshold": min score filter
        fetch_k: MMR candidates before reranking
        lambda_mult: MMR diversity (0 = diverse, 1 = relevant)
        score_threshold: For threshold search
        filter: Metadata filter (e.g. {"page": 1})

    Returns:
        Configured retriever
    """
    if search_type == "mmr":
        search_kwargs = {
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        }
    elif search_type == "similarity_score_threshold":
        search_kwargs = {
            "k": k,
            "score_threshold": score_threshold,
        }
    else:  # similarity
        search_kwargs = {"k": k}

    if filter:
        search_kwargs["filter"] = filter

    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


# ═══════════════════════════════════════════════════════════
#   🔎 Search with Scores
# ═══════════════════════════════════════════════════════════

def search_with_scores(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
    filter: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Document, float]]:
    """
    Search and return (Document, score) tuples.

    Lower score = more similar (Chroma uses L2 distance).
    """
    return vectorstore.similarity_search_with_score(query, k=k, filter=filter)


def search_with_relevance(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
    filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Search and return results with a human-friendly 'relevance' score (0-1).

    relevance = 1 / (1 + distance)
    """
    results = vectorstore.similarity_search_with_score(query, k=k, filter=filter)
    output = []
    for doc, dist in results:
        output.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "distance": float(dist),
            "relevance": round(1 / (1 + float(dist)), 4),
        })
    return output


# ═══════════════════════════════════════════════════════════
#   🎯 Cross-Encoder Reranking
# ═══════════════════════════════════════════════════════════

_reranker_cache: Dict[str, Any] = {}


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = 3,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> List[Document]:
    """
    Rerank documents using a CrossEncoder (higher quality than vector search alone).

    Args:
        query: User query
        documents: Candidate documents
        top_n: Return only top N
        model_name: CrossEncoder model ID

    Returns:
        Top-N reranked documents
    """
    try:
        from sentence_transformers import CrossEncoder

        if model_name not in _reranker_cache:
            print(f"🔤 Loading reranker: {model_name}")
            _reranker_cache[model_name] = CrossEncoder(model_name, max_length=512)

        reranker = _reranker_cache[model_name]
        pairs = [(query, d.page_content) for d in documents]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_n]]
    except Exception as e:
        print(f"⚠️ Reranking failed: {e}")
        return documents[:top_n]


# ═══════════════════════════════════════════════════════════
#   🧹 Collection Management
# ═══════════════════════════════════════════════════════════

def _delete_collection(persist_dir: str, collection_name: str) -> None:
    """Delete a collection from disk (silently ignores if not found)."""
    try:
        temp_store = Chroma(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_function=None,
        )
        temp_store.delete_collection()
    except Exception:
        pass


def delete_all_data(persist_dir: str = DEFAULT_PERSIST_DIR) -> None:
    """Delete the entire Chroma persistence directory. ⚠️ Irreversible!"""
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"🗑️ Deleted {persist_dir}")
    else:
        print(f"⚠️ {persist_dir} doesn't exist")


def list_collections(persist_dir: str = DEFAULT_PERSIST_DIR) -> List[str]:
    """List all collection names in the persistence directory."""
    if not os.path.exists(persist_dir):
        return []
    try:
        import chromadb
        client = chromadb.PersistentClient(path=persist_dir)
        return [c.name for c in client.list_collections()]
    except Exception as e:
        print(f"⚠️ Could not list collections: {e}")
        return []


def delete_collection(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> bool:
    """Delete a single collection. Returns True on success."""
    try:
        _delete_collection(persist_dir, collection_name)
        return True
    except Exception as e:
        print(f"❌ Delete failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════
#   📊 Vectorstore Info & Analytics
# ═══════════════════════════════════════════════════════════

def get_vectorstore_info(vectorstore: Chroma) -> Dict[str, Any]:
    """Get info about a vectorstore (count, name, sample metadata)."""
    try:
        collection = vectorstore._collection
        count = collection.count()

        # Sample a few to extract metadata keys
        sample = collection.get(limit=min(10, count)) if count > 0 else {}
        metas = sample.get("metadatas", []) or []
        meta_keys = set()
        for m in metas:
            if isinstance(m, dict):
                meta_keys.update(m.keys())

        return {
            "collection_name": collection.name,
            "document_count": count,
            "metadata_fields": sorted(meta_keys),
        }
    except Exception as e:
        return {"error": str(e)}


def get_disk_usage(persist_dir: str = DEFAULT_PERSIST_DIR) -> Dict[str, Any]:
    """Report how much disk space the vector DB uses."""
    if not os.path.exists(persist_dir):
        return {"exists": False, "size_mb": 0}

    total = 0
    file_count = 0
    for root, _, files in os.walk(persist_dir):
        for f in files:
            fp = os.path.join(root, f)
            total += os.path.getsize(fp)
            file_count += 1

    return {
        "exists": True,
        "size_mb": round(total / (1024 * 1024), 2),
        "file_count": file_count,
    }


# ═══════════════════════════════════════════════════════════
#   📦 Backup & Restore
# ═══════════════════════════════════════════════════════════

def backup_vectorstore(
    persist_dir: str = DEFAULT_PERSIST_DIR,
    backup_dir: Optional[str] = None,
) -> str:
    """
    Backup the Chroma directory to a timestamped zip.

    Returns:
        Path to the backup file
    """
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"❌ {persist_dir} not found")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if backup_dir is None:
        backup_dir = f"./backups/chroma_{timestamp}"

    os.makedirs(os.path.dirname(backup_dir), exist_ok=True)

    archive_path = shutil.make_archive(
        base_name=backup_dir,
        format="zip",
        root_dir=persist_dir,
    )
    print(f"💾 Backup saved: {archive_path}")
    return archive_path


def restore_vectorstore(
    backup_zip: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> None:
    """Restore Chroma from a backup zip (overwrites existing)."""
    if not os.path.exists(backup_zip):
        raise FileNotFoundError(f"❌ Backup not found: {backup_zip}")

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    os.makedirs(persist_dir, exist_ok=True)
    shutil.unpack_archive(backup_zip, persist_dir, format="zip")
    print(f"♻️ Restored to {persist_dir}")


# ═══════════════════════════════════════════════════════════
#   🌐 Multi-Collection (Workspaces)
# ═══════════════════════════════════════════════════════════

def list_workspaces(persist_dir: str = DEFAULT_PERSIST_DIR) -> List[str]:
    """Alias for list_collections (concept: workspaces)."""
    return list_collections(persist_dir)


def create_workspace(
    name: str,
    persist_dir: str = DEFAULT_PERSIST_DIR,
) -> None:
    """Create an empty collection (workspace)."""
    os.makedirs(persist_dir, exist_ok=True)
    try:
        embeddings = get_embedding_model()
        Chroma(
            collection_name=name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        print(f"✅ Workspace created: {name}")
    except Exception as e:
        print(f"❌ Failed: {e}")


# ═══════════════════════════════════════════════════════════
#   📈 Embedding Benchmark
# ═══════════════════════════════════════════════════════════

def benchmark_embedding_model(
    model_name: str,
    sample_texts: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Benchmark a single embedding model.

    Returns:
        Dict with load time, encode time, dim, device
    """
    if sample_texts is None:
        sample_texts = [
            "This is a short test sentence.",
            "این یک جمله آزمایشی کوتاه است.",
            "Machine learning is transforming industries.",
            "یادگیری ماشین در حال تغییر صنایع است.",
        ] * 5  # 20 texts total

    t0 = time.time()
    model = get_embedding_model(model_name, device=device)
    load_time = time.time() - t0

    t0 = time.time()
    embeddings = model.embed_documents(sample_texts)
    encode_time = time.time() - t0

    return {
        "model": model_name,
        "device": device or _detect_device(),
        "load_time": round(load_time, 3),
        "encode_time": round(encode_time, 3),
        "texts_per_sec": round(len(sample_texts) / encode_time, 1) if encode_time > 0 else 0,
        "dim": len(embeddings[0]) if embeddings else 0,
    }


def compare_embedding_models(
    models: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Benchmark multiple models and return results."""
    if models is None:
        models = list(EMBEDDING_MODELS.keys())

    results = []
    for m in models:
        try:
            results.append(benchmark_embedding_model(m, device=device))
        except Exception as e:
            results.append({"model": m, "error": str(e)})
    return results


# ═══════════════════════════════════════════════════════════
#   🧪 Self-Test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Testing rag_pipeline.py")
    print(f"Device detected: {_detect_device()}")
    print(f"Embedding models available: {len(EMBEDDING_MODELS)}")
    for name, info in EMBEDDING_MODELS.items():
        print(f"  • {name}")
        print(f"      → {info['desc']} ({info['dim']}d, ~{info['size_mb']}MB, {info['speed']})")

    print("\n📊 Disk usage:")
    print(get_disk_usage())

    print("\n🌐 Collections:")
    print(list_collections())

    print("\n✅ rag_pipeline.py is ready.")