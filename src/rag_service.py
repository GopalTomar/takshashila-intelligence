"""
rag_service.py — the ONE canonical RAG entry point.

Every interface (Streamlit dashboard, Mattermost bot, HTTP API) calls
:func:`answer_query` here. It is a thin, stable adapter over the shared engine in
``src.rag_pipeline`` — it does NOT re-implement retrieval, reranking, evidence
validation, citation generation or grounding. Its job is to:

  * make sure the process is serving the CURRENT knowledge-base version
    (``vector_store.load_index`` self-heals across processes),
  * run the shared pipeline,
  * attach a request id + the active KB version,
  * emit one structured, secret-free log line per request, and
  * return a single normalized response contract that the dashboard, the bot and
    the API all consume identically — so the same question yields materially the
    same answer regardless of which interface asked it.

Response contract (see also README "Frontend/backend contract")::

    {
      "request_id":   str,
      "query":        str,
      "answer":       str,
      "grounded":     bool,
      "confidence":   "HIGH" | "MEDIUM" | "LOW" | "NONE",
      "sources":      [ {title, url, source_type, snippet, author?, date?} ],
      "kb_version":   str,
      "model":        str,
      "interface":    str,
      "latency_ms":   int,
      "retrieval_ms": int,
      "generation_ms":int,
    }
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional

from src import config
from src.utils import get_logger

logger = get_logger("rag_service")


def _public_source(ch: Dict) -> Dict:
    """A citation record safe to send to any client (no internal scores/text dumps)."""
    text = (ch.get("text") or "").strip()
    snippet = (text[:280] + "…") if len(text) > 280 else text
    src = (ch.get("source") or ch.get("source_type") or "").lower()
    return {
        "title": ch.get("title") or "Untitled",
        "url": ch.get("url") or ch.get("original_url") or "",
        "source_type": src,
        "source_name": ch.get("source_name") or config.source_display_name(src, src),
        "author": (ch.get("author") or "") or None,
        "date": (ch.get("date") or "") or None,
        "snippet": snippet,
    }


def kb_version() -> str:
    """Active KB version string (cheap read of the manifest)."""
    from src import kb_version as kbv
    return kbv.current_version()


def answer_query(
    query: str,
    *,
    interface: str = "api",
    request_id: Optional[str] = None,
    top_k: int = config.TOP_K,
    model: Optional[str] = None,
    length: str = "normal",
    temperature: float = config.DEFAULT_TEMP,
    conversation_id: Optional[str] = None,
    **filters,
) -> Dict:
    """
    Canonical RAG call. ``filters`` may include source/category/author/year, which
    are passed straight through to the shared pipeline.
    """
    request_id = request_id or uuid.uuid4().hex[:12]
    query = (query or "").strip()
    started = time.perf_counter()

    if not query:
        return {
            "request_id": request_id, "query": query, "answer": "",
            "grounded": False, "confidence": "NONE", "sources": [],
            "kb_version": kb_version(), "model": (model or config.GROQ_MODEL),
            "interface": interface, "latency_ms": 0,
            "retrieval_ms": 0, "generation_ms": 0,
            "error": "empty_query",
        }

    # Ensure this process is serving the current KB (auto-reload if a rebuild
    # happened elsewhere). Cheap when nothing changed.
    from src import vector_store
    try:
        vector_store.load_index()
    except FileNotFoundError:
        return {
            "request_id": request_id, "query": query,
            "answer": "The knowledge base has not been built yet.",
            "grounded": False, "confidence": "NONE", "sources": [],
            "kb_version": kb_version(), "model": (model or config.GROQ_MODEL),
            "interface": interface, "latency_ms": 0,
            "retrieval_ms": 0, "generation_ms": 0, "error": "no_index",
        }

    from src.rag_pipeline import answer as _answer

    result = _answer(
        query=query, top_k=top_k, model=model or config.GROQ_MODEL,
        temperature=temperature, length=length, **filters,
    )

    confidence = str(result.get("confidence", "none")).upper()
    sources = [_public_source(s) for s in (result.get("sources") or [])]
    grounded = bool(sources) and confidence != "NONE"
    latency_ms = int((time.perf_counter() - started) * 1000)
    retrieval_ms = int(float(result.get("retrieval_time", 0.0)) * 1000)
    generation_ms = int(float(result.get("generation_time", 0.0)) * 1000)

    # One structured, secret-free log line per request — never logs the API key,
    # tokens, or full user payloads beyond the (short) query length.
    logger.info(
        "rag_request id=%s interface=%s kb=%s conf=%s grounded=%s "
        "chunks=%d sources=%d retrieval_ms=%d generation_ms=%d latency_ms=%d qlen=%d",
        request_id, interface, kb_version(), confidence, grounded,
        len(result.get("chunks") or []), len(sources),
        retrieval_ms, generation_ms, latency_ms, len(query),
    )

    return {
        "request_id": request_id,
        "query": query,
        "conversation_id": conversation_id,
        "answer": result.get("answer", ""),
        "grounded": grounded,
        "confidence": confidence,
        "sources": sources,
        "kb_version": kb_version(),
        "model": (model or config.GROQ_MODEL),
        "interface": interface,
        "top_score": round(float(result.get("top_score", 0.0)), 4),
        "latency_ms": latency_ms,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
    }


def status() -> Dict:
    """
    Operational status for /health, /ready, /rag/status and the admin panel.
    Contains only safe, non-secret operational facts.
    """
    from src import kb_version as kbv
    manifest = kbv.read_manifest() or {}
    from src import vector_store
    ready = False
    vectors = 0
    try:
        vectors = vector_store.ntotal()
        ready = vectors > 0 or config.FAISS_INDEX.exists()
    except Exception:
        ready = config.FAISS_INDEX.exists()

    return {
        "service": "takshashila-intelligence-rag",
        "ready": bool(config.FAISS_INDEX.exists()),
        "kb_version": manifest.get("version", "unversioned"),
        "kb_built_at": manifest.get("built_at"),
        "kb_counts": manifest.get("counts", {}),
        "kb_source_counts": manifest.get("source_counts", {}),
        "index_hash": manifest.get("index_hash"),
        "loaded_vectors": vectors,
        "loaded_version": vector_store.loaded_fingerprint(),
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.GROQ_MODEL,
        "available_models": list(config.AVAILABLE_MODELS),
        "groq_configured": bool(config.GROQ_API_KEY),
    }
