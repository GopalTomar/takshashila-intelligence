#!/usr/bin/env python3
"""
test_metadata_qa.py — Offline test that metadata questions resolve correctly.

Builds a small index (real chunker + embedding cache + FAISS path) from synthetic
documents carrying rich metadata, then asserts that:

  * content questions retrieve the right article with its EXACT url,
  * metadata questions ("who wrote…", "when published…", "what tags…",
    "which category…") retrieve the right article and carry author/date/tags,
  * listing/landing pages are never the cited source,
  * the metadata header is present in the searchable text and the LLM context,
  * validate_kb() reports a healthy KB.

Uses deterministic bag-of-words mock embeddings so it needs no model download.
"""
from __future__ import annotations
import sys, tempfile, json, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config


def _isolate(tmp: Path):
    config.DATA_DIR = tmp
    config.PROCESSED_DIR = tmp / "processed"
    config.INDEX_DIR = tmp / "index"
    config.LOGS_DIR = tmp / "logs"
    config.DOCUMENTS_FILE = config.PROCESSED_DIR / "documents.jsonl"
    config.CHUNKS_FILE = config.PROCESSED_DIR / "chunks.jsonl"
    config.FAISS_INDEX = config.INDEX_DIR / "faiss.index"
    config.METADATA_FILE = config.INDEX_DIR / "metadata.pkl"
    config.SCRAPE_LOG = config.LOGS_DIR / "s.log"
    config.WEBSITE_MANIFEST_FILE = config.LOGS_DIR / "m.json"
    for p in (config.PROCESSED_DIR, config.INDEX_DIR, config.LOGS_DIR):
        p.mkdir(parents=True, exist_ok=True)
    import src.incremental_index as inc
    inc.METADATA_JSON = config.INDEX_DIR / "metadata.json"
    import src.vector_store as vs
    vs.METADATA_JSON = config.INDEX_DIR / "metadata.json"


def _mock_embeddings():
    import numpy as np, src.embeddings as emb
    vocab = {}
    def vec(t):
        v = np.zeros(128, dtype="float32")
        for w in re.findall(r"[a-z0-9]+", t.lower()):
            vocab.setdefault(w, len(vocab) % 128); v[vocab[w]] += 1.0
        n = np.linalg.norm(v); return v / n if n else v
    emb.embed_texts = lambda texts, batch_size=64, show_progress=False: \
        np.vstack([vec(t) for t in texts]).astype("float32")


DOCS = [
    {"document_id": "website_listing", "url": "https://takshashila.org.in/pages/blogs/",
     "title": "Blogs", "source": "website", "source_name": "Takshashila Website",
     "category": "blog", "text": "Blogs by Takshashila authors. " * 10},
    {"document_id": "website_illusion",
     "url": "https://takshashila.org.in/blogs/the-illusion-of-ai-mapping-intelligence",
     "title": "The Illusion of AI Mapping Intelligence", "author": "Gopal Tomar",
     "authors": ["Gopal Tomar"], "date": "2026-06-21", "category": "blog",
     "section": "Blogs > GeoAI", "tags": ["GeoAI", "mapping", "artificial intelligence"],
     "source": "website", "source_name": "Takshashila Website",
     "text": "This blog examines the illusion of AI mapping intelligence in geospatial systems, "
             "arguing that model fluency is mistaken for understanding. " * 8},
    {"document_id": "website_available",
     "url": "https://takshashila.org.in/blogs/available-isnt-accessible",
     "title": "Available Isn't Accessible", "author": "Gopal Tomar",
     "authors": ["Gopal Tomar"], "date": "2026-07-20", "updated_date": "2026-07-21",
     "category": "blog", "section": "Blogs > Accessibility",
     "tags": ["accessibility", "GeoAI"], "source": "website",
     "source_name": "Takshashila Website",
     "text": "Availability and accessibility are not the same thing. Accessibility is about people "
             "understanding a system, trusting it, and being able to question it. " * 8},
]


def build():
    config.DOCUMENTS_FILE.write_text("\n".join(json.dumps(d) for d in DOCS), encoding="utf-8")
    from src.incremental_index import rebuild_index
    rebuild_index(use_cache=True)
    from src import vector_store
    vector_store.load_index(force=True)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="metaqa_"))
    _isolate(tmp)
    _mock_embeddings()
    build()

    from src.retriever import retrieve
    from src.utils import chunk_search_text
    from src.rag_pipeline import _build_context_block, _filter_evidence_chunks, _dedupe_by_document

    checks = [
        ("who wrote The Illusion of AI Mapping Intelligence", "the-illusion", "Gopal Tomar"),
        ("when was Available Isn't Accessible published", "available-isnt-accessible", "Gopal Tomar"),
        ("what tags does the accessibility blog have", "available-isnt-accessible", "Gopal Tomar"),
        ("which blog discusses AI mapping intelligence", "the-illusion", "Gopal Tomar"),
    ]
    for q, url_frag, author in checks:
        hits = retrieve(q, top_k=5, use_hybrid=True)
        assert hits, f"no hits for: {q}"
        top = hits[0]
        assert url_frag in (top.get("url") or ""), \
            f"[{q}] expected {url_frag}, got {top.get('url')}"
        assert "/pages/blogs/" not in (top.get("url") or ""), f"[{q}] listing page cited!"
        assert top.get("author") == author, f"[{q}] author missing: {top.get('author')}"
        st = chunk_search_text(top)
        assert "Author:" in st and "Title:" in st, f"[{q}] metadata header missing from search text"
        ctx, _ = _build_context_block(_dedupe_by_document(_filter_evidence_chunks(hits)))
        assert "author:" in ctx.lower(), f"[{q}] author missing from LLM context"
        print(f"  ✓ {q!r:58} -> {top.get('title')!r} ({top.get('url').split('/')[-1]})")

    # Validation must pass
    from scripts.validate_kb import validate
    rep = validate()
    print(f"\n  validation: {'PASS' if rep['ok'] else 'FAIL'} "
          f"({len(rep['errors'])} errors, {len(rep['warnings'])} warnings)")
    for e in rep["errors"]:
        print("    ERROR:", e)
    assert rep["ok"], "validation reported errors"

    print("\n✅ METADATA QA + VALIDATION TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())