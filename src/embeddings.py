"""
embeddings.py — Local sentence-transformer embeddings for FAISS indexing
"""

import os
from typing import List

import numpy as np

# Suppress Windows symlink warning that can cause cache allocation failures
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from src import config
from src.utils import get_logger

logger = get_logger("embeddings", config.SCRAPE_LOG)

_MODEL = None   # lazy singleton


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")

        last_exc = None
        for attempt in range(1, 4):
            try:
                _MODEL = SentenceTransformer(config.EMBEDDING_MODEL)
                logger.info("Embedding model loaded successfully")
                break
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"Model load attempt {attempt} failed: {exc}. "
                    + ("Retrying…" if attempt < 3 else "Giving up.")
                )

        if _MODEL is None:
            raise RuntimeError(
                f"\n\n❌  Could not load embedding model '{config.EMBEDDING_MODEL}' "
                f"after 3 attempts.\n"
                f"Last error: {last_exc}\n\n"
                "── Fixes ──────────────────────────────────────────────────\n"
                "1. Delete the broken cache folder and retry:\n"
                "   C:\\Users\\<you>\\.cache\\huggingface\\hub\\"
                "models--BAAI--bge-small-en-v1.5\\\n\n"
                "2. Or switch to the lighter 22 MB model — edit your .env:\n"
                "   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2\n\n"
                "3. Or run VS Code / terminal as Administrator to allow\n"
                "   Windows symlinks (needed for HF cache on some machines).\n"
            )
    return _MODEL


def embed_texts(texts: List[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """
    Embed a list of texts.
    Returns normalised float32 numpy array of shape (N, DIM).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # cosine via inner-product
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string; return shape (1, DIM) float32."""
    return embed_texts([query])
