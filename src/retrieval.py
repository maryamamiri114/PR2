#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieval.py - RAG retrieval system with FAISS and embeddings
"""

import os, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Iterable

import numpy as np
import faiss

# ---------- Optional deps ----------
_HAS_ST = True
try:
    from sentence_transformers import SentenceTransformer, models
except Exception:
    _HAS_ST = False


# ============================================================================
# Embedding model setup
# ============================================================================
EMB_MODEL = os.getenv("EMB_MODEL", "facebook/contriever-msmarco")
EMB_DEVICE = os.getenv("EMB_DEVICE")  # e.g., "cuda" or "cpu"

def build_sbert(name: str):
    """Build sentence transformer model, wrapping HF models if needed."""
    from sentence_transformers import SentenceTransformer, models
    # If it's a native Sentence-Transformers checkpoint, load directly.
    if name.startswith("sentence-transformers/"):
        return SentenceTransformer(name)
    # Otherwise (e.g., "facebook/contriever-msmarco"), wrap HF model + mean pooling.
    word = models.Transformer(name)
    pool = models.Pooling(
        word.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    return SentenceTransformer(modules=[word, pool])

_sbert = build_sbert(EMB_MODEL) if _HAS_ST else None
if _sbert and EMB_DEVICE:
    _sbert.to(EMB_DEVICE)


# ============================================================================
# Data structures
# ============================================================================
@dataclass
class Snippet:
    text: str
    source: str


# ============================================================================
# Snippet building from records
# ============================================================================
def _iter_fields(rec: Dict[str, Any], key: str) -> Iterable[str]:
    """Iterator over field values in a record, handling strings and lists."""
    val = rec.get(key)
    if val is None: 
        return
    if isinstance(val, str):
        s = val.strip()
        if s: 
            yield s
        return
    if isinstance(val, list):
        for item in val:
            if isinstance(item, dict) and "text" in item:
                s = str(item["text"]).strip()
                if s: 
                    yield s
            else:
                s = str(item).strip()
                if s: 
                    yield s


def build_snippets(rec: Dict[str, Any], sources: List[str]) -> List[Snippet]:
    """
    Extract snippets from a record based on source fields.
    
    Args:
        rec: Record dictionary with fields like 'profile', 'history', etc.
        sources: List of field names to extract from (e.g., ['profile', 'history'])
    
    Returns:
        List of Snippet objects
    """
    out: List[Snippet] = []
    for src in sources:
        src_u = src.upper()
        if src == "profile":
            for s in _iter_fields(rec, "profile"):
                out.append(Snippet(text=s, source="PROFILE"))
        elif src == "history":
            for s in _iter_fields(rec, "history"):
                out.append(Snippet(text=s, source="HISTORY"))
        elif src in ("past_questions", "past_q"):
            for s in _iter_fields(rec, "past_questions"):
                out.append(Snippet(text=s, source="PAST_QUESTIONS"))
        else:
            for s in _iter_fields(rec, src):
                out.append(Snippet(text=s, source=src_u))
    return out


# ============================================================================
# Per-example retriever with FAISS
# ============================================================================
class PerExampleRetriever:
    """
    FAISS-based retriever for a single example's snippets.
    Encodes snippets once, then allows fast similarity search.
    """
    
    def __init__(self, snippets: List[Snippet]):
        self.snippets = snippets
        corpus = [s.text for s in snippets]
        
        if not corpus or _sbert is None:
            self.index = None
            self.emb = np.zeros((0, 384), dtype=np.float32)
            return
        
        # Encode all snippets
        self.emb = _sbert.encode(corpus, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(self.emb)
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.emb.shape[1])
        self.index.add(self.emb)

    def search(self, query: str, k: int = 3) -> List[Tuple[Snippet, float]]:
        """
        Search for top-k most similar snippets to the query.
        
        Args:
            query: Search query string
            k: Number of results to return
        
        Returns:
            List of (Snippet, score) tuples
        """
        if self.index is None or not self.snippets or _sbert is None:
            return []
        
        # Encode query
        q = _sbert.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q)
        
        # Search
        D, I = self.index.search(q, min(k, len(self.snippets)))
        return [(self.snippets[idx], float(score)) for score, idx in zip(D[0], I[0])]


# ============================================================================
# Tag utilities for parsing model outputs
# ============================================================================
def _tag_pat(name: str) -> re.Pattern:
    """Create regex pattern for XML-style tags."""
    return re.compile(rf"<\s*{name}\s*>(.*?)</\s*{name}\s*>", re.S | re.I)


TAG_RE = {
    "think": _tag_pat("think"),
    "search": _tag_pat("search"),
    "answer": _tag_pat("answer"),
    "info": _tag_pat("information"),
}


def get_tag(text: str, name: str) -> Optional[str]:
    """Extract content from XML-style tags like <think>...</think>."""
    m = TAG_RE[name].search(text or "")
    return m.group(1).strip() if m else None
