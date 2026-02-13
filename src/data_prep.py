#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_prep.py - Data loading and prompt preparation
"""

import os, re, json, sys
from typing import List, Dict, Any, Optional

# ---------- Optional deps ----------
_HAS_DATASETS = True
try:
    from datasets import load_dataset, concatenate_datasets
except Exception:
    _HAS_DATASETS = False

# Import from retrieval module
from retrieval import build_snippets, PerExampleRetriever, _iter_fields


# ============================================================================
# Instruction templates
# ============================================================================
INSTRUCTION_TEMPLATE = (
    "Your task is to generate a personalized response to the user's question. "
    "To do this, you can perform a series of actions, including thinking in <think> and </think> tags, "
    "searching for information from the user's past interactions with the system (i.e., previously asked questions "
    "and detailed information needs) by generating a search query in <search> and </search> tags, "
    "and finally providing the answer in <answer> and </answer> tags. "
    "The retrieved information from user history will be provided to you inside <information> and </information> tags. "
    "You need to first think about the question and how to generate a personalized answer for the user. "
    "In this thinking process, you should try to understand the user's preferences and needs based on their past interactions with the system. "
    "The thinking process should be inside <think> and </think> tags, and the final answer should be inside <answer> and </answer> tags. "
    "If you need to search for information about the user from their history, you can do this by generating a non-empty search query inside <search> and </search> tags. "
    "The retrieved information will be provided to you inside <information> and </information> tags. "
    "You can use this information in the thinking process and answer generation. "
    "Nothing should be outside the mentioned tags except the initial question provided to you. "
    "Now, answer the following question:\n"
).strip()


# ============================================================================
# HuggingFace dataset utilities
# ============================================================================
def _normalize_hf_configs(cfg):
    """
    Normalize config input to list of strings.
    Accepts: 'A,B,C' | 'A B C' | ['A','B','C'] | 'A' -> ['A','B','C']
    """
    if isinstance(cfg, (list, tuple)):
        return [str(x) for x in cfg]
    if isinstance(cfg, str):
        return [c for c in re.split(r"[,\s]+", cfg.strip()) if c]
    return [str(cfg)]


def load_default_dataset(config, split: str, n: Optional[int]) -> List[Dict[str, Any]]:
    """
    Load dataset from HuggingFace LaMP-QA.
    
    Args:
        config: Dataset config name(s) - can be string, list, or comma-separated
        split: Dataset split (train/validation/test)
        n: Optional limit on number of examples
    
    Returns:
        List of dataset records as dicts
    """
    assert _HAS_DATASETS, "Install datasets or provide --input-jsonl"

    configs = _normalize_hf_configs(config)
    if not configs:
        raise ValueError("No valid --hf-config provided.")

    # Load and concatenate multiple configs if needed
    parts = [load_dataset("alireza7/LaMP-QA", name=c, split=split) for c in configs]
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)

    # Limit number of examples if requested
    if n is not None and n >= 0:
        ds = ds.select(range(min(n, len(ds))))
    
    return list(ds)


def load_input_jsonl(path: str, n: Optional[int]) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.
    
    Args:
        path: Path to JSONL file
        n: Optional limit on number of lines
    
    Returns:
        List of parsed JSON objects
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if n is not None and n >= 0 and len(rows) >= n:
                break
    return rows


# ============================================================================
# Metadata extraction helpers
# ============================================================================
def extract_details(rec: Dict[str, Any]) -> str:
    """
    Extract details/description from record.
    Tries multiple fields in priority order.
    """
    # Try explicit detail fields first
    for key in ("details", "detail", "detailed_explanation", "description"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    
    # Fall back to history
    hist = " ".join(list(_iter_fields(rec, "history") or [])) or ""
    if hist.strip():
        return hist.strip()
    
    # Fall back to profile
    prof = " ".join(list(_iter_fields(rec, "profile") or [])) or ""
    if prof.strip():
        return prof.strip()
    
    return ""


def extract_aspects(rec: Dict[str, Any], fallback_question: str) -> List[Dict[str, str]]:
    """
    Extract evaluation aspects from record.
    If not present, creates default aspect.
    """
    aspects = rec.get("aspects")
    if isinstance(aspects, list) and aspects and all(isinstance(x, dict) for x in aspects):
        return [
            {
                "aspect": str(x.get("aspect", "")),
                "reason": str(x.get("reason", "")),
                "evidence": str(x.get("evidence", "")),
            }
            for x in aspects
        ]
    
    # Default aspect if none provided
    return [{
        "aspect": "Answer relevance",
        "reason": "",
        "evidence": fallback_question
    }]


# ============================================================================
# Controller for building prompts
# ============================================================================
class PromptBuilder:
    """
    Helper class for building training prompts with retrieval.
    Stateless - can be used across multiple examples.
    """
    
    @staticmethod
    def format_info_block(items: List[tuple]) -> str:
        """Format retrieved snippets as numbered list."""
        if not items:
            return ""
        return "\n".join(
            f"[{i}] ({snip.source}, score={score:.3f}) {snip.text}"
            for i, (snip, score) in enumerate(items, 1)
        )
    
    @staticmethod
    def build_rag_prompt(question: str, retriever: PerExampleRetriever, topk: int = 3) -> str:
        """
        Build a RAG prompt with embedded retrieval results.
        
        Args:
            question: User question
            retriever: Retriever with user's history/profile
            topk: Number of snippets to retrieve
        
        Returns:
            Complete prompt with <information> block
        """
        # Retrieve information
        hits = retriever.search(question, k=topk) if question else []
        
        # Guarantee something in <information> if we have any snippets
        if (not hits) and getattr(retriever, "snippets", None):
            take = min(topk, len(retriever.snippets))
            hits = [(retriever.snippets[i], 0.0) for i in range(take)]
        
        info_block = PromptBuilder.format_info_block(hits)
        
        # Build complete prompt
        prompt = (
            f"{INSTRUCTION_TEMPLATE}\n\n"
            f"Question:\n{question}\n\n"
            f"<information>\n{info_block}\n</information>\n\n"
            "(Answering)"
        )
        
        return prompt
    
    @staticmethod
    def build_vanilla_prompt(question: str) -> str:
        """
        Build a vanilla prompt without retrieval.
        
        Args:
            question: User question
        
        Returns:
            Prompt without <information> block
        """
        prompt = (
            f"{INSTRUCTION_TEMPLATE}\n\n"
            f"Question:\n{question}\n\n"
            "(Answering)"
        )
        return prompt


# ============================================================================
# Main prompt preparation
# ============================================================================
def prepare_stacked_prompts(args, planner_llm=None) -> List[Dict[str, Any]]:
    """
    Prepare training prompts: K RAG + M Vanilla per question.
    
    This is the main entry point for data preparation. It:
    1. Loads data from JSONL or HuggingFace
    2. Extracts metadata (details, aspects)
    3. Builds retrievers from user history/profile
    4. Generates K RAG prompts + M vanilla prompts per question
    
    Args:
        args: Command-line arguments with configuration
        planner_llm: Optional vLLM engine (not used in current implementation)
    
    Returns:
        List of dicts with: prompt, group_id, mode, question, details, aspects
    """
    # Load data
    if args.input_jsonl and os.path.exists(args.input_jsonl):
        data = load_input_jsonl(args.input_jsonl, args.n)

        # Check if this is already precomputed prompts
        if data and all(isinstance(r, dict) and "prompt" in r for r in data):
            print(f"[PREP] Loaded {len(data)} precomputed prompts from {args.input_jsonl}", file=sys.stderr)
            return data

        # Normalize question field
        for r in data:
            if "question" not in r and "prompt" in r:
                r["question"] = r.get("prompt", "")
    else:
        data = load_default_dataset(args.hf_config, args.hf_split, args.n)

    # Parse sources for retrieval
    sources = [s.strip() for s in (args.sources or "").split(",") if s.strip()] or ["profile"]
    
    # Get stacking parameters
    rag_per_q = max(1, getattr(args, "rag_per_q", 4))
    van_per_q = max(1, getattr(args, "van_per_q", 1))
    topk = getattr(args, "topk", 3)
    
    prepared: List[Dict[str, Any]] = []

    # Process each record
    for idx, rec in enumerate(data):
        qid = str(rec.get("id", idx))
        question = (rec.get("question", "") or rec.get("prompt", "")).strip()

        # Extract metadata
        details = extract_details(rec)
        aspects = extract_aspects(rec, question)

        # Build retriever for this example
        snippets = build_snippets(rec, sources=sources)
        retr = PerExampleRetriever(snippets)

        # Generate K RAG prompts
        for _ in range(rag_per_q):
            rag_prompt = PromptBuilder.build_rag_prompt(question, retriever=retr, topk=topk)
            prepared.append({
                "group_id": qid,
                "mode": "rag",
                "prompt": rag_prompt,
                "question": question,
                "details": details,
                "aspects": aspects,
            })

        # Generate M Vanilla prompts
        for _ in range(van_per_q):
            vanilla_prompt = PromptBuilder.build_vanilla_prompt(question)
            prepared.append({
                "group_id": qid,
                "mode": "van",
                "prompt": vanilla_prompt,
                "question": question,
                "details": details,
                "aspects": aspects,
            })

        print(
            f"[PREP] {idx+1}/{len(data)} qid={qid} "
            f"({rag_per_q}x RAG + {van_per_q}x VAN)",
            file=sys.stderr,
        )

    return prepared


# ============================================================================
# Export prepared data to JSONL
# ============================================================================
def save_prepared_prompts(prepared: List[Dict[str, Any]], output_path: str):
    """
    Save prepared prompts to JSONL file for later reuse.
    
    Args:
        prepared: List of prepared prompt dicts
        output_path: Path to output JSONL file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for item in prepared:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[PREP] Saved {len(prepared)} prompts to {output_path}", file=sys.stderr)
