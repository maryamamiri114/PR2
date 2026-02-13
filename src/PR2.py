#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pr2.py - Main script for personalized RAG training
Coordinates vLLM engines, data loading, judging, and training
"""

import os, re, json, argparse, sys
from typing import List, Dict, Any, Optional
from pathlib import Path

# ---------- Optional deps ----------
_HAS_VLLM = True
try:
    from vllm import LLM, SamplingParams
except Exception:
    _HAS_VLLM = False

_HAS_OPENAI = True
try:
    from openai import OpenAI
except Exception:
    _HAS_OPENAI = False

_HAS_DATASETS = True
try:
    from datasets import load_dataset, concatenate_datasets
except Exception:
    _HAS_DATASETS = False

_HAS_TRANSFORMERS = True
try:
    from transformers import AutoTokenizer
except Exception:
    _HAS_TRANSFORMERS = False

# Import from other modules
from retrieval import (
    build_snippets, PerExampleRetriever, get_tag, _iter_fields
)
from training import train_trl_grpo_stacked


# ============================================================================
# Instruction templates
# ============================================================================
INSTRUCTION_TEMPLATE = """
Your task is to generate a personalized response to the user's question based on the required personalization aspects in the question.
You may think in <think>...</think>, search in <search>...</search>, read <information>...</information>, and answer in <answer>...</answer>.
After reading <information>, reflect whether you have enough to personalize the answer and show that reflection in <think>...</think>.
The final answer MUST be inside <answer>...</answer>. Produce no text outside tags.
""".strip()


ANS_FORMAT_HEADER = (
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
# vLLM management
# ============================================================================
ACTIVE_LLMS: List["LLM"] = []


def _with_visible_devices(devs: Optional[str], fn):
    """Temporarily set CUDA_VISIBLE_DEVICES for engine initialization."""
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devs is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devs)
    try:
        return fn()
    finally:
        if devs is not None:
            if prev is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev


def _init_llm_core(model: str,
                   dtype: str = "bfloat16",
                   tensor_parallel_size: int = 1,
                   max_model_len: int = 2048,
                   gpu_memory_utilization: float = 0.90) -> "LLM":
    """Initialize a vLLM engine."""
    assert _HAS_VLLM, "vLLM is required"
    llm = LLM(
        model=model,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    ACTIVE_LLMS.append(llm)
    return llm


def safe_shutdown():
    """Clean up vLLM engines and distributed processes."""
    for llm in ACTIVE_LLMS:
        try:
            llm.shutdown()
        except Exception:
            pass
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def _maybe_apply_chat_template(llm: "LLM", system_text: str, user_text: str) -> str:
    """Apply chat template if available, otherwise concatenate."""
    tok = llm.get_tokenizer()
    tmpl = getattr(tok, "chat_template", None)
    if tmpl:
        try:
            return tok.apply_chat_template(
                [{"role": "system", "content": system_text},
                 {"role": "user", "content": user_text}],
                tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return f"{system_text}\n\n{user_text}"


def call_llm(llm: "LLM",
             system_text: str,
             user_text: str,
             temperature: float = 0.7,
             max_tokens: int = 256,
             top_p: float = 0.95,
             stop: Optional[List[str]] = None) -> str:
    """Call vLLM engine for text generation."""
    formatted = _maybe_apply_chat_template(llm, system_text, user_text)
    sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=top_p, stop=stop or [])
    outs = llm.generate([formatted], sampling)
    return outs[0].outputs[0].text.strip()


# ============================================================================
# Controller for 2-step RAG
# ============================================================================
class Controller2Step:
    """
    Two-step RAG controller:
    1. Planning: Think + generate search query
    2. Retrieval + Answer: Get info + generate final answer
    """
    
    def __init__(self, llm: "LLM", template: str,
                 temperature: float = 0.7, top_p: float = 0.95,
                 topk_default: int = 3):
        self.llm = llm
        self.template = template
        self.temperature = temperature
        self.top_p = top_p
        self.topk_default = topk_default

    def _format_info_block(self, items: List[tuple]) -> str:
        """Format retrieved snippets as numbered list."""
        if not items:
            return ""
        return "\n".join(
            f"[{i}] ({snip.source}, score={score:.3f}) {snip.text}"
            for i, (snip, score) in enumerate(items, 1)
        )

    def build_training_prompt(self, question: str, retriever: PerExampleRetriever, topk: int = None) -> str:
        """
        Build a complete training prompt with embedded retrieval.
        For training, we retrieve once and embed the results directly in the prompt.
        """
        topk = topk or self.topk_default
        
        # Retrieve information
        hits = retriever.search(question, k=topk) if question else []
        
        # Guarantee something in <information> if we have any snippets
        if (not hits) and getattr(retriever, "snippets", None):
            take = min(topk, len(retriever.snippets))
            hits = [(retriever.snippets[i], 0.0) for i in range(take)]
        
        info_block = self._format_info_block(hits)
        
        # Build complete prompt
        prompt = (
            f"{ANS_FORMAT_HEADER}\n\n"
            f"Question:\n{question}\n\n"
            f"<information>\n{info_block}\n</information>\n\n"
            "(Answering)"
        )
        
        return prompt


# ============================================================================
# Data loading
# ============================================================================
def _normalize_hf_configs(cfg):
    """Normalize config input to list of strings."""
    if isinstance(cfg, (list, tuple)):
        return [str(x) for x in cfg]
    if isinstance(cfg, str):
        return [c for c in re.split(r"[,\s]+", cfg.strip()) if c]
    return [str(cfg)]


def load_default_dataset(config, split: str, n: Optional[int]) -> List[Dict[str, Any]]:
    """Load dataset from HuggingFace."""
    assert _HAS_DATASETS, "Install datasets or provide --input-jsonl"

    configs = _normalize_hf_configs(config)
    if not configs:
        raise ValueError("No valid --hf-config provided.")

    parts = [load_dataset("alireza7/LaMP-QA", name=c, split=split) for c in configs]
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)

    if n is not None and n >= 0:
        ds = ds.select(range(min(n, len(ds))))
    return list(ds)


def load_input_jsonl(path: str, n: Optional[int]) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
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
# Prompt preparation
# ============================================================================
def prepare_stacked_prompts(args) -> List[Dict[str, Any]]:
    """
    Prepare training prompts: K RAG + 1 Vanilla per question.
    
    Returns:
        List of dicts with: prompt, group_id, mode, question, details, aspects
    """
    # Load data
    if args.input_jsonl and os.path.exists(args.input_jsonl):
        data = load_input_jsonl(args.input_jsonl, args.n)

        # Check if this is already precomputed prompts
        if data and all(isinstance(r, dict) and "prompt" in r for r in data):
            print(f"[STACK] Loaded {len(data)} precomputed prompts from {args.input_jsonl}", file=sys.stderr)
            return data

        # Normalize question field
        for r in data:
            if "question" not in r and "prompt" in r:
                r["question"] = r.get("prompt", "")
    else:
        data = load_default_dataset(args.hf_config, args.hf_split, args.n)

    # Build planner LLM
    def _mk():
        return _init_llm_core(
            model=args.planner_model,
            dtype=args.planner_dtype or args.dtype,
            tensor_parallel_size=args.planner_tensor_parallel_size or args.tensor_parallel_size,
            max_model_len=args.planner_max_model_len or args.max_model_len,
            gpu_memory_utilization=(
                args.planner_gpu_mem
                if args.planner_gpu_mem is not None
                else (args.gpu_mem if args.gpu_mem is not None else 0.90)
            ),
        )

    planner_llm = _with_visible_devices(args.planner_devices, _mk)
    ctl = Controller2Step(
        planner_llm,
        INSTRUCTION_TEMPLATE,
        temperature=args.temperature,
        top_p=args.top_p,
        topk_default=args.topk,
    )

    sources = [s.strip() for s in (args.sources or "").split(",") if s.strip()] or ["profile"]
    prepared: List[Dict[str, Any]] = []

    for idx, rec in enumerate(data):
        qid = str(rec.get("id", idx))
        question = (rec.get("question", "") or rec.get("prompt", "")).strip()

        # Extract metadata
        def _get_details(r: Dict[str, Any]) -> str:
            for key in ("details", "detail", "detailed_explanation", "description"):
                val = r.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            hist = " ".join(list(_iter_fields(r, "history") or [])) or ""
            if hist.strip():
                return hist.strip()
            prof = " ".join(list(_iter_fields(r, "profile") or [])) or ""
            if prof.strip():
                return prof.strip()
            return ""

        def _get_aspects(r: Dict[str, Any], fallback_q: str):
            aspects = r.get("aspects")
            if isinstance(aspects, list) and aspects and all(isinstance(x, dict) for x in aspects):
                return [
                    {
                        "aspect": str(x.get("aspect", "")),
                        "reason": str(x.get("reason", "")),
                        "evidence": str(x.get("evidence", "")),
                    }
                    for x in aspects
                ]
            return [{"aspect": "Answer relevance", "reason": "", "evidence": fallback_q}]

        details = _get_details(rec)
        aspects = _get_aspects(rec, question)

        # Build retriever
        snippets = build_snippets(rec, sources=sources)
        retr = PerExampleRetriever(snippets)

        # Generate K RAG prompts
        for _ in range(max(1, args.rag_per_q)):
            rag_prompt = ctl.build_training_prompt(question, retriever=retr, topk=args.topk)
            prepared.append({
                "group_id": qid,
                "mode": "rag",
                "prompt": rag_prompt,
                "question": question,
                "details": details,
                "aspects": aspects,
            })

        # Generate 1 Vanilla prompt
        for _ in range(max(1, args.van_per_q)):
            vanilla_prompt = (
                f"{ANS_FORMAT_HEADER}\n\n"
                f"Question:\n{question}\n\n"
                "(Answering)"
            )
            prepared.append({
                "group_id": qid,
                "mode": "van",
                "prompt": vanilla_prompt,
                "question": question,
                "details": details,
                "aspects": aspects,
            })

        print(
            f"[STACK] {idx+1}/{len(data)} qid={qid} "
            f"({args.rag_per_q}x RAG + {args.van_per_q}x VAN)",
            file=sys.stderr,
        )

    return prepared


# ============================================================================
# Judge evaluation
# ============================================================================
_EVAL_PROMPT_SYSTEM = """You are a fair and insightful judge with exceptional reasoning and analytical abilities. Your task is to evaluate a user's question, a generated response, and an important aspect. Decide if the response addresses the aspect. Return JSON with: {"match_score": 0|1|2}."""

_EVAL_PROMPT_USER = """
question: {question}
details: {details}
response: {response}
aspect: {aspects}

Return only a valid JSON object in ```json``` with a single key "match_score".
"""


def _parse_json_lenient(json_str: str) -> Dict[str, Any]:
    """Parse JSON leniently, with fallbacks."""
    s = (json_str or "").replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        import json5
        return json5.loads(s)
    except Exception:
        pass
    m = re.search(r'"?match_score"?\s*:\s*([0-2])', s)
    if m:
        return {"match_score": int(m.group(1))}
    return {"match_score": 0}


def _format_aspect_text(asp: dict) -> str:
    """Format aspect dict as text."""
    return (
        f"-aspect: {str(asp.get('aspect', ''))}\n"
        f"    -reason: {str(asp.get('reason', ''))}\n"
        f"    -evidence: {str(asp.get('evidence', ''))}"
    )


def _truncate_to_ids(tok, text: Optional[str], max_toks: Optional[int]):
    """Truncate text to max tokens using tokenizer."""
    if tok is None or max_toks is None or max_toks <= 0:
        return text or ""
    ids = tok.encode(text or "", add_special_tokens=False)
    if len(ids) <= max_toks:
        return text or ""
    return tok.decode(ids[:max_toks], skip_special_tokens=True)


def _load_eval_tokenizer(args):
    """Load tokenizer for judge model."""
    name = args.eval_tokenizer or args.eval_model or args.trl_policy_model
    if not (name and _HAS_TRANSFORMERS):
        return None
    try:
        return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    except Exception:
        return None


def _count_toks(tok, text: str) -> int:
    """Count tokens in text."""
    if tok is None:
        return max(1, len(text or "") // 4)
    return len(tok.encode(text or "", add_special_tokens=False))


def _fit_prompt_to_budget(tok, system: str, q: str, d: str, r: str, a: str,
                          budget_prompt_only: int,
                          min_q: int = 32, min_d: int = 128,
                          min_r: int = 128, min_a: int = 32):
    """
    Fit prompt components to token budget by iterative truncation.
    Returns: (q_fit, d_fit, r_fit, a_fit, total_prompt_tokens)
    """
    # Start with current values
    q_fit, d_fit, r_fit, a_fit = q, d, r, a
    
    # Count tokens
    sys_toks = _count_toks(tok, system)
    q_toks = _count_toks(tok, q_fit)
    d_toks = _count_toks(tok, d_fit)
    r_toks = _count_toks(tok, r_fit)
    a_toks = _count_toks(tok, a_fit)
    
    # Template overhead (approximate)
    template_toks = _count_toks(tok, _EVAL_PROMPT_USER.format(question="", details="", response="", aspects=""))
    
    total = sys_toks + q_toks + d_toks + r_toks + a_toks + template_toks
    
    # Iteratively shrink if over budget
    while total > budget_prompt_only:
        # Shrink largest field that's above minimum
        if d_toks > min_d:
            d_fit = _truncate_to_ids(tok, d_fit, int(d_toks * 0.8))
            d_toks = _count_toks(tok, d_fit)
        elif r_toks > min_r:
            r_fit = _truncate_to_ids(tok, r_fit, int(r_toks * 0.8))
            r_toks = _count_toks(tok, r_fit)
        elif q_toks > min_q:
            q_fit = _truncate_to_ids(tok, q_fit, int(q_toks * 0.8))
            q_toks = _count_toks(tok, q_fit)
        elif a_toks > min_a:
            a_fit = _truncate_to_ids(tok, a_fit, int(a_toks * 0.8))
            a_toks = _count_toks(tok, a_fit)
        else:
            break  # Can't shrink further
        
        total = sys_toks + q_toks + d_toks + r_toks + a_toks + template_toks
    
    return q_fit, d_fit, r_fit, a_fit, total


def _create_eval_pairs_all(queries: List[str], responses: List[str], 
                           details_list: List[str], aspects_list: List[List[Dict[str, str]]]):
    """Create evaluation pairs for all queries and aspects."""
    pairs, ids = [], []
    for i, (q, r, d, aspects) in enumerate(zip(queries, responses, details_list, aspects_list)):
        aspects = aspects or [{"aspect": "Answer relevance", "reason": "", "evidence": q or ""}]
        for j, asp in enumerate(aspects):
            a_txt = _format_aspect_text(asp)
            user = _EVAL_PROMPT_USER.format(question=q or "", details=d or "", response=r or "", aspects=a_txt)
            pairs.append((_EVAL_PROMPT_SYSTEM, user))
            ids.append((i, j))
    return ids, pairs


def evaluator_local(queries, responses, details_list, aspects_list, llm: "LLM",
                    eval_max_new_tokens: int = 256, judge_temp: float = 0.1, args=None) -> Dict[str, Any]:
    """Evaluate using local vLLM judge."""
    tok = _load_eval_tokenizer(args) if args else None
    B_total = getattr(args, "eval_total_token_budget", 1500)
    overhead = getattr(args, "eval_overhead_tokens", 24)
    caps = {
        "q": getattr(args, "eval_q_toks", 128),
        "d": getattr(args, "eval_details_toks", 512),
        "r": getattr(args, "eval_response_toks", 512),
        "a": getattr(args, "eval_aspect_toks", 64)
    }
    
    ids, pairs0 = _create_eval_pairs_all(queries, responses, details_list, aspects_list)

    # Fit each prompt to budget
    pairs = []
    for (s, u) in pairs0:
        # Extract components from user prompt
        m_q = re.search(r"question:\s*(.*)\ndetails:", u, re.S)
        q = (m_q.group(1) if m_q else "")
        m_d = re.search(r"details:\s*(.*)\nresponse:", u, re.S)
        d = (m_d.group(1) if m_d else "")
        m_r = re.search(r"response:\s*(.*)\naspect:", u, re.S)
        r = (m_r.group(1) if m_r else "")
        m_a = re.search(r"aspect:\s*(.*)\n\nReturn only", u, re.S)
        a = (m_a.group(1) if m_a else "")

        # Truncate to caps
        q_tr = _truncate_to_ids(tok, q, caps["q"])
        d_tr = _truncate_to_ids(tok, d, caps["d"])
        r_tr = _truncate_to_ids(tok, r, caps["r"])
        a_tr = _truncate_to_ids(tok, a, caps["a"])

        # Fit to budget
        q_fit, d_fit, r_fit, a_fit, prompt_tokens = _fit_prompt_to_budget(
            tok, _EVAL_PROMPT_SYSTEM, q_tr, d_tr, r_tr, a_tr,
            budget_prompt_only=max(1, B_total - 1 - overhead),
            min_q=32, min_d=128, min_r=128, min_a=32
        )
        
        user_fit = _EVAL_PROMPT_USER.format(question=q_fit, details=d_fit, response=r_fit, aspects=a_fit)
        pairs.append((_EVAL_PROMPT_SYSTEM, user_fit))

    # Generate
    prompts = [_maybe_apply_chat_template(llm, s, u) for (s, u) in pairs]
    outs = llm.generate(prompts, SamplingParams(temperature=judge_temp, top_p=0.95, max_tokens=eval_max_new_tokens))

    # Parse scores
    import collections
    acc = collections.defaultdict(dict)
    for (qi, aj), out in zip(ids, outs):
        text = out.outputs[0].text or ""
        obj = _parse_json_lenient(text)
        
        # Fallback if judge didn't return JSON
        if "match_score" not in obj:
            q = queries[qi]
            r = responses[qi]
            qw = set(w.lower() for w in re.findall(r"\w+", q) if len(w) > 3)
            rw = set(w.lower() for w in re.findall(r"\w+", r) if len(w) > 3)
            overlap = len(qw & rw)
            toks = len(re.findall(r"\w+", r))
            obj = {"match_score": 2 if (toks >= 20 and overlap >= 3) else (1 if (toks >= 8 and overlap >= 1) else 0)}
        
        try:
            sc = int(max(0, min(2, int(round(float(obj.get('match_score', 0)))))))
        except Exception:
            sc = 0
        acc[qi][aj] = sc

    # Aggregate per question
    per_q_scores = []
    for qi in range(len(queries)):
        aspects = aspects_list[qi] or [{"aspect": "Answer relevance"}]
        total = sum(int(acc.get(qi, {}).get(aj, 0)) for aj in range(len(aspects)))
        denom = max(1, len(aspects))
        per_q_scores.append({"id": qi, "score": total / denom})
    
    return {"per_question_scores": per_q_scores}


def evaluator_remote_openai(queries, responses, details_list, aspects_list,
                            base_url: str, api_key: Optional[str], model_name: str,
                            eval_max_new_tokens=256, tok=None, eval_caps=None,
                            judge_temp: float = 0.1, args=None) -> Dict[str, Any]:
    """Evaluate using remote OpenAI-compatible API."""
    assert _HAS_OPENAI, "Install openai>=1.40.0 to use remote judge"
    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key or "EMPTY")
    caps = eval_caps or {"q": 128, "d": 512, "r": 512, "a": 64}
    B_total = getattr(args, "eval_total_token_budget", 1500)
    overhead = getattr(args, "eval_overhead_tokens", 24)

    per_q_scores = []
    for qi, (q, r, d, aspects) in enumerate(zip(queries, responses, details_list, aspects_list)):
        aspects = aspects or [{"aspect": "Answer relevance", "reason": "", "evidence": q or ""}]
        q_trunc = _truncate_to_ids(tok, q, caps.get("q"))
        d_trunc = _truncate_to_ids(tok, d, caps.get("d"))
        r_trunc = _truncate_to_ids(tok, r, caps.get("r"))
        
        total = 0
        for asp in aspects:
            a_txt = _format_aspect_text(asp)
            a_trunc = _truncate_to_ids(tok, a_txt, caps.get("a"))
            
            system_str = _EVAL_PROMPT_SYSTEM
            q_fit, d_fit, r_fit, a_fit, prompt_tokens = _fit_prompt_to_budget(
                tok, system_str, q_trunc, d_trunc, r_trunc, a_trunc,
                budget_prompt_only=max(1, B_total - 1 - overhead),
                min_q=32, min_d=128, min_r=128, min_a=32
            )
            
            allowed_gen = max(1, B_total - overhead - prompt_tokens)
            max_new = min(eval_max_new_tokens, allowed_gen)
            
            user = _EVAL_PROMPT_USER.format(
                question=q_fit, details=d_fit, response=r_fit, aspects=a_fit
            )
            
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_str},
                        {"role": "user", "content": user}
                    ],
                    temperature=judge_temp,
                    max_tokens=int(max_new),
                    response_format={"type": "json_object"},
                )
                text = (resp.choices[0].message.content or "").strip()
                obj = _parse_json_lenient(text)
                
                # Fallback
                if "match_score" not in obj:
                    qw = set(w.lower() for w in re.findall(r"\w+", q) if len(w) > 3)
                    rw = set(w.lower() for w in re.findall(r"\w+", r) if len(w) > 3)
                    overlap = len(qw & rw)
                    toks = len(re.findall(r"\w+", r))
                    obj = {"match_score": 2 if (toks >= 20 and overlap >= 3) else (1 if (toks >= 8 and overlap >= 1) else 0)}
                
                sc = int(max(0, min(2, int(round(float(obj.get('match_score', 0)))))))
            except Exception:
                sc = 0
            
            total += sc
        
        denom = max(1, len(aspects))
        per_q_scores.append({"id": qi, "score": total / denom})
    
    return {"per_question_scores": per_q_scores}


# ========= Singleton judge with dynamic VRAM =========
_JUDGE_LLM = None


def _safe_frac(free_gb: float, total_gb: float, margin: float = 0.9, floor: float = 0.02) -> float:
    """Calculate safe GPU memory fraction."""
    if total_gb <= 0:
        return max(floor, 0.02)
    frac = (free_gb / total_gb) * margin
    return max(floor, min(frac, 0.20))


def get_or_make_judge(args):
    """Get or create singleton judge LLM."""
    global _JUDGE_LLM
    if _JUDGE_LLM is not None:
        return _JUDGE_LLM
    
    import torch
    dev0 = 0
    if args.eval_devices:
        try:
            dev0 = int(str(args.eval_devices).split(",")[0])
        except Exception:
            dev0 = 0
    
    free_b, total_b = torch.cuda.mem_get_info(device=dev0)
    free_gb = free_b / (1024**3)
    total_gb = total_b / (1024**3)
    first_frac = args.eval_gpu_mem if args.eval_gpu_mem is not None else _safe_frac(free_gb, total_gb)
    
    def _mk_with(frac: float):
        return _init_llm_core(
            model=(args.eval_model or args.trl_policy_model),
            dtype=(args.eval_dtype or args.dtype or "bfloat16"),
            tensor_parallel_size=(args.eval_tensor_parallel_size or args.tensor_parallel_size or 1),
            max_model_len=(args.eval_max_model_len or args.max_model_len or 2048),
            gpu_memory_utilization=float(frac),
        )
    
    try:
        _JUDGE_LLM = _with_visible_devices(args.eval_devices, lambda: _mk_with(first_frac))
    except ValueError as e:
        if "Free memory on device" in str(e):
            free_b, total_b = torch.cuda.mem_get_info(device=dev0)
            free_gb = free_b / (1024**3)
            total_gb = total_b / (1024**3)
            frac = _safe_frac(free_gb, total_gb, margin=0.85, floor=0.02)
            _JUDGE_LLM = _with_visible_devices(args.eval_devices, lambda: _mk_with(frac))
        else:
            raise
    
    return _JUDGE_LLM


def judge_batch(args, queries: List[str], responses: List[str], 
                details: List[str], aspects: List[List[Dict[str, str]]]) -> List[float]:
    """Judge a batch of completions."""
    if args.eval_endpoint:
        tok = _load_eval_tokenizer(args)
        out = evaluator_remote_openai(
            queries, responses, details, aspects,
            base_url=args.eval_endpoint, api_key=args.eval_api_key,
            model_name=(args.eval_model or "judge"),
            eval_max_new_tokens=args.eval_max_new_tokens,
            tok=tok,
            eval_caps={
                "q": args.eval_q_toks,
                "d": args.eval_details_toks,
                "r": args.eval_response_toks,
                "a": args.eval_aspect_toks
            },
            judge_temp=args.eval_sampling_temperature,
            args=args,
        )
    else:
        llm = get_or_make_judge(args)
        out = evaluator_local(
            queries, responses, details, aspects, llm=llm,
            eval_max_new_tokens=args.eval_max_new_tokens,
            judge_temp=args.eval_sampling_temperature,
            args=args
        )
    
    return [float(item.get("score", 0.0)) for item in out.get("per_question_scores", [])]


# ============================================================================
# CLI
# ============================================================================
def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="RAG(4x)+Vanilla(1x) stacked GRPO with normalized advantage"
    )

    # Data
    ap.add_argument("--input-jsonl", type=str, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--sources", type=str, default="profile")
    ap.add_argument("--hf-config", nargs="+", default="Society_and_Culture")
    ap.add_argument("--hf-split", type=str, default="train")

    # Stacking
    ap.add_argument("--rag-per-q", type=int, default=4)
    ap.add_argument("--van-per-q", type=int, default=1)

    # Planner
    ap.add_argument("--planner-model", type=str, required=True)
    ap.add_argument("--planner-dtype", type=str, default=None)
    ap.add_argument("--planner-tensor-parallel-size", type=int, default=None)
    ap.add_argument("--planner-max-model-len", type=int, default=None)
    ap.add_argument("--planner-gpu-mem", type=float, default=None)
    ap.add_argument("--planner-devices", type=str, default=None)
    ap.add_argument("--topk", type=int, default=2)

    # Policy / TRL
    ap.add_argument("--trl-train", action="store_true")
    ap.add_argument("--trl-policy-model", type=str, required=True)
    ap.add_argument("--trl-save-dir", type=str, default="./trl_out")
    ap.add_argument("--trl-batch-size", type=int, default=1)
    ap.add_argument("--trl-grad-accum", type=int, default=1)
    ap.add_argument("--trl-n-generations", type=int, default=2)
    ap.add_argument("--trl-max-new-tokens", type=int, default=96)

    # Sampling
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)

    # Compute
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.78)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--train-max-len", type=int, default=2048)
    ap.add_argument("--optim", type=str, default="adafactor")

    # Judge
    ap.add_argument("--eval-endpoint", type=str, default=None)
    ap.add_argument("--eval-api-key", type=str, default=None)
    ap.add_argument("--eval-model", type=str, default=None)
    ap.add_argument("--eval-dtype", type=str, default=None)
    ap.add_argument("--eval-tensor-parallel-size", type=int, default=None)
    ap.add_argument("--eval-max-model-len", type=int, default=1200)
    ap.add_argument("--eval-gpu-mem", type=float, default=None)
    ap.add_argument("--eval-devices", type=str, default=None)

    # Judge caps
    ap.add_argument("--eval-q-toks", type=int, default=128)
    ap.add_argument("--eval-details-toks", type=int, default=512)
    ap.add_argument("--eval-response-toks", type=int, default=512)
    ap.add_argument("--eval-aspect-toks", type=int, default=64)
    ap.add_argument("--eval-tokenizer", type=str, default=None)
    ap.add_argument("--eval-max-new-tokens", type=int, default=64)
    ap.add_argument("--eval-sampling-temperature", type=float, default=0.1)
    ap.add_argument("--eval-total-token-budget", type=int, default=1200)
    ap.add_argument("--eval-overhead-tokens", type=int, default=24)

    # Reward tuning
    ap.add_argument("--k-shift", type=float, default=2.0)
    ap.add_argument("--zero-out-vanilla", type=bool, default=True)
    ap.add_argument("--van-penalty", type=float, default=0.0)
    ap.add_argument("--struct-penalty", type=float, default=0.5)

    # Checkpointing
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--save-total-limit", type=int, default=4)
    ap.add_argument("--genlog-jsonl", type=str, default=None)
    ap.add_argument("--scorelog-jsonl", type=str, default=None)

    return ap


def main():
    args = build_arg_parser().parse_args()
    try:
        if not args.trl_train:
            raise AssertionError("Pass --trl-train to start training")
        
        # Prepare prompts
        prepared = prepare_stacked_prompts(args)
        
        # Train
        train_trl_grpo_stacked(args, prepared, judge_batch, safe_shutdown)
        
    finally:
        safe_shutdown()


if __name__ == "__main__":
    main()
