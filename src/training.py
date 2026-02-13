#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training.py - TRL GRPO training
"""

import os, sys, math, time, json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import torch

# ---------- Optional deps ----------
_HAS_TRL = True
try:
    from trl import GRPOTrainer, GRPOConfig
except Exception:
    _HAS_TRL = False

_HAS_TRANSFORMERS = True
try:
    from transformers import AutoTokenizer
except Exception:
    _HAS_TRANSFORMERS = False

_HAS_DATASETS = True
try:
    from datasets import Dataset as HFDataset
except Exception:
    _HAS_DATASETS = False


# Import from other modules
from retrieval import get_tag


# ============================================================================
# Utility functions
# ============================================================================
def _world_size():
    """Get distributed world size."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return max(1, dist.get_world_size())
    except Exception:
        pass
    try:
        return max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except Exception:
        return 1


def _compute_generation_batch_size(per_device_bs: int, grad_accum: int, num_generations: int) -> int:
    """Compute generation batch size that's divisible by both global batch and num_generations."""
    global_bs = max(1, per_device_bs) * max(1, _world_size()) * max(1, grad_accum)
    return math.lcm(global_bs, max(1, num_generations))


def _append_jsonl(path, obj):
    """Append a JSON object to a JSONL file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ============================================================================
# Dataset preparation for TRL
# ============================================================================
def make_trl_dataset_from_prepared(prepared: List[Dict[str, Any]]):
    """Convert prepared prompts to TRL-compatible dataset."""
    if _HAS_DATASETS:
        return HFDataset.from_list([
            {k: rec[k] for k in ("prompt", "details", "aspects")} 
            for rec in prepared
        ])
    return [
        {"prompt": rec["prompt"], "details": rec["details"], "aspects": rec["aspects"]} 
        for rec in prepared
    ]


# ============================================================================
# Reward function factory
# ============================================================================
def make_reward_function(args, meta: Dict[str, Dict], judge_batch_fn):
    """
    Create reward function for GRPO training.
    
    The reward function:
    1. Groups completions by question (group_id)
    2. Judges each completion using the judge model
    3. Computes normalized advantages (RAG vs vanilla baseline)
    4. Applies structural penalty if <think> doesn't come before <answer>
    
    Args:
        args: Command-line arguments with hyperparameters
        meta: Dict mapping prompts to their metadata (group_id, mode, etc.)
        judge_batch_fn: Function to score batches of completions
    
    Returns:
        reward_func: Callable for TRL trainer
    """
    
    def reward_func(*, prompts=None, completions=None, **kw):
        """
        Reward for stacked RAG(K) + Vanilla(1) per question with structural penalty.
        
        Scoring:
        - Groups completions by question
        - Computes advantage of each RAG answer vs vanilla baseline
        - Penalizes vanilla answers
        - Penalizes missing structure (<think> before <answer>)
        """
        # ---------------- Hyperparameters ----------------
        EPS = 1e-6
        K_SHIFT = float(getattr(args, "k_shift", 2.0))
        ZERO_OUT_VANILLA = bool(getattr(args, "zero_out_vanilla", True))
        VANILLA_PENALTY = float(getattr(args, "van_penalty", 0.0))
        K_RAG = int(getattr(args, "rag_per_q", 4))
        STRUCT_PENALTY = float(getattr(args, "struct_penalty", 0.5))

        # ------------- One-time initialization -------------
        if not hasattr(reward_func, "_inited"):
            outdir = getattr(args, "trl_save_dir", None) or "./trl_out"
            os.makedirs(outdir, exist_ok=True)
            reward_func._scorelog_path = str(Path(outdir) / "judge_scores.jsonl") \
                if not getattr(args, "scorelog_jsonl", None) else args.scorelog_jsonl
            reward_func._genlog_path = str(Path(outdir) / "reward_traces.jsonl") \
                if not getattr(args, "genlog_jsonl", None) else args.genlog_jsonl
            reward_func._counter = 0
            reward_func._inited = True

        scorelog_path_s = reward_func._scorelog_path
        genlog_path_s = reward_func._genlog_path

        # ------------- Normalize TRL inputs -------------
        comps = completions if completions is not None else (
            kw.get("completions") or kw.get("responses") or kw.get("outputs") or []
        )

        def _as_text(c):
            """Convert completion to text string."""
            if isinstance(c, str):
                return c
            if isinstance(c, list) and c and isinstance(c[0], dict) and "content" in c[0]:
                asst = [m.get("content", "") for m in c if isinstance(m, dict) and m.get("role") == "assistant"]
                return "".join(asst) if asst else "".join(m.get("content", "") for m in c if isinstance(m, dict))
            return str(c)

        # Build flat lists + robust modes/gids
        q_list, d_list, a_list, modes, gids = [], [], [], [], []

        # Handle batched completions (B x G format)
        if comps and isinstance(comps[0], list):
            B = len(comps)
            G = len(comps[0]) if B > 0 else 1
            flat = []
            base_prompts = (prompts or [""] * B)
            
            for b in range(B):
                p = base_prompts[b] if b < len(base_prompts) else ""
                m = meta.get(p)
                gid_b = (m.get("group_id") if (m and isinstance(m, dict) and "group_id" in m) else f"gid::{b}")
                
                for j in range(G):
                    c = comps[b][j]
                    flat.append(_as_text(c))
                    
                    if m is not None and isinstance(m, dict):
                        q = m.get("question", p)
                        d = m.get("details", "")
                        a = m.get("aspects", [{"aspect": "Answer relevance", "reason": "", "evidence": p}])
                        mode = m.get("mode", ("rag" if j < K_RAG else "van"))
                    else:
                        q, d = p, ""
                        a = [{"aspect": "Answer relevance", "reason": "", "evidence": p}]
                        mode = "rag" if j < K_RAG else "van"
                    
                    q_list.append(q)
                    d_list.append(d)
                    a_list.append(a)
                    modes.append(mode)
                    gids.append(gid_b)
            
            expanded_prompts = [base_prompts[b] if b < len(base_prompts) else "" for b in range(B) for _ in range(G)]
        else:
            # Flat completions
            flat = [_as_text(c) for c in (comps or [])]
            base_prompts = (prompts or [""] * len(flat))
            expanded_prompts = base_prompts[:]
            
            for i, p in enumerate(expanded_prompts):
                m = meta.get(p)
                
                if m is not None and isinstance(m, dict):
                    q = m.get("question", p)
                    d = m.get("details", "")
                    a = m.get("aspects", [{"aspect": "Answer relevance", "reason": "", "evidence": p}])
                    mode = m.get("mode", "rag")
                    gid = m.get("group_id", f"gid::flat::{i}")
                else:
                    q, d = p, ""
                    a = [{"aspect": "Answer relevance", "reason": "", "evidence": p}]
                    mode = "rag"
                    gid = f"gid::flat::{i}"
                
                q_list.append(q)
                d_list.append(d)
                a_list.append(a)
                modes.append(mode)
                gids.append(gid)
            
            B, G = len(expanded_prompts), 1

        # -------- Structural check (think-before-answer) --------
        has_think_then_answer = []
        for txt in flat:
            t = get_tag(txt, "think")
            a = get_tag(txt, "answer")
            ok = bool(t) and bool(a) and (txt.find("<think") < txt.find("<answer"))
            has_think_then_answer.append(ok)

        # -------- Judge ONLY the <answer> part --------
        flat_answers = []
        for txt in flat:
            a_txt = get_tag(txt, "answer")
            flat_answers.append(a_txt if (a_txt and a_txt.strip()) else txt)

        # Call judge to score all completions
        s_all = judge_batch_fn(args, q_list, flat_answers, d_list, a_list)

        # Save per-call judge scores
        _append_jsonl(scorelog_path_s, {"ts": time.time(), "call_idx": reward_func._counter, "scores": s_all})

        # ------------- Group indices -------------
        idxs_by_gid = defaultdict(list)
        for i, gid in enumerate(gids):
            idxs_by_gid[gid].append(i)

        rewards = [0.0] * len(flat)

        # ------------- Per-group computation -------------
        for gid, idxs in idxs_by_gid.items():
            rag_idx = [i for i in idxs if modes[i] == "rag"]
            van_idx = [i for i in idxs if modes[i] == "van"]

            rag_scores = [s_all[i] for i in rag_idx]
            van_scores = [s_all[i] for i in van_idx]
            r_v = (sum(van_scores) / len(van_scores)) if van_scores else 0.0

            # Compute statistics over RAG only
            if len(rag_scores) >= 2:
                mu_rag = sum(rag_scores) / len(rag_scores)
                var_rag = sum((x - mu_rag) ** 2 for x in rag_scores) / max(1, len(rag_scores) - 1)
                std_rag = math.sqrt(var_rag) if var_rag > 0 else 1.0
            elif len(rag_scores) == 1:
                mu_rag, std_rag = rag_scores[0], 1.0
            else:
                mu_rag, std_rag = 0.0, 1.0
            std_rag = max(std_rag, EPS)

            # Compute statistics over RAG + vanilla (all)
            all_scores = rag_scores + [r_v]
            if len(all_scores) >= 2:
                mu_all = sum(all_scores) / len(all_scores)
                var_all = sum((x - mu_all) ** 2 for x in all_scores) / max(1, len(all_scores) - 1)
                std_all = math.sqrt(var_all) if var_all > 0 else 1.0
            else:
                mu_all = all_scores[0] if all_scores else 0.0
                std_all = 1.0
            std_all = max(std_all, EPS)

            # Compute reward for each completion
            for i in idxs:
                s_i = s_all[i]

                if modes[i] == "van":
                    # Vanilla: either zero out or penalize
                    if ZERO_OUT_VANILLA:
                        r_i = 0.0
                    else:
                        base = abs(((-mu_all) / std_all) - (K_SHIFT * r_v / std_rag))
                        r_i = -VANILLA_PENALTY * float(base)
                else:
                    # RAG: normalized advantage
                    A_all_i = (s_i - r_v - mu_all) / std_all
                    hatA_i = A_all_i - (K_SHIFT * r_v / std_rag)
                    r_i = float(hatA_i)

                # Apply structural penalty if <think> not before <answer>
                if not has_think_then_answer[i]:
                    r_i -= STRUCT_PENALTY

                rewards[i] = r_i

                # Log trace
                _append_jsonl(genlog_path_s, {
                    "ts": time.time(),
                    "call_idx": reward_func._counter,
                    "i": i, "gid": gid, "mode": modes[i],
                    "judge_score": s_i, "reward": r_i,
                    "struct_ok": bool(has_think_then_answer[i]),
                    "struct_penalty": (STRUCT_PENALTY if not has_think_then_answer[i] else 0.0),
                    "r_v": r_v, "mu_rag": mu_rag, "std_rag": std_rag,
                    "mu_all": mu_all, "std_all": std_all,
                    "prompt": expanded_prompts[i],
                    "completion": flat[i],
                    "answer_only": flat_answers[i],
                })

        # Increment call index
        reward_func._counter += 1

        # Return rewards as tensor
        t = torch.tensor(rewards, dtype=torch.float32)
        if comps and isinstance(comps[0], list):
            t = t.view(B, G)
        return t

    return reward_func


# ============================================================================
# Main training function
# ============================================================================
def train_trl_grpo_stacked(args, prepared: List[Dict[str, Any]], judge_batch_fn, safe_shutdown_fn):
    """
    Train policy model using GRPO with stacked RAG + vanilla prompts.
    
    Args:
        args: Command-line arguments
        prepared: List of prepared prompts with metadata
        judge_batch_fn: Function to score completions
        safe_shutdown_fn: Function to clean up vLLM engines
    """
    assert _HAS_TRL, "Install trl>=0.9 to run GRPO training"

    # Create TRL dataset
    ds = make_trl_dataset_from_prepared(prepared)

    print("[MEM] Shutting down planner LLMs before loading policy...", file=sys.stderr)
    safe_shutdown_fn()

    # Build prompt -> meta mapping for reward function
    meta = {
        rec["prompt"]: {k: rec[k] for k in ("group_id", "mode", "question", "details", "aspects")} 
        for rec in prepared
    }

    # Create reward function
    reward_func = make_reward_function(args, meta, judge_batch_fn)

    # Compute generation batch size
    gen_batch = _compute_generation_batch_size(
        per_device_bs=args.trl_batch_size,
        grad_accum=args.trl_grad_accum,
        num_generations=args.trl_n_generations,
    )

    # Training configuration
    training_args = GRPOConfig(
        output_dir=args.trl_save_dir or "./trl_out",
        max_steps=200,
        num_train_epochs=1,
        per_device_train_batch_size=args.trl_batch_size,
        gradient_accumulation_steps=args.trl_grad_accum,
        logging_steps=args.log_every,
        learning_rate=args.lr,
        bf16=(args.dtype == "bfloat16"),
        fp16=(args.dtype == "float16"),
        remove_unused_columns=False,
        max_prompt_length=max(64, (args.train_max_len // 2) if args.train_max_len else 2048),
        max_completion_length=args.trl_max_new_tokens,
        num_generations=args.trl_n_generations,
        generation_batch_size=gen_batch,
        temperature=args.temperature,
        top_p=args.top_p,
        generation_kwargs={"do_sample": True},
        optim=args.optim,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],  # Silence W&B etc.
        # Checkpointing
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        logging_dir=args.trl_save_dir or "./trl_out",
    )

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(args.trl_policy_model, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Create trainer
    trainer = GRPOTrainer(
        model=args.trl_policy_model,
        args=training_args,
        train_dataset=ds,
        reward_funcs=[reward_func],
        processing_class=tok,
    )

    # Train
    trainer.train()
    
    # Save final model
    if args.trl_save_dir:
        try:
            trainer.model.save_pretrained(args.trl_save_dir)
            print(f"[TRL] Saved policy to {args.trl_save_dir}")
        except Exception as e:
            print(f"[TRL] Failed to save model: {e}", file=sys.stderr)
