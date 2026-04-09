#!/usr/bin/env python3
"""
Varentropy measurement for Gemma 4 E4B-IT on 969 DS Critique Bank items.

Adapted from varentropy_test.py (Qwen) for mlx_vlm (Gemma 4).
Only runs Phase 1 (initial answer) with H and V capture at the answer token.
Behavioral data (Phase 2 critique response) already exists in trials_Gemma4-E4B-IT.json.
"""

import json
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load as vlm_load
from mlx_vlm.generate import generate_step as vlm_generate_step

RESULTS_DIR = Path("results/varentropy")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRIALS_PATH = Path("results/sweep_gemma4/trials_Gemma4-E4B-IT.json")
MAX_TOKENS = 64


def generate_with_varentropy(model, tokenizer, messages, answer_options):
    """Generate response, capturing H and V at the answer token position."""
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    if "<|think|>" in prompt or "<think>" in prompt:
        if "</think>" not in prompt:
            prompt = prompt + "</think>\n\n"

    prompt_tokens = mx.array(tokenizer.encode(prompt)).reshape(1, -1)

    option_token_ids = {}
    for opt in answer_options:
        for variant in [opt, f" {opt}", f"({opt}"]:
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if toks:
                option_token_ids[toks[-1]] = opt

    tokens = []
    logprobs = []
    entropies = []
    varentropies = []
    answer_token_data = None

    for step_out, _ in zip(
        vlm_generate_step(
            prompt_tokens, model, None, None, max_tokens=MAX_TOKENS
        ),
        range(MAX_TOKENS),
    ):
        tok, logits = step_out
        tok_id = int(tok)
        if tok_id == tokenizer.eos_token_id:
            break
        if tok_id in (106, 1):
            break
        tokens.append(tok_id)

        probs = mx.softmax(logits.reshape(-1), axis=-1)
        log_p = mx.log(probs + 1e-10)
        neg_log_p = -log_p

        H = float(mx.sum(probs * neg_log_p))
        V = float(mx.sum(probs * neg_log_p * neg_log_p)) - H * H

        logprobs.append(float(log_p[tok_id]))
        entropies.append(H)
        varentropies.append(V)

        if answer_token_data is None and tok_id in option_token_ids:
            answer_token_data = {
                "answer": option_token_ids[tok_id],
                "step": len(tokens) - 1,
                "H": H,
                "V": V,
                "logprob": float(log_p[tok_id]),
            }

    text = tokenizer.decode(tokens)

    if answer_token_data is None:
        answer = extract_answer(text, answer_options)
        if answer:
            answer_token_data = {
                "answer": answer, "step": -1,
                "H": None, "V": None, "logprob": None,
            }

    return {
        "text": text,
        "mean_logprob": float(np.mean(logprobs)) if logprobs else 0.0,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mean_varentropy": float(np.mean(varentropies)) if varentropies else 0.0,
        "answer_token": answer_token_data,
        "n_tokens": len(tokens),
    }


def extract_answer(response, valid_labels):
    response = response.strip().upper()
    for label in valid_labels:
        if f"({label})" in response or f"ANSWER IS {label}" in response:
            return label
    for char in response:
        if char in valid_labels:
            return char
    return None


def main():
    print("=" * 60)
    print("VARENTROPY: Gemma 4 E4B-IT")
    print("=" * 60)

    # Load existing trials to get the question set
    print("Loading existing trials...")
    with open(TRIALS_PATH) as f:
        existing_trials = json.load(f)
    print(f"  {len(existing_trials)} trials")

    # We need the original items to regenerate Phase 1.
    # Load DS Critique Bank to get questions.
    from datasets import load_dataset
    ds = load_dataset("allenai/DS_Critique_Bank", split="train")

    # Build qid -> question mapping
    qid_to_question = {}
    for item in ds:
        qid = item['qid']
        if qid not in qid_to_question:
            qid_to_question[qid] = item['question']

    print(f"  {len(qid_to_question)} unique questions in DS Critique Bank")

    # Load model
    print("\nLoading Gemma 4 E4B-IT...")
    t_load = time.time()
    model, processor = vlm_load("google/gemma-4-E4B-it")
    tokenizer = processor.tokenizer
    print(f"  Loaded in {time.time() - t_load:.1f}s")

    results = []
    n_matched = 0
    n_fallback = 0
    t0 = time.time()

    for i, trial in enumerate(existing_trials):
        qid = trial['qid']
        question = qid_to_question.get(qid)
        if question is None:
            continue

        labels = re.findall(r'\(([A-Z])\)', question)
        if not labels:
            labels = ['A', 'B', 'C', 'D']

        messages = [
            {"role": "user", "content": question + "\n\nAnswer with just the letter."}
        ]
        gen = generate_with_varentropy(model, tokenizer, messages, labels)

        if gen["answer_token"] is None:
            results.append({
                "idx": i, "qid": qid,
                "answer_H": None, "answer_V": None,
                "answer_logprob": None, "matched_original": False,
            })
            continue

        matched = gen["answer_token"]["answer"] == trial["initial_answer"]
        if gen["answer_token"]["step"] == -1:
            n_fallback += 1

        results.append({
            "idx": i,
            "qid": qid,
            "dataset": trial["dataset"],
            "model_correct": trial["model_correct"],
            "initial_answer": trial["initial_answer"],
            "regenerated_answer": gen["answer_token"]["answer"],
            "matched_original": matched,
            "sdt": trial["sdt"],
            "did_revise": trial["did_revise"],
            "answer_H": gen["answer_token"]["H"],
            "answer_V": gen["answer_token"]["V"],
            "answer_logprob": gen["answer_token"]["logprob"],
            "mean_entropy": gen["mean_entropy"],
            "mean_varentropy": gen["mean_varentropy"],
        })

        if matched:
            n_matched += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            match_rate = n_matched / (i + 1)
            print(f"  [{i+1}/{len(existing_trials)}] {elapsed:.1f}s  "
                  f"match_rate={match_rate:.1%}  fallback={n_fallback}")

    elapsed = time.time() - t0
    n_valid = sum(1 for r in results if r["answer_H"] is not None)
    print(f"\n  Done: {n_valid} valid of {len(results)} in {elapsed:.1f}s")
    print(f"  Match rate with original: {n_matched}/{len(results)} = {n_matched/len(results)*100:.1f}%")

    # Save
    out_path = RESULTS_DIR / "varentropy_E4B.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")

    # Quick summary
    valid = [r for r in results if r["answer_H"] is not None and r["matched_original"]]
    if valid:
        Hs = np.array([r["answer_H"] for r in valid])
        Vs = np.array([r["answer_V"] for r in valid])
        print(f"\n  Matched trials: {len(valid)}")
        print(f"  Mean H: {np.mean(Hs):.3f}  Mean V: {np.mean(Vs):.3f}")
        print(f"  H-V correlation: {np.corrcoef(Hs, Vs)[0,1]:.3f}")

        # FA rate split by varentropy (correct trials only)
        correct = [r for r in valid if r["model_correct"]]
        if correct:
            v_median = np.median([r["answer_V"] for r in correct])
            high_v = [r for r in correct if r["answer_V"] >= v_median]
            low_v = [r for r in correct if r["answer_V"] < v_median]
            fa_high = sum(1 for r in high_v if r["sdt"] == "FALSE_ALARM") / len(high_v) if high_v else 0
            fa_low = sum(1 for r in low_v if r["sdt"] == "FALSE_ALARM") / len(low_v) if low_v else 0
            print(f"\n  FA rate (correct trials, median V split):")
            print(f"    High V: {fa_high:.3f}  Low V: {fa_low:.3f}  Gap: {fa_low - fa_high:+.3f}")
            print(f"    V median: {v_median:.3f}")
            if fa_low > fa_high:
                print(f"    -> V is PROTECTIVE (high V = lower FA)")
            else:
                print(f"    -> V is a RISK FACTOR (high V = higher FA)")

    del model, processor
    mx.clear_cache()


if __name__ == "__main__":
    main()
