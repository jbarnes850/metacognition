#!/usr/bin/env python3
"""
Test whether varentropy adds predictive signal for revision behavior.

For each model (0.8B, 2B, 4B, 9B) on 150 ARC-Challenge items:
1. Extract logit vector at the answer token position before critique
2. Compute entropy H and varentropy V from the softmax distribution
3. Test whether V predicts revision behavior above and beyond H alone
4. Track how the V effect scales across model sizes
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate_step
from scipy.stats import norm, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

MODELS = [
    ("0.8B", "models/Qwen3.5-0.8B"),
    ("2B", "models/Qwen3.5-2B"),
    ("4B", "models/Qwen3.5-4B"),
    ("9B", "models/Qwen3.5-9B"),
]
REGISTRY_PATH = "registry/v1_items.json"
RESULTS_DIR = Path("results/varentropy")
MAX_TOKENS = 64


# ---------------------------------------------------------------------------
# Generation with varentropy capture
# ---------------------------------------------------------------------------

def generate_with_varentropy(model, tokenizer, messages, answer_options):
    """Generate response, capturing H and V at the answer token position."""
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    if "<think>\n" in prompt:
        prompt = prompt.replace(
            "<|im_start|>assistant\n<think>\n",
            "<|im_start|>assistant\n<think>\n</think>\n",
        )
    prompt_tokens = mx.array(tokenizer.encode(prompt))

    # Build set of token IDs that map to answer options
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
        generate_step(prompt_tokens, model, max_tokens=MAX_TOKENS),
        range(MAX_TOKENS),
    ):
        tok, logits = step_out
        tok_id = int(tok)
        if tok_id == tokenizer.eos_token_id:
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
    """Extract MC answer from response."""
    response = response.strip().upper()
    for label in valid_labels:
        if f"({label})" in response or f"ANSWER IS {label}" in response:
            return label
    for char in response:
        if char in valid_labels:
            return char
    return None


# ---------------------------------------------------------------------------
# Per-model experiment
# ---------------------------------------------------------------------------

def run_model(model_name, model_path, items):
    """Run the varentropy experiment for one model."""
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")

    model, tokenizer = load(model_path)
    trials = []
    n_fallback = 0
    t0 = time.time()

    for idx, item in enumerate(items):
        qid = item["item_id"]
        question = item["question"]
        gold = item["gold_answer"]
        options = item["answer_options"]

        # Phase 1: Initial answer
        messages = [
            {"role": "user", "content": question + "\n\nAnswer with just the letter."},
        ]
        gen1 = generate_with_varentropy(model, tokenizer, messages, options)

        if gen1["answer_token"] is None:
            continue
        if gen1["answer_token"]["step"] == -1:
            n_fallback += 1

        initial_answer = gen1["answer_token"]["answer"]
        model_correct = initial_answer == gold

        # Phase 2: Present critique
        critique_type = "invalid" if model_correct else "valid"
        critique_text = None
        for c in item["critiques"]:
            if c["type"] == critique_type and c["answer_mode"] == "aware":
                critique_text = c["text"]
                break
        if critique_text is None:
            continue

        messages2 = [
            {"role": "user", "content": question + "\n\nAnswer with just the letter."},
            {"role": "assistant", "content": f"({initial_answer})"},
            {"role": "user", "content": critique_text
                + "\n\nWhat is the correct answer? Just the letter."},
        ]
        gen2 = generate_with_varentropy(model, tokenizer, messages2, options)

        revised_answer = extract_answer(gen2["text"], options)
        did_revise = revised_answer is not None and revised_answer != initial_answer

        if model_correct:
            sdt = "FALSE_ALARM" if did_revise else "CORRECT_REJECTION"
        else:
            sdt = "HIT" if did_revise else "MISS"

        trials.append({
            "idx": idx,
            "qid": qid,
            "model_correct": model_correct,
            "initial_answer": initial_answer,
            "gold": gold,
            "critique_type": critique_type,
            "did_revise": did_revise,
            "revised_answer": revised_answer,
            "sdt": sdt,
            "answer_H": gen1["answer_token"]["H"],
            "answer_V": gen1["answer_token"]["V"],
            "answer_logprob": gen1["answer_token"]["logprob"],
            "mean_entropy": gen1["mean_entropy"],
            "mean_varentropy": gen1["mean_varentropy"],
        })

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            n_c = sum(1 for t in trials if t["model_correct"])
            print(f"  [{idx+1}/{len(items)}] {elapsed:.1f}s  "
                  f"acc={n_c}/{len(trials)}  fallback={n_fallback}")

    elapsed = time.time() - t0
    print(f"  Done: {len(trials)} trials in {elapsed:.1f}s")

    del model, tokenizer
    mx.clear_cache()

    return trials


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def analyze_model(model_name, trials):
    """Analyze one model's results. Returns summary dict."""
    valid = [t for t in trials if t["answer_H"] is not None]
    if len(valid) < 20:
        print(f"  {model_name}: too few valid trials ({len(valid)})")
        return None

    # SDT
    sdt_counts = {}
    for t in valid:
        sdt_counts[t["sdt"]] = sdt_counts.get(t["sdt"], 0) + 1

    hits = sdt_counts.get("HIT", 0)
    misses = sdt_counts.get("MISS", 0)
    fas = sdt_counts.get("FALSE_ALARM", 0)
    crs = sdt_counts.get("CORRECT_REJECTION", 0)
    sig_n = hits + misses
    noise_n = fas + crs

    hr = np.clip((hits + 0.5) / (sig_n + 1), 0.001, 0.999) if sig_n > 0 else 0.5
    far = np.clip((fas + 0.5) / (noise_n + 1), 0.001, 0.999) if noise_n > 0 else 0.5
    dp = norm.ppf(hr) - norm.ppf(far)

    accuracy = sum(1 for t in valid if t["model_correct"]) / len(valid)

    # H-V correlation
    Hs = np.array([t["answer_H"] for t in valid])
    Vs = np.array([t["answer_V"] for t in valid])
    r_hv, p_hv = pearsonr(Hs, Vs)

    # Logistic regression AUCs (5-fold CV)
    y = np.array([int(t["did_revise"]) for t in valid])
    H_arr = Hs.reshape(-1, 1)
    V_arr = Vs.reshape(-1, 1)
    HV_arr = np.column_stack([Hs, Vs])

    aucs = {"H": [], "V": [], "H+V": []}
    if len(np.unique(y)) == 2:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(H_arr, y):
            y_tr, y_te = y[train_idx], y[test_idx]
            if len(np.unique(y_te)) < 2:
                continue
            for name, X in [("H", H_arr), ("V", V_arr), ("H+V", HV_arr)]:
                clf = LogisticRegression(random_state=42).fit(X[train_idx], y_tr)
                pred = clf.predict_proba(X[test_idx])[:, 1]
                aucs[name].append(roc_auc_score(y_te, pred))

    auc_H = float(np.mean(aucs["H"])) if aucs["H"] else None
    auc_V = float(np.mean(aucs["V"])) if aucs["V"] else None
    auc_HV = float(np.mean(aucs["H+V"])) if aucs["H+V"] else None
    delta = (auc_HV - auc_H) if (auc_HV and auc_H) else None

    # False alarm analysis (correct-answer trials)
    correct_trials = [t for t in valid if t["model_correct"]]
    v_coef_fa = None
    fa_rate_high_v = None
    fa_rate_low_v = None
    v_median = None

    if len(correct_trials) > 10:
        y_fa = np.array([int(t["sdt"] == "FALSE_ALARM") for t in correct_trials])
        if len(np.unique(y_fa)) == 2 and sum(y_fa) >= 3:
            H_fa = np.array([t["answer_H"] for t in correct_trials]).reshape(-1, 1)
            V_fa = np.array([t["answer_V"] for t in correct_trials]).reshape(-1, 1)
            HV_fa = np.column_stack([H_fa.ravel(), V_fa.ravel()])

            clf_hv = LogisticRegression(random_state=42).fit(HV_fa, y_fa)
            v_coef_fa = float(clf_hv.coef_[0][1])

            v_median = float(np.median(V_fa))
            high_v = [t for t in correct_trials if t["answer_V"] >= v_median]
            low_v = [t for t in correct_trials if t["answer_V"] < v_median]
            if high_v and low_v:
                fa_rate_high_v = sum(
                    1 for t in high_v if t["sdt"] == "FALSE_ALARM"
                ) / len(high_v)
                fa_rate_low_v = sum(
                    1 for t in low_v if t["sdt"] == "FALSE_ALARM"
                ) / len(low_v)

    # Mean H and V
    mean_H = float(np.mean(Hs))
    mean_V = float(np.mean(Vs))

    summary = {
        "model": model_name,
        "n_trials": len(valid),
        "accuracy": accuracy,
        "d_prime": dp,
        "hit_rate": hr,
        "false_alarm_rate": far,
        "hits": hits, "misses": misses,
        "false_alarms": fas, "correct_rejections": crs,
        "mean_H": mean_H,
        "mean_V": mean_V,
        "H_V_correlation": r_hv,
        "H_V_corr_p": p_hv,
        "auc_H": auc_H,
        "auc_V": auc_V,
        "auc_HV": auc_HV,
        "auc_delta": delta,
        "v_coef_fa": v_coef_fa,
        "v_median": v_median,
        "fa_rate_high_v": fa_rate_high_v,
        "fa_rate_low_v": fa_rate_low_v,
    }

    # Print per-model summary
    print(f"\n  {model_name}: acc={accuracy:.1%}  d'={dp:.3f}  "
          f"HR={hr:.3f}  FAR={far:.3f}")
    print(f"    H-V corr: r={r_hv:.3f} (p={p_hv:.4f})")
    if auc_H and auc_HV:
        print(f"    AUC: H={auc_H:.3f}  V={auc_V:.3f}  H+V={auc_HV:.3f}  "
              f"delta={delta:+.3f}")
    if v_coef_fa is not None:
        print(f"    V coef (FA model): {v_coef_fa:+.4f}  "
              f"({'protective' if v_coef_fa < 0 else 'risk factor'})")
    if fa_rate_high_v is not None:
        print(f"    FA rate: high-V={fa_rate_high_v:.0%}  low-V={fa_rate_low_v:.0%}")

    return summary


# ---------------------------------------------------------------------------
# Cross-model comparative analysis
# ---------------------------------------------------------------------------

def comparative_analysis(all_summaries):
    """Full cross-model analysis."""
    print("\n" + "=" * 70)
    print("CROSS-MODEL VARENTROPY ANALYSIS")
    print("=" * 70)

    # Table 1: Core metrics
    print("\n--- Table 1: SDT + Varentropy metrics ---")
    header = ("  Model    Acc    d'      HR    FAR    "
              "H      V    r(H,V)  AUC_H  AUC_V  AUC_HV  delta")
    print(header)
    print("  " + "-" * 80)
    for s in all_summaries:
        auc_h = f"{s['auc_H']:.3f}" if s['auc_H'] else "  N/A"
        auc_v = f"{s['auc_V']:.3f}" if s['auc_V'] else "  N/A"
        auc_hv = f"{s['auc_HV']:.3f}" if s['auc_HV'] else "   N/A"
        delta = f"{s['auc_delta']:+.3f}" if s['auc_delta'] else "   N/A"
        print(f"  {s['model']:>5s}  {s['accuracy']:5.1%}  {s['d_prime']:6.3f}  "
              f"{s['hit_rate']:5.3f}  {s['false_alarm_rate']:5.3f}  "
              f"{s['mean_H']:5.3f}  {s['mean_V']:5.3f}  {s['H_V_correlation']:7.3f}  "
              f"{auc_h}  {auc_v}  {auc_hv}  {delta}")

    # Table 2: V coefficient and FA rate split
    print("\n--- Table 2: Varentropy and false alarm vulnerability ---")
    print(f"  {'Model':>5s}  {'V coef':>8s}  {'Direction':>12s}  "
          f"{'FA high-V':>10s}  {'FA low-V':>10s}  {'Spread':>7s}")
    print("  " + "-" * 60)
    for s in all_summaries:
        if s['v_coef_fa'] is not None:
            direction = "protective" if s['v_coef_fa'] < 0 else "risk"
            spread = ""
            if s['fa_rate_high_v'] is not None and s['fa_rate_low_v'] is not None:
                spread = f"{s['fa_rate_low_v'] - s['fa_rate_high_v']:+.0%}"
            print(f"  {s['model']:>5s}  {s['v_coef_fa']:+8.4f}  {direction:>12s}  "
                  f"{s['fa_rate_high_v']:10.0%}  {s['fa_rate_low_v']:10.0%}  "
                  f"{spread:>7s}")
        else:
            print(f"  {s['model']:>5s}  {'N/A':>8s}")

    # Scaling trends
    print("\n--- Scaling trends ---")

    models = [s['model'] for s in all_summaries]
    d_primes = [s['d_prime'] for s in all_summaries]
    deltas = [s['auc_delta'] for s in all_summaries if s['auc_delta'] is not None]
    correlations = [s['H_V_correlation'] for s in all_summaries]
    v_coefs = [s['v_coef_fa'] for s in all_summaries if s['v_coef_fa'] is not None]
    mean_Vs = [s['mean_V'] for s in all_summaries]

    print(f"  d-prime scaling:     {' -> '.join(f'{d:.3f}' for d in d_primes)}")
    if deltas:
        print(f"  AUC delta scaling:   {' -> '.join(f'{d:+.3f}' for d in deltas)}")
    print(f"  H-V corr scaling:    {' -> '.join(f'{r:.3f}' for r in correlations)}")
    if v_coefs:
        print(f"  V coef (FA) scaling: {' -> '.join(f'{c:+.4f}' for c in v_coefs)}")
    print(f"  Mean V scaling:      {' -> '.join(f'{v:.3f}' for v in mean_Vs)}")

    # Key questions
    print("\n--- Key questions ---")

    # Q1: Does V's protective effect strengthen with scale?
    if len(v_coefs) >= 2:
        if all(c < 0 for c in v_coefs):
            if abs(v_coefs[-1]) > abs(v_coefs[0]):
                print("  Q1: V's protective effect STRENGTHENS with scale")
            else:
                print("  Q1: V is protective at all scales but does NOT strengthen")
        elif v_coefs[0] > 0 and v_coefs[-1] < 0:
            print("  Q1: V flips from risk factor to protective with scale")
        elif all(c > 0 for c in v_coefs):
            print("  Q1: V is a risk factor at all scales (high V -> more FA)")
        else:
            print("  Q1: V coefficient direction is inconsistent across scales")

    # Q2: Does H-V independence hold across scales?
    if all(abs(r) < 0.5 for r in correlations):
        print("  Q2: H and V remain independent across all scales (all |r| < 0.5)")
    else:
        high_corr = [m for m, r in zip(models, correlations) if abs(r) >= 0.5]
        print(f"  Q2: H-V become correlated at: {', '.join(high_corr)}")

    # Q3: Does V's AUC contribution scale?
    if deltas:
        if all(d > 0.01 for d in deltas):
            print("  Q3: V adds AUC at all scales")
        elif deltas[-1] > deltas[0]:
            print("  Q3: V's AUC contribution increases with scale")
        else:
            print("  Q3: V's AUC contribution diminishes with scale")

    # Q4: Does mean V change with scale?
    if mean_Vs[-1] > mean_Vs[0] * 1.1:
        print("  Q4: Mean V increases with scale (sharper decision states)")
    elif mean_Vs[-1] < mean_Vs[0] * 0.9:
        print("  Q4: Mean V decreases with scale (more peaked distributions)")
    else:
        print("  Q4: Mean V stable across scales")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading registry...")
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)
    items = registry["items"]
    print(f"  {len(items)} items")

    all_summaries = []
    all_trials = {}

    for model_name, model_path in MODELS:
        if not Path(model_path).exists():
            print(f"\n  Skipping {model_name}: {model_path} not found")
            continue

        trials = run_model(model_name, model_path, items)

        # Save per-model trials
        out_path = RESULTS_DIR / f"varentropy_{model_name}.json"
        with open(out_path, "w") as f:
            json.dump(trials, f, indent=2)

        summary = analyze_model(model_name, trials)
        if summary:
            all_summaries.append(summary)
        all_trials[model_name] = trials

    # Cross-model analysis
    if len(all_summaries) >= 2:
        comparative_analysis(all_summaries)

    # Save summary
    summary_path = RESULTS_DIR / "varentropy_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
