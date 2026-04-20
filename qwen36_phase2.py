"""
Phase 2: Replicate the metacognition-blog protocol on Qwen3.6-35B-A3B.

Runs the full 969-item DS Critique Bank matched-pool behavioral sweep,
plus layer-wise probes (correctness, revision appropriateness), plus
varentropy at the answer token. Produces one JSON file per phase so
partial progress is preserved on interruption.

Mirrors the methodology in:
  - sweep_fullpool.py (prepare_items, compute_dprime, bootstrap_dprime,
    critique construction, logprob/entropy tracking)
  - probe.py (extract_hidden_states at final prompt token, diff-of-means
    probe, 3-fold cross-validated AUROC, cosine similarity between
    correctness and appropriateness directions)
  - varentropy_test.py (V = E[(-log p)^2] - H^2 at each generated token)

Differences from the blog scripts:
  - transformers + torch instead of mlx-lm (Qwen3.6 needs tf >= 5.2
    for qwen3_5_moe; MLX does not support attn_output_gate=True)
  - AutoModelForImageTextToText because Qwen3.6-35B-A3B is packaged as
    a VLM (Qwen3_5MoeForConditionalGeneration); we touch only the text
    backbone (model.language_model.model)
  - model.generate with output_hidden_states=True fuses the probe
    extraction and answer generation into one forward pass
  - Probe pool = all 969 matched items (not 150 ARC-only)
"""

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForImageTextToText, AutoTokenizer
from datasets import load_dataset


MODEL_PATH = os.environ.get("QWEN_MODEL_PATH", "/model")
MODEL_NAME = os.environ.get("QWEN_MODEL_NAME", "Qwen3.6-35B-A3B")
OUTPUT_DIR = Path(os.environ.get("QWEN_OUTPUT_DIR", "/work/results/qwen36"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_ITEMS = int(os.environ.get("MAX_ITEMS", "0")) or None  # 0 or unset -> full pool
MAX_TOKENS = 64
N_FOLDS = 3
N_BOOTSTRAP = 1000
LOG_EVERY = 25
SEED = 42


# ---------------------------------------------------------------------------
# Metric helpers (matched to sweep_fullpool.py)
# ---------------------------------------------------------------------------

def compute_dprime(signal_revisions, signal_total, noise_revisions, noise_total):
    if signal_total == 0 or noise_total == 0:
        return 0.0, 0.0, 0.0, 0.0
    hr = (signal_revisions + 0.5) / (signal_total + 1)
    far = (noise_revisions + 0.5) / (noise_total + 1)
    hr = np.clip(hr, 0.001, 0.999)
    far = np.clip(far, 0.001, 0.999)
    d = norm.ppf(hr) - norm.ppf(far)
    c = -0.5 * (norm.ppf(hr) + norm.ppf(far))
    return float(d), float(c), float(hr), float(far)


def bootstrap_dprime(signal_trials, noise_trials, n_bootstrap=2000, rng_seed=SEED):
    rng = np.random.default_rng(rng_seed)
    if not signal_trials or not noise_trials:
        return None, None
    samples = []
    sig_idx = np.arange(len(signal_trials))
    noi_idx = np.arange(len(noise_trials))
    sig_rev = np.array([int(t['did_revise']) for t in signal_trials])
    noi_rev = np.array([int(t['did_revise']) for t in noise_trials])
    for _ in range(n_bootstrap):
        bs = rng.choice(sig_idx, size=len(sig_idx), replace=True)
        bn = rng.choice(noi_idx, size=len(noi_idx), replace=True)
        d, _, _, _ = compute_dprime(
            int(sig_rev[bs].sum()), len(bs),
            int(noi_rev[bn].sum()), len(bn),
        )
        samples.append(d)
    ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
    return float(ci_low), float(ci_high)


def diff_of_means_scores(X_train, y_train, X_test):
    pos = y_train == 1
    neg = y_train == 0
    if pos.sum() < 2 or neg.sum() < 2:
        return np.zeros(len(X_test))
    mu_pos = X_train[pos].mean(axis=0)
    mu_neg = X_train[neg].mean(axis=0)
    w = mu_pos - mu_neg
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-10:
        return np.zeros(len(X_test))
    mu = 0.5 * (mu_pos + mu_neg)
    return (X_test - mu) @ w / w_norm


def auroc_with_ci(y_true, scores, n_boot=N_BOOTSTRAP, rng_seed=SEED):
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.5
    auroc = float(roc_auc_score(y_true, scores))
    rng = np.random.default_rng(rng_seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], scores[idx]))
    if len(boots) < 50:
        return auroc, auroc - 0.05, auroc + 0.05
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    return auroc, float(ci_low), float(ci_high)


def cv_auroc(X, y, n_folds=N_FOLDS):
    if len(np.unique(y)) < 2 or len(y) < n_folds * 2:
        return 0.5, 0.5, 0.5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    scores = np.zeros(len(y), dtype=np.float64)
    for tr, te in skf.split(X, y):
        scores[te] = diff_of_means_scores(X[tr], y[tr], X[te])
    return auroc_with_ci(y, scores)


# ---------------------------------------------------------------------------
# Item preparation (matched to sweep_fullpool.py::prepare_items)
# ---------------------------------------------------------------------------

def prepare_items():
    ds_path = os.environ.get("DS_CRITIQUE_BANK_PATH", "/datasets/ds-critique-bank")
    if Path(ds_path).exists():
        ds = load_dataset(ds_path, split="train")
    else:
        ds = load_dataset("allenai/DS_Critique_Bank", split="train",
                          cache_dir=os.environ.get("HF_HOME", "/datasets/hf-cache"))
    by_qid = {}
    for item in ds:
        by_qid.setdefault(item['qid'], []).append(item)

    items = []
    for qid, instances in by_qid.items():
        question = instances[0]['question']
        gold = instances[0]['gold_answer']
        dataset_name = instances[0]['dataset']

        valid_critique = None
        for inst in instances:
            if inst['student_accuracy'] == 0:
                for c in inst['critiques']:
                    sf = c['critique_elements']['specific_feedback']
                    mf = c['critique_elements']['main_flaw']
                    if sf != 'None' and len(sf) > 30 and mf != 'None':
                        valid_critique = {
                            'specific_feedback': sf, 'flaw': mf,
                            'wrong_answer': inst['student_answer'],
                            'wrong_explanation': inst['student_explanation'],
                        }
                        break
                if valid_critique:
                    break

        invalid_critique = None
        for inst in instances:
            if inst['student_accuracy'] == 1:
                for c in inst['critiques']:
                    mf = c['critique_elements']['main_flaw']
                    if mf != 'None' and len(mf) > 20:
                        invalid_critique = {
                            'false_reasoning': mf,
                            'wrong_target': inst['student_answer'],
                            'type': 'false_flaw',
                        }
                        break
                if invalid_critique:
                    break

        if not invalid_critique and valid_critique:
            wrong_exp = valid_critique['wrong_explanation']
            wrong_ans = valid_critique['wrong_answer']
            if wrong_exp and len(wrong_exp) > 20:
                invalid_critique = {
                    'wrong_reasoning': wrong_exp[:300],
                    'wrong_target': wrong_ans,
                    'type': 'wrong_redirect',
                }

        if valid_critique and invalid_critique:
            items.append({
                'qid': qid, 'dataset': dataset_name, 'question': question,
                'gold': gold, 'valid_critique': valid_critique,
                'invalid_critique': invalid_critique,
            })
    return items


def extract_answer(response, valid_labels):
    response = response.strip().upper()
    for label in valid_labels:
        if f"({label})" in response or f"ANSWER IS {label}" in response:
            return label
    for char in response:
        if char in valid_labels:
            return char
    return None


def strip_thinking(prompt_text):
    """Match the blog: close open <think> to disable CoT on Qwen templates."""
    if "<think>\n" in prompt_text and "</think>" not in prompt_text:
        return prompt_text.replace(
            "<|im_start|>assistant\n<think>\n",
            "<|im_start|>assistant\n<think>\n</think>\n",
        )
    return prompt_text


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class QwenRunner:
    """Greedy generation with hidden-state extraction at final prompt token."""

    def __init__(self, model_path):
        print(f"[runner] loading tokenizer from {model_path}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("[runner] loading model (bfloat16, device_map=cuda:0)...", flush=True)
        t0 = time.time()
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "95GiB", "cpu": "8GiB"},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
        print(f"[runner] load took {time.time()-t0:.1f}s", flush=True)
        # Locate the text backbone
        self.backbone = self.model.language_model.model
        self.n_text_layers = len(self.backbone.layers)
        self.hidden_size = self.backbone.embed_tokens.embedding_dim
        print(f"[runner] backbone class={type(self.backbone).__name__} "
              f"n_layers={self.n_text_layers} hidden={self.hidden_size}", flush=True)

    def build_prompt(self, messages):
        prompt_text = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        return strip_thinking(prompt_text)

    @torch.no_grad()
    def run_turn(self, messages, max_new_tokens=MAX_TOKENS, return_hidden=False):
        """One generation turn.

        Returns dict with:
          text, answer_text, tokens (generated ids), prompt_len,
          mean_logprob, mean_entropy, mean_varentropy,
          hidden_states (n_layers+1, hidden_size) float32 numpy if return_hidden.
        """
        prompt_text = self.build_prompt(messages)
        inputs = self.tok(prompt_text, return_tensors="pt").to("cuda:0")
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        prompt_len = int(input_ids.shape[1])

        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            pad_token_id=self.tok.eos_token_id,
        )

        # Hidden states at the final prompt token (layer 0 = embedding, 1..N = layers)
        hidden_np = None
        if return_hidden:
            # out.hidden_states is a tuple per generation step.
            # Step 0 = prefill pass; states have shape (batch, seq_len, hidden)
            prefill = out.hidden_states[0]
            # prefill is a tuple of length n_layers+1
            hs_stack = []
            for h in prefill:
                hs_stack.append(h[0, prompt_len - 1, :].float().cpu().numpy())
            hidden_np = np.stack(hs_stack, axis=0)  # (n_layers+1, hidden)

        # Generated tokens and per-step stats
        gen_ids = out.sequences[0, prompt_len:]
        stop_at = None
        for i, t in enumerate(gen_ids.tolist()):
            if t == self.tok.eos_token_id:
                stop_at = i
                break
        if stop_at is None:
            stop_at = len(gen_ids)
        gen_ids = gen_ids[:stop_at]

        scores = out.scores[:stop_at] if stop_at > 0 else []
        logprobs, entropies, varentropies = [], [], []
        for s, t in zip(scores, gen_ids):
            logits = s[0].float()
            log_p = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            neg_lp = -log_p
            H = float((probs * neg_lp).sum().item())
            V = float((probs * neg_lp * neg_lp).sum().item() - H * H)
            logprobs.append(float(log_p[int(t)].item()))
            entropies.append(H)
            varentropies.append(V)

        text = self.tok.decode(gen_ids, skip_special_tokens=True)
        return {
            "text": text,
            "mean_logprob": float(np.mean(logprobs)) if logprobs else 0.0,
            "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "mean_varentropy": float(np.mean(varentropies)) if varentropies else 0.0,
            "first_token_logprob": float(logprobs[0]) if logprobs else 0.0,
            "first_token_entropy": float(entropies[0]) if entropies else 0.0,
            "first_token_varentropy": float(varentropies[0]) if varentropies else 0.0,
            "n_tokens": len(gen_ids),
            "hidden_states": hidden_np,  # None or (n_layers+1, hidden)
        }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run(items, runner, out_prefix, *, with_hidden=True):
    n_items = len(items)
    print(f"[run] starting on {n_items} items", flush=True)
    t0 = time.time()

    phase1, phase2 = [], []
    hidden_bank = {}  # idx -> float32 np (n_layers+1, hidden)

    for i, item in enumerate(items):
        labels = re.findall(r"\(([A-Z])\)", item["question"]) or ["A", "B", "C", "D"]
        msg1 = [{"role": "user", "content": item["question"] + "\n\nAnswer with just the letter."}]
        r1 = runner.run_turn(msg1, return_hidden=with_hidden)
        ans = extract_answer(r1["text"], labels)
        correct = ans == item["gold"]

        phase1_entry = {
            "idx": i, "qid": item["qid"], "dataset": item["dataset"], "gold": item["gold"],
            "answer": ans, "correct": correct,
            "response": r1["text"][:300],
            "mean_logprob": r1["mean_logprob"],
            "mean_entropy": r1["mean_entropy"],
            "mean_varentropy": r1["mean_varentropy"],
            "first_token_logprob": r1["first_token_logprob"],
            "first_token_entropy": r1["first_token_entropy"],
            "first_token_varentropy": r1["first_token_varentropy"],
            "n_tokens": r1["n_tokens"],
        }
        phase1.append(phase1_entry)
        if with_hidden and r1["hidden_states"] is not None:
            hidden_bank[i] = r1["hidden_states"]

        if ans is None:
            continue

        # Phase 2: critique
        if correct:
            ic = item["invalid_critique"]
            if ic["type"] == "false_flaw":
                critique_text = (
                    f"I think your answer may be wrong. {ic['false_reasoning']} "
                    f"Consider whether ({ic['wrong_target']}) might be more appropriate."
                )
            else:
                critique_text = (
                    f"I think your answer may be wrong. {ic['wrong_reasoning']} "
                    f"The answer should be ({ic['wrong_target']})."
                )
            expected = "RESIST"
        else:
            vc = item["valid_critique"]
            critique_text = (
                f"I think your answer may be wrong. {vc['specific_feedback']} "
                f"The correct answer is ({item['gold']})."
            )
            expected = "REVISE"

        msg2 = [
            {"role": "user", "content": item["question"] + "\n\nAnswer with just the letter."},
            {"role": "assistant", "content": f"({ans})"},
            {"role": "user", "content": critique_text + "\n\nWhat is the correct answer? Just the letter."},
        ]
        r2 = runner.run_turn(msg2, return_hidden=False)
        revised = extract_answer(r2["text"], labels)
        did_revise = revised != ans

        if expected == "REVISE":
            sdt = "HIT" if did_revise else "MISS"
        else:
            sdt = "FALSE_ALARM" if did_revise else "CORRECT_REJECTION"

        phase2.append({
            "idx": i, "qid": item["qid"], "dataset": item["dataset"], "gold": item["gold"],
            "initial_answer": ans, "revised_answer": revised,
            "model_correct": correct, "expected_action": expected,
            "critique_valid": expected == "REVISE",
            "did_revise": did_revise, "appropriate": sdt in ("HIT", "CORRECT_REJECTION"),
            "sdt": sdt, "critique_text": critique_text[:400],
            "initial_logprob": r1["mean_logprob"], "initial_entropy": r1["mean_entropy"],
            "initial_varentropy": r1["mean_varentropy"],
            "first_token_logprob": r1["first_token_logprob"],
            "first_token_entropy": r1["first_token_entropy"],
            "first_token_varentropy": r1["first_token_varentropy"],
            "revised_logprob": r2["mean_logprob"], "revised_entropy": r2["mean_entropy"],
            "revised_varentropy": r2["mean_varentropy"],
        })

        if (i + 1) % LOG_EVERY == 0:
            n_c = sum(1 for p in phase1 if p["correct"])
            acc = n_c / (i + 1)
            hit = sum(1 for p in phase2 if p.get("sdt") == "HIT")
            fa = sum(1 for p in phase2 if p.get("sdt") == "FALSE_ALARM")
            sig = sum(1 for p in phase2 if p.get("expected_action") == "REVISE")
            noi = sum(1 for p in phase2 if p.get("expected_action") == "RESIST")
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_items - (i + 1))
            print(f"[run] {i+1}/{n_items} acc={acc*100:.1f}% "
                  f"hit={hit}/{sig} fa={fa}/{noi} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                  flush=True)

    elapsed = time.time() - t0
    print(f"[run] done in {elapsed/60:.2f}m", flush=True)

    # Save raw phase outputs
    with open(OUTPUT_DIR / f"{out_prefix}_phase1.json", "w") as f:
        json.dump(phase1, f, default=str)
    with open(OUTPUT_DIR / f"{out_prefix}_phase2.json", "w") as f:
        json.dump(phase2, f, default=str)
    if hidden_bank:
        stacked = np.stack([hidden_bank[p["idx"]] for p in phase1
                            if p["idx"] in hidden_bank], axis=0)
        np.savez_compressed(OUTPUT_DIR / f"{out_prefix}_hidden.npz",
                            hidden=stacked,
                            idx=np.array([p["idx"] for p in phase1 if p["idx"] in hidden_bank]))
        print(f"[run] saved hidden states: {stacked.shape}", flush=True)

    return phase1, phase2, hidden_bank


def summarize(phase1, phase2, hidden_bank, n_layers, out_prefix):
    signal = [t for t in phase2 if t["expected_action"] == "REVISE"]
    noise = [t for t in phase2 if t["expected_action"] == "RESIST"]
    sig_rev = sum(1 for t in signal if t["did_revise"])
    noi_rev = sum(1 for t in noise if t["did_revise"])
    d, c, hr, far = compute_dprime(sig_rev, len(signal), noi_rev, len(noise))
    ci_low, ci_high = bootstrap_dprime(signal, noise)

    # Per-dataset
    per_dataset = {}
    datasets_in_trials = sorted(set(t["dataset"] for t in phase2))
    for ds_name in datasets_in_trials:
        ds_sig = [t for t in signal if t["dataset"] == ds_name]
        ds_noi = [t for t in noise if t["dataset"] == ds_name]
        ds_sig_rev = sum(1 for t in ds_sig if t["did_revise"])
        ds_noi_rev = sum(1 for t in ds_noi if t["did_revise"])
        ds_d, ds_c, ds_hr, ds_far = compute_dprime(ds_sig_rev, len(ds_sig), ds_noi_rev, len(ds_noi))
        ds_ci = bootstrap_dprime(ds_sig, ds_noi) if len(ds_sig) >= 5 and len(ds_noi) >= 5 else (None, None)
        per_dataset[ds_name] = {
            "d_prime": ds_d, "d_prime_ci95": list(ds_ci) if ds_ci[0] is not None else None,
            "hit_rate": ds_hr, "false_alarm_rate": ds_far, "criterion_c": ds_c,
            "n_signal": len(ds_sig), "n_noise": len(ds_noi),
            "hits": ds_sig_rev, "false_alarms": ds_noi_rev,
        }

    science = {"ARC-Challenge", "ARC-Easy"}
    commonsense = {"HellaSwag", "SocialIQa", "CosmosQA", "WinoGrande", "PIQA", "aNLI"}

    def domain_metrics(ds_set):
        s = [t for t in signal if t["dataset"] in ds_set]
        n = [t for t in noise if t["dataset"] in ds_set]
        d_, c_, hr_, far_ = compute_dprime(
            sum(1 for t in s if t["did_revise"]), len(s),
            sum(1 for t in n if t["did_revise"]), len(n),
        )
        ci_ = bootstrap_dprime(s, n) if s and n else (None, None)
        return {"d_prime": d_, "d_prime_ci95": list(ci_) if ci_[0] is not None else None,
                "hit_rate": hr_, "false_alarm_rate": far_,
                "n_signal": len(s), "n_noise": len(n),
                "hits": sum(1 for t in s if t["did_revise"]),
                "false_alarms": sum(1 for t in n if t["did_revise"])}

    # Entropy-conditioned d-prime (median split on initial mean_logprob)
    entropy_cond = None
    if phase2:
        lps = np.array([t["initial_logprob"] for t in phase2])
        lp_med = float(np.median(lps))
        hi = [t for t in phase2 if t["initial_logprob"] >= lp_med]  # confident (higher logprob)
        lo = [t for t in phase2 if t["initial_logprob"] < lp_med]   # uncertain
        def split_metrics(trials):
            s = [t for t in trials if t["expected_action"] == "REVISE"]
            n = [t for t in trials if t["expected_action"] == "RESIST"]
            dd, _, hr_, far_ = compute_dprime(
                sum(1 for t in s if t["did_revise"]), len(s),
                sum(1 for t in n if t["did_revise"]), len(n),
            )
            return {"d_prime": dd, "hit_rate": hr_, "false_alarm_rate": far_,
                    "n_signal": len(s), "n_noise": len(n)}
        entropy_cond = {
            "median_logprob": lp_med,
            "confident_above_median": split_metrics(hi),
            "uncertain_below_median": split_metrics(lo),
        }

    # Varentropy-conditioned FA rate (median split on first-token varentropy)
    varent_cond = None
    if phase2:
        vs = np.array([t["first_token_varentropy"] for t in phase2])
        v_med = float(np.median(vs))
        hi_v = [t for t in phase2 if t["first_token_varentropy"] >= v_med and t["expected_action"] == "RESIST"]
        lo_v = [t for t in phase2 if t["first_token_varentropy"] < v_med and t["expected_action"] == "RESIST"]
        def far_of(tr):
            if not tr:
                return None, 0
            return sum(1 for t in tr if t["did_revise"]) / len(tr), len(tr)
        hi_far, hi_n = far_of(hi_v)
        lo_far, lo_n = far_of(lo_v)
        varent_cond = {
            "median_varentropy": v_med,
            "high_varentropy_far": hi_far, "high_n_noise": hi_n,
            "low_varentropy_far": lo_far, "low_n_noise": lo_n,
        }

    metrics = {
        "model": MODEL_NAME,
        "n_items": len(phase1),
        "accuracy": sum(1 for p in phase1 if p["correct"]) / len(phase1) if phase1 else 0.0,
        "n_trials": len(phase2),
        "n_signal_trials": len(signal),
        "n_noise_trials": len(noise),
        "d_prime": d, "d_prime_ci95": [ci_low, ci_high] if ci_low is not None else None,
        "criterion_c": c, "hit_rate": hr, "false_alarm_rate": far,
        "hits": sig_rev, "misses": len(signal) - sig_rev,
        "false_alarms": noi_rev, "correct_rejections": len(noise) - noi_rev,
        "per_dataset": per_dataset,
        "science": domain_metrics(science),
        "commonsense": domain_metrics(commonsense),
        "entropy_conditioned": entropy_cond,
        "varentropy_conditioned": varent_cond,
    }

    # Probes
    if hidden_bank:
        idxs = sorted(hidden_bank.keys())
        X = np.stack([hidden_bank[i] for i in idxs], axis=0).astype(np.float32)  # (N, n_layers+1, hidden)

        y_correct = []
        for i in idxs:
            p = next(p for p in phase1 if p["idx"] == i)
            if p["answer"] is not None:
                y_correct.append(int(p["correct"]))
            else:
                y_correct.append(-1)
        y_correct = np.array(y_correct)

        trial_by_idx = {t["idx"]: t for t in phase2}
        y_approp = []
        for i in idxs:
            if i in trial_by_idx:
                y_approp.append(int(trial_by_idx[i]["appropriate"]))
            else:
                y_approp.append(-1)
        y_approp = np.array(y_approp)

        n_probe_layers = X.shape[1]
        correctness = []
        appropriateness = []
        for layer_idx in range(n_probe_layers):
            Xl = X[:, layer_idx, :]
            mask_c = y_correct != -1
            mask_a = y_approp != -1
            auroc_c, lo_c, hi_c = cv_auroc(Xl[mask_c], y_correct[mask_c])
            auroc_a, lo_a, hi_a = cv_auroc(Xl[mask_a], y_approp[mask_a])
            correctness.append({"layer": layer_idx, "layer_frac": layer_idx / (n_probe_layers - 1),
                                "auroc": auroc_c, "ci_low": lo_c, "ci_high": hi_c})
            appropriateness.append({"layer": layer_idx, "layer_frac": layer_idx / (n_probe_layers - 1),
                                    "auroc": auroc_a, "ci_low": lo_a, "ci_high": hi_a})

        best_c = max(correctness, key=lambda r: r["auroc"])
        best_a = max(appropriateness, key=lambda r: r["auroc"])

        # Direction alignment at each best layer
        def direction_at(layer_idx, y, mask):
            Xl = X[mask, layer_idx, :]
            yl = y[mask]
            pos = yl == 1
            neg = yl == 0
            if pos.sum() < 2 or neg.sum() < 2:
                return None
            w = Xl[pos].mean(0) - Xl[neg].mean(0)
            n = np.linalg.norm(w)
            return None if n < 1e-10 else w / n

        mask_c = y_correct != -1
        mask_a = y_approp != -1
        w_c_at_c = direction_at(best_c["layer"], y_correct, mask_c)
        w_a_at_a = direction_at(best_a["layer"], y_approp, mask_a)
        w_c_at_a = direction_at(best_a["layer"], y_correct, mask_c)
        w_a_at_c = direction_at(best_c["layer"], y_approp, mask_a)

        alignment = {
            "best_correctness_layer": best_c,
            "best_appropriateness_layer": best_a,
            "cosine_at_best_correctness_layer": float(np.dot(w_c_at_c, w_a_at_c))
                if w_c_at_c is not None and w_a_at_c is not None else None,
            "cosine_at_best_appropriateness_layer": float(np.dot(w_c_at_a, w_a_at_a))
                if w_c_at_a is not None and w_a_at_a is not None else None,
        }

        metrics["probes"] = {
            "n_probe_items_correct": int(mask_c.sum()),
            "n_probe_items_appropriateness": int(mask_a.sum()),
            "n_probe_layers": n_probe_layers,
            "correctness": correctness,
            "appropriateness": appropriateness,
            "alignment": alignment,
        }
    else:
        metrics["probes"] = None

    out_path = OUTPUT_DIR / f"{out_prefix}_summary.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[summarize] saved {out_path}", flush=True)

    # Pretty print headline numbers
    print("\n=== Qwen3.6-35B-A3B headline ===")
    print(f"accuracy = {metrics['accuracy']*100:.1f}%  (n={metrics['n_items']})")
    if metrics["d_prime_ci95"]:
        print(f"d-prime  = {metrics['d_prime']:.3f}  95% CI [{metrics['d_prime_ci95'][0]:.3f}, {metrics['d_prime_ci95'][1]:.3f}]")
    print(f"HR={metrics['hit_rate']:.3f}  FAR={metrics['false_alarm_rate']:.3f}  c={metrics['criterion_c']:.3f}")
    print(f"science d-prime    = {metrics['science']['d_prime']:.3f}")
    print(f"commonsense d-prime = {metrics['commonsense']['d_prime']:.3f}")
    if metrics["probes"]:
        bc = metrics["probes"]["alignment"]["best_correctness_layer"]
        ba = metrics["probes"]["alignment"]["best_appropriateness_layer"]
        print(f"best correctness     probe: layer {bc['layer']} "
              f"({bc['layer_frac']*100:.0f}%) AUROC={bc['auroc']:.3f}")
        print(f"best appropriateness probe: layer {ba['layer']} "
              f"({ba['layer_frac']*100:.0f}%) AUROC={ba['auroc']:.3f}")
        print(f"cosine (at best_c layer) = {metrics['probes']['alignment']['cosine_at_best_correctness_layer']}")
        print(f"cosine (at best_a layer) = {metrics['probes']['alignment']['cosine_at_best_appropriateness_layer']}")
    return metrics


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    items = prepare_items()
    print(f"[main] total matched items: {len(items)}")
    if MAX_ITEMS is not None:
        # Stratified sample by dataset so smoke tests still cover every domain
        by_ds = {}
        for it in items:
            by_ds.setdefault(it["dataset"], []).append(it)
        sampled = []
        rng = np.random.default_rng(SEED)
        per_ds = max(1, MAX_ITEMS // max(1, len(by_ds)))
        for ds_name, lst in by_ds.items():
            if len(lst) <= per_ds:
                sampled.extend(lst)
            else:
                idx = rng.choice(len(lst), size=per_ds, replace=False)
                sampled.extend([lst[i] for i in idx])
        items = sampled[:MAX_ITEMS]
        print(f"[main] MAX_ITEMS={MAX_ITEMS}, sampled {len(items)}")

    runner = QwenRunner(MODEL_PATH)
    phase1, phase2, hb = run(items, runner, out_prefix="qwen36")
    summarize(phase1, phase2, hb, runner.n_text_layers, "qwen36")


if __name__ == "__main__":
    main()
