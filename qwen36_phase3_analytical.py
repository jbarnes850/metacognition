"""
Phase 3 analytical experiments on Qwen3.6-35B-A3B.

All four experiments use only the Phase 2 outputs (qwen36_phase1.json,
qwen36_phase2.json, qwen36_hidden_final.npz) produced by
qwen36_phase2_sglang.py. No additional inference.

Experiments and citations:

  E_TASK_DECOMP. Maskey et al. (2026). Decompose the appropriateness
    direction at the final layer into a global diff-of-means and per-
    domain diff-of-means. Report cos(global, science-residual),
    cos(global, commonsense-residual), cos(science, commonsense), and
    held-out AUROC using (a) only the global direction vs (b) domain-
    specific directions. Prediction: if the commonsense ceiling in
    smaller Qwens is a missing task-specific subspace rather than a
    missing global signal, we expect cos(sci, com) < cos(global, sci)
    and higher AUROC_com_by_com than AUROC_com_by_global.

  E_VAR2D. Ahmed et al. (2026) [varentropy framework] + the blog's
    varentropy-conditioned FA finding. 2x2 factorial cut on (entropy,
    varentropy) of first generated token, measured only on noise
    (invalid-critique) trials to track false alarm rate. In Qwen3.5
    high-V was protective (lower FAR); in E4B it was a risk factor.
    Qwen3.6 gets its first row on this axis.

  E_COSINE_DOMAIN. The blog's cosine similarity result (-0.86 at 0.8B
    -> +0.30 at 9B) is computed pooled across domains. Split by
    science vs commonsense. Prediction: if the commonsense ceiling
    reflects a failure of control-confidence integration specifically
    on commonsense, cos(correctness, appropriateness) should be lower
    on commonsense than on science.

  E_CONF_GATE. Kadavath et al. (2022); Kumaran et al. (2026). FAR
    conditioned on confidence (median split on first-token logprob),
    split further by domain. The blog found gap opens up to 0.29 at
    9B. Does Qwen3.6 at 35B continue the trend, or does the gap
    saturate?

A small bonus cut E_RESP_LENGTH captures mean response-token count per
(SDT, domain) since response length has been informally associated
with reasoning effort in Qwen templates.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

OUTPUT_DIR = Path(os.environ.get("QWEN_OUTPUT_DIR", "/work/results/qwen36"))
OUT_PREFIX = "qwen36"
SEED = 42

SCIENCE = {"ARC-Challenge", "ARC-Easy"}
COMMONSENSE = {"HellaSwag", "SocialIQa", "CosmosQA", "WinoGrande", "PIQA", "aNLI"}


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


def load_phase2():
    phase1 = json.load(open(OUTPUT_DIR / f"{OUT_PREFIX}_phase1.json"))
    phase2 = json.load(open(OUTPUT_DIR / f"{OUT_PREFIX}_phase2.json"))
    summary = json.load(open(OUTPUT_DIR / f"{OUT_PREFIX}_summary.json"))
    npz_path = OUTPUT_DIR / f"{OUT_PREFIX}_hidden_final.npz"
    if not npz_path.exists():
        # Fallback: transformers-backed Phase 2 saves as qwen36_hidden.npz
        npz_path = OUTPUT_DIR / f"{OUT_PREFIX}_hidden.npz"
    npz = np.load(npz_path)
    hidden = npz["hidden"]  # (N, hidden) or (N, n_layers+1, hidden)
    if hidden.ndim == 3:
        hidden = hidden[:, -1, :]
    idx = npz["idx"]
    return phase1, phase2, summary, hidden, idx


# ---------------------------------------------------------------------------
# E_TASK_DECOMP
# ---------------------------------------------------------------------------

def e_task_decomp(hidden, idx_arr, phase2):
    trial_by_idx = {t["idx"]: t for t in phase2}
    X_all, y_all, dom_all = [], [], []
    for k, i in enumerate(idx_arr):
        t = trial_by_idx.get(int(i))
        if t is None:
            continue
        X_all.append(hidden[k])
        y_all.append(int(t["appropriate"]))
        dom = "sci" if t["dataset"] in SCIENCE else "com" if t["dataset"] in COMMONSENSE else "oth"
        dom_all.append(dom)
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all)
    dom_all = np.array(dom_all)

    def dir_of(mask):
        Xm, ym = X_all[mask], y_all[mask]
        if (ym == 1).sum() < 3 or (ym == 0).sum() < 3:
            return None
        w = Xm[ym == 1].mean(0) - Xm[ym == 0].mean(0)
        n = np.linalg.norm(w) + 1e-10
        return w / n

    def cos(a, b):
        if a is None or b is None:
            return None
        return float(np.dot(a, b))

    w_glob = dir_of(np.ones(len(y_all), dtype=bool))
    w_sci = dir_of(dom_all == "sci")
    w_com = dir_of(dom_all == "com")

    def held_out_auroc(mask_fit, mask_eval):
        if mask_fit.sum() < 6 or mask_eval.sum() < 6:
            return None
        Xf, yf = X_all[mask_fit], y_all[mask_fit]
        Xe, ye = X_all[mask_eval], y_all[mask_eval]
        if len(np.unique(yf)) < 2 or len(np.unique(ye)) < 2:
            return None
        scores = diff_of_means_scores(Xf, yf, Xe)
        return float(roc_auc_score(ye, scores))

    # Leave-one-domain-out style
    sci_mask = dom_all == "sci"
    com_mask = dom_all == "com"
    auroc_sci_by_global = held_out_auroc(np.ones(len(y_all), dtype=bool), sci_mask)
    auroc_com_by_global = held_out_auroc(np.ones(len(y_all), dtype=bool), com_mask)
    auroc_sci_by_sci = held_out_auroc(sci_mask, sci_mask)
    auroc_com_by_com = held_out_auroc(com_mask, com_mask)
    auroc_sci_by_com = held_out_auroc(com_mask, sci_mask)
    auroc_com_by_sci = held_out_auroc(sci_mask, com_mask)

    return {
        "n_sci": int(sci_mask.sum()),
        "n_com": int(com_mask.sum()),
        "cos_global_sci": cos(w_glob, w_sci),
        "cos_global_com": cos(w_glob, w_com),
        "cos_sci_com": cos(w_sci, w_com),
        "auroc_sci_by_global": auroc_sci_by_global,
        "auroc_com_by_global": auroc_com_by_global,
        "auroc_sci_by_sci": auroc_sci_by_sci,
        "auroc_com_by_com": auroc_com_by_com,
        "auroc_sci_by_com": auroc_sci_by_com,
        "auroc_com_by_sci": auroc_com_by_sci,
        "citation": "Maskey, Dras, Naseem (2026) arxiv 2603.27518",
        "prediction": (
            "If the commonsense ceiling reflects a missing task-specific "
            "subspace, auroc_com_by_com > auroc_com_by_global AND "
            "cos_sci_com < cos_global_sci."
        ),
    }


# ---------------------------------------------------------------------------
# E_VAR2D
# ---------------------------------------------------------------------------

def e_var2d(phase2):
    if not phase2:
        return None
    noise = [t for t in phase2 if t["expected_action"] == "RESIST"]
    if not noise:
        return None
    Hs = np.array([t["first_token_entropy"] for t in noise])
    Vs = np.array([t["first_token_varentropy"] for t in noise])
    h_med = float(np.median(Hs))
    v_med = float(np.median(Vs))

    cells = {}
    for (h_lbl, h_sel), (v_lbl, v_sel) in [
        (("high_H", Hs >= h_med), ("high_V", Vs >= v_med)),
        (("high_H", Hs >= h_med), ("low_V", Vs < v_med)),
        (("low_H", Hs < h_med), ("high_V", Vs >= v_med)),
        (("low_H", Hs < h_med), ("low_V", Vs < v_med)),
    ]:
        mask = h_sel & v_sel
        n = int(mask.sum())
        if n == 0:
            cells[f"{h_lbl}__{v_lbl}"] = {"n": 0, "far": None}
            continue
        far = float(np.mean([1.0 if noise[i]["did_revise"] else 0.0
                             for i in range(len(noise)) if mask[i]]))
        cells[f"{h_lbl}__{v_lbl}"] = {"n": n, "far": far}

    return {
        "median_entropy": h_med,
        "median_varentropy": v_med,
        "cells": cells,
        "blog_reference": (
            "Qwen: high-V protective, FAR gap 17-31pp. E4B: high-V is risk, "
            "FAR gap 32pp reversed. Qwen3.6 is a new data point on this axis."
        ),
        "citation": "Ahmed, Ong, DeLuca (2026) arxiv 2603.24929",
    }


# ---------------------------------------------------------------------------
# E_COSINE_DOMAIN
# ---------------------------------------------------------------------------

def e_cosine_domain(hidden, idx_arr, phase1, phase2):
    trial_by_idx = {t["idx"]: t for t in phase2}
    p1_by_idx = {p["idx"]: p for p in phase1}

    # Correctness y is from phase1, appropriateness y is from phase2
    X_c, y_c, dom_c = [], [], []
    X_a, y_a, dom_a = [], [], []
    for k, i in enumerate(idx_arr):
        p = p1_by_idx.get(int(i))
        t = trial_by_idx.get(int(i))
        if p is not None and p["answer"] is not None:
            X_c.append(hidden[k])
            y_c.append(int(p["correct"]))
            dom = "sci" if p["dataset"] in SCIENCE else "com" if p["dataset"] in COMMONSENSE else "oth"
            dom_c.append(dom)
        if t is not None:
            X_a.append(hidden[k])
            y_a.append(int(t["appropriate"]))
            dom = "sci" if t["dataset"] in SCIENCE else "com" if t["dataset"] in COMMONSENSE else "oth"
            dom_a.append(dom)
    X_c, y_c, dom_c = np.array(X_c, dtype=np.float32), np.array(y_c), np.array(dom_c)
    X_a, y_a, dom_a = np.array(X_a, dtype=np.float32), np.array(y_a), np.array(dom_a)

    def dir_of(X, y, mask):
        Xm, ym = X[mask], y[mask]
        if (ym == 1).sum() < 3 or (ym == 0).sum() < 3:
            return None
        w = Xm[ym == 1].mean(0) - Xm[ym == 0].mean(0)
        n = np.linalg.norm(w) + 1e-10
        return w / n

    def cos(a, b):
        if a is None or b is None:
            return None
        return float(np.dot(a, b))

    w_c_all = dir_of(X_c, y_c, np.ones(len(y_c), dtype=bool))
    w_a_all = dir_of(X_a, y_a, np.ones(len(y_a), dtype=bool))
    w_c_sci = dir_of(X_c, y_c, dom_c == "sci")
    w_c_com = dir_of(X_c, y_c, dom_c == "com")
    w_a_sci = dir_of(X_a, y_a, dom_a == "sci")
    w_a_com = dir_of(X_a, y_a, dom_a == "com")

    return {
        "cosine_all": cos(w_c_all, w_a_all),
        "cosine_science": cos(w_c_sci, w_a_sci),
        "cosine_commonsense": cos(w_c_com, w_a_com),
        "n_science": int((dom_c == "sci").sum()),
        "n_commonsense": int((dom_c == "com").sum()),
        "prediction": (
            "If the commonsense ceiling reflects failed integration of "
            "confidence and control specifically on commonsense, "
            "cosine_commonsense < cosine_science."
        ),
        "blog_reference": "At 9B pooled cosine=0.30; per-domain not reported.",
    }


# ---------------------------------------------------------------------------
# E_CONF_GATE
# ---------------------------------------------------------------------------

def e_conf_gate(phase2):
    if not phase2:
        return None
    # Split on first-token logprob (confidence proxy); further split by domain
    def conf_gap(trials):
        if len(trials) < 4:
            return None
        lps = np.array([t["first_token_logprob"] for t in trials])
        med = float(np.median(lps))
        confident = [t for t in trials if t["first_token_logprob"] >= med]
        uncertain = [t for t in trials if t["first_token_logprob"] < med]
        def far(lst):
            if not lst:
                return None
            return sum(1 for t in lst if t["did_revise"]) / len(lst)
        def hr(lst):
            if not lst:
                return None
            return sum(1 for t in lst if t["did_revise"] and t["expected_action"] == "REVISE") / max(
                1, sum(1 for t in lst if t["expected_action"] == "REVISE"))
        # Compute FAR on noise trials within each group
        conf_noise = [t for t in confident if t["expected_action"] == "RESIST"]
        unc_noise = [t for t in uncertain if t["expected_action"] == "RESIST"]
        return {
            "median_logprob": med,
            "confident_far": far(conf_noise),
            "uncertain_far": far(unc_noise),
            "confident_n_noise": len(conf_noise),
            "uncertain_n_noise": len(unc_noise),
            "gap_unc_minus_conf": (
                far(unc_noise) - far(conf_noise)
                if far(conf_noise) is not None and far(unc_noise) is not None else None
            ),
        }

    return {
        "overall": conf_gap(phase2),
        "science": conf_gap([t for t in phase2 if t["dataset"] in SCIENCE]),
        "commonsense": conf_gap([t for t in phase2 if t["dataset"] in COMMONSENSE]),
        "blog_reference": "9B pooled gap: unc=0.511, conf=0.217, gap=0.29",
        "citation": "Kadavath et al. (2022), Kumaran et al. (2026)",
    }


# ---------------------------------------------------------------------------
# E_RESP_LENGTH (bonus)
# ---------------------------------------------------------------------------

def e_resp_length(phase1, phase2):
    p1_by = {p["idx"]: p for p in phase1}
    buckets = {}
    for t in phase2:
        p = p1_by.get(t["idx"])
        if p is None:
            continue
        dom = "sci" if t["dataset"] in SCIENCE else "com" if t["dataset"] in COMMONSENSE else "oth"
        key = f"{t['sdt']}__{dom}"
        buckets.setdefault(key, []).append(p["n_tokens"])
    return {k: {"n": len(v), "mean_n_tokens": float(np.mean(v)) if v else None}
            for k, v in buckets.items()}


def main():
    t0 = time.time()
    phase1, phase2, summary, hidden, idx_arr = load_phase2()
    print(f"[phase3] loaded: phase1={len(phase1)} phase2={len(phase2)} "
          f"hidden_shape={hidden.shape}", flush=True)

    results = {
        "model": summary.get("model", "Qwen3.6-35B-A3B"),
        "source_summary": {
            "accuracy": summary.get("accuracy"),
            "d_prime": summary.get("d_prime"),
            "d_prime_ci95": summary.get("d_prime_ci95"),
            "hit_rate": summary.get("hit_rate"),
            "false_alarm_rate": summary.get("false_alarm_rate"),
            "n_trials": summary.get("n_trials"),
        },
        "E_task_decomp": e_task_decomp(hidden, idx_arr, phase2),
        "E_var2d": e_var2d(phase2),
        "E_cosine_domain": e_cosine_domain(hidden, idx_arr, phase1, phase2),
        "E_conf_gate": e_conf_gate(phase2),
        "E_resp_length": e_resp_length(phase1, phase2),
        "elapsed_s": None,
    }
    results["elapsed_s"] = time.time() - t0

    out_path = OUTPUT_DIR / f"{OUT_PREFIX}_phase3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[phase3] saved -> {out_path}", flush=True)

    print("\n=== Phase 3 quick look ===")
    print("task_decomp:", json.dumps(results["E_task_decomp"], indent=2))
    print("var2d:", json.dumps(results["E_var2d"], indent=2))
    print("cosine_domain:", json.dumps(results["E_cosine_domain"], indent=2))
    print("conf_gate (overall):", json.dumps(results["E_conf_gate"]["overall"], indent=2))


if __name__ == "__main__":
    main()
