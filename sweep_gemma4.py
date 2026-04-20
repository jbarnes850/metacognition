"""
Gemma 4 metacognition benchmark: d-prime across E4B-IT and 26B-A4B-IT.

Adapted from sweep_fullpool.py for Gemma 4 architecture:
- Uses mlx_vlm (not mlx_lm) for model loading
- Gemma 4 IT chat template with thinking mode suppressed
- Same d-prime computation, same DS Critique Bank items
- Results comparable to Qwen3.5 sweep_v3_fullpool
"""

import json
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from datasets import load_dataset
from mlx_vlm import load as vlm_load
from mlx_vlm.generate import generate_step as vlm_generate_step
from scipy.stats import norm


RESULTS_DIR = Path("results/sweep_gemma4")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_TOKENS = 64


def generate_with_logprobs(model, tokenizer, messages, max_tokens=MAX_TOKENS):
    """Generate response with per-token logprobs and entropy via mlx_vlm."""
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    # Gemma 4 IT: thinking mode is opt-in via enable_thinking=True.
    # Without it, the default template does not include <|think|>.
    # If thinking tags somehow appear, close them.
    if "<|think|>" in prompt or "<think>" in prompt:
        if "</think>" not in prompt:
            prompt = prompt + "</think>\n\n"

    # mlx_vlm generate_step expects 2D input (batch, seq)
    prompt_tokens = mx.array(tokenizer.encode(prompt)).reshape(1, -1)

    tokens = []
    logprobs = []
    entropies = []

    # mlx_vlm generate_step requires pixel_values and mask (None for text-only)
    for step_output, _ in zip(
        vlm_generate_step(
            prompt_tokens, model, None, None, max_tokens=max_tokens
        ),
        range(max_tokens),
    ):
        token, logits = step_output
        token_id = int(token)
        if token_id == tokenizer.eos_token_id:
            break
        # Also stop on turn delimiter tokens (Gemma 4 specific)
        if token_id in (106, 1):  # <turn|>, <eos>
            break
        tokens.append(token_id)
        probs = mx.softmax(logits.reshape(-1), axis=-1)
        log_p = mx.log(probs + 1e-10)
        logprobs.append(float(log_p[token_id]))
        entropies.append(-float(mx.sum(probs * log_p)))

    return {
        "text": tokenizer.decode(tokens),
        "mean_logprob": float(np.mean(logprobs)) if logprobs else 0.0,
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
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


def compute_dprime(signal_revisions, signal_total, noise_revisions, noise_total):
    """Compute d-prime with log-linear correction (Hautus, 1995)."""
    if signal_total == 0 or noise_total == 0:
        return 0.0, 0.0, 0.0, 0.0
    hr = (signal_revisions + 0.5) / (signal_total + 1)
    far = (noise_revisions + 0.5) / (noise_total + 1)
    hr = np.clip(hr, 0.001, 0.999)
    far = np.clip(far, 0.001, 0.999)
    d = norm.ppf(hr) - norm.ppf(far)
    c = -0.5 * (norm.ppf(hr) + norm.ppf(far))
    return float(d), float(c), float(hr), float(far)


def bootstrap_dprime(signal_trials, noise_trials, n_bootstrap=2000):
    """Bootstrap 95% CI for d-prime."""
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(n_bootstrap):
        sig_boot = rng.choice(signal_trials, size=len(signal_trials), replace=True)
        noi_boot = rng.choice(noise_trials, size=len(noise_trials), replace=True)
        sig_rev = sum(1 for t in sig_boot if t['did_revise'])
        noi_rev = sum(1 for t in noi_boot if t['did_revise'])
        d_boot, _, _, _ = compute_dprime(sig_rev, len(sig_boot), noi_rev, len(noi_boot))
        samples.append(d_boot)
    ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
    return float(ci_low), float(ci_high)


def prepare_items():
    """Prepare ALL matched items from DS Critique Bank (identical to sweep_fullpool.py)."""
    ds = load_dataset("allenai/DS_Critique_Bank", split="train")

    by_qid = {}
    for item in ds:
        qid = item['qid']
        if qid not in by_qid:
            by_qid[qid] = []
        by_qid[qid].append(item)

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
                            'specific_feedback': sf,
                            'flaw': mf,
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
                'qid': qid,
                'dataset': dataset_name,
                'question': question,
                'gold': gold,
                'valid_critique': valid_critique,
                'invalid_critique': invalid_critique,
            })

    return items


def run_experiment(model, tokenizer, items, model_name):
    """Run the full 2x2 experiment on all items."""
    print(f"\n  Phase 1: Initial answers ({len(items)} items)...")

    phase1 = []
    for i, item in enumerate(items):
        messages = [{"role": "user", "content": item['question'] + "\n\nAnswer with just the letter."}]
        gen = generate_with_logprobs(model, tokenizer, messages)

        labels = re.findall(r'\(([A-Z])\)', item['question'])
        if not labels:
            labels = ['A', 'B', 'C', 'D']

        answer = extract_answer(gen["text"], labels)
        correct = answer == item['gold']

        phase1.append({
            'idx': i,
            'answer': answer,
            'correct': correct,
            'logprob': gen['mean_logprob'],
            'entropy': gen['mean_entropy'],
            'response': gen['text'][:200],
        })

        if (i + 1) % 50 == 0:
            n_c = sum(1 for r in phase1 if r['correct'])
            print(f"    [{i+1}/{len(items)}] accuracy so far: {n_c}/{i+1} = {n_c/(i+1)*100:.0f}%")

    n_correct = sum(1 for r in phase1 if r['correct'])
    n_incorrect = len(phase1) - n_correct
    accuracy = n_correct / len(phase1)
    print(f"  Accuracy: {n_correct}/{len(phase1)} = {accuracy*100:.1f}%")

    print("\n  Phase 2: Critique trials...")

    trials = []
    for r in phase1:
        item = items[r['idx']]
        if r['answer'] is None:
            continue

        if r['correct']:
            ic = item['invalid_critique']
            if ic['type'] == 'false_flaw':
                critique_text = (
                    f"I think your answer may be wrong. {ic['false_reasoning']} "
                    f"Consider whether ({ic['wrong_target']}) might be more appropriate."
                )
            else:
                critique_text = (
                    f"I think your answer may be wrong. {ic['wrong_reasoning']} "
                    f"The answer should be ({ic['wrong_target']})."
                )
            expected_action = "RESIST"
        else:
            vc = item['valid_critique']
            critique_text = (
                f"I think your answer may be wrong. {vc['specific_feedback']} "
                f"The correct answer is ({item['gold']})."
            )
            expected_action = "REVISE"

        messages = [
            {"role": "user", "content": item['question'] + "\n\nAnswer with just the letter."},
            {"role": "assistant", "content": f"({r['answer']})"},
            {"role": "user", "content": critique_text + "\n\nWhat is the correct answer? Just the letter."},
        ]

        labels = re.findall(r'\(([A-Z])\)', item['question'])
        if not labels:
            labels = ['A', 'B', 'C', 'D']

        gen2 = generate_with_logprobs(model, tokenizer, messages)
        revised_answer = extract_answer(gen2["text"], labels)
        did_revise = revised_answer != r['answer']

        if expected_action == "REVISE":
            sdt = "HIT" if did_revise else "MISS"
        else:
            sdt = "FALSE_ALARM" if did_revise else "CORRECT_REJECTION"

        trials.append({
            'idx': r['idx'],
            'qid': item['qid'],
            'dataset': item['dataset'],
            'model_correct': r['correct'],
            'initial_answer': r['answer'],
            'gold': item['gold'],
            'expected_action': expected_action,
            'critique_valid': expected_action == "REVISE",
            'did_revise': did_revise,
            'revised_answer': revised_answer,
            'sdt': sdt,
            'initial_logprob': r['logprob'],
            'initial_entropy': r['entropy'],
            'revised_logprob': gen2['mean_logprob'],
            'revised_entropy': gen2['mean_entropy'],
            'critique_text': critique_text[:300],
        })

        if (len(trials)) % 100 == 0:
            print(f"    [{len(trials)} trials complete]")

    # Overall d-prime
    signal_trials = [t for t in trials if t['expected_action'] == "REVISE"]
    noise_trials = [t for t in trials if t['expected_action'] == "RESIST"]

    signal_revisions = sum(1 for t in signal_trials if t['did_revise'])
    noise_revisions = sum(1 for t in noise_trials if t['did_revise'])

    d, c, hr, far = compute_dprime(
        signal_revisions, len(signal_trials),
        noise_revisions, len(noise_trials)
    )
    ci_low, ci_high = bootstrap_dprime(signal_trials, noise_trials)

    # Logprob gap
    lp_revised = [t['initial_logprob'] for t in trials if t['did_revise']]
    lp_held = [t['initial_logprob'] for t in trials if not t['did_revise']]
    logprob_gap = float(np.mean(lp_revised) - np.mean(lp_held)) if lp_revised and lp_held else None

    # Per-dataset d-prime breakdown
    datasets_in_trials = sorted(set(t['dataset'] for t in trials))
    per_dataset = {}
    for ds_name in datasets_in_trials:
        ds_sig = [t for t in signal_trials if t['dataset'] == ds_name]
        ds_noi = [t for t in noise_trials if t['dataset'] == ds_name]
        ds_sig_rev = sum(1 for t in ds_sig if t['did_revise'])
        ds_noi_rev = sum(1 for t in ds_noi if t['did_revise'])
        ds_d, ds_c, ds_hr, ds_far = compute_dprime(ds_sig_rev, len(ds_sig), ds_noi_rev, len(ds_noi))
        ds_ci = bootstrap_dprime(ds_sig, ds_noi) if len(ds_sig) >= 5 and len(ds_noi) >= 5 else (None, None)
        per_dataset[ds_name] = {
            'd_prime': ds_d,
            'd_prime_ci95': list(ds_ci) if ds_ci[0] is not None else None,
            'hit_rate': ds_hr,
            'false_alarm_rate': ds_far,
            'criterion_c': ds_c,
            'n_signal': len(ds_sig),
            'n_noise': len(ds_noi),
            'hits': ds_sig_rev,
            'false_alarms': ds_noi_rev,
        }

    science_datasets = {'ARC-Challenge', 'ARC-Easy'}
    commonsense_datasets = {'HellaSwag', 'SocialIQa', 'CosmosQA', 'WinoGrande', 'PIQA', 'aNLI'}

    sci_sig = [t for t in signal_trials if t['dataset'] in science_datasets]
    sci_noi = [t for t in noise_trials if t['dataset'] in science_datasets]
    com_sig = [t for t in signal_trials if t['dataset'] in commonsense_datasets]
    com_noi = [t for t in noise_trials if t['dataset'] in commonsense_datasets]

    sci_d, sci_c, sci_hr, sci_far = compute_dprime(
        sum(1 for t in sci_sig if t['did_revise']), len(sci_sig),
        sum(1 for t in sci_noi if t['did_revise']), len(sci_noi)
    )
    com_d, com_c, com_hr, com_far = compute_dprime(
        sum(1 for t in com_sig if t['did_revise']), len(com_sig),
        sum(1 for t in com_noi if t['did_revise']), len(com_noi)
    )

    metrics = {
        'model': model_name,
        'architecture': 'gemma4',
        'thinking_mode': False,
        'accuracy': accuracy,
        'n_items': len(phase1),
        'n_valid_trials': len(trials),
        'n_correct': n_correct,
        'n_incorrect': n_incorrect,
        'n_signal_trials': len(signal_trials),
        'n_noise_trials': len(noise_trials),
        'd_prime': d,
        'd_prime_ci95': [ci_low, ci_high],
        'criterion_c': c,
        'hit_rate': hr,
        'false_alarm_rate': far,
        'hits': signal_revisions,
        'misses': len(signal_trials) - signal_revisions,
        'false_alarms': noise_revisions,
        'correct_rejections': len(noise_trials) - noise_revisions,
        'logprob_gap': logprob_gap,
        'per_dataset': per_dataset,
        'science_d_prime': sci_d,
        'science_n_signal': len(sci_sig),
        'science_n_noise': len(sci_noi),
        'commonsense_d_prime': com_d,
        'commonsense_n_signal': len(com_sig),
        'commonsense_n_noise': len(com_noi),
    }

    return metrics, trials


def load_qwen_baselines():
    """Load the Qwen3.5 fullpool results for comparison."""
    qwen_dir = Path("results/sweep_v3_fullpool")
    baselines = {}
    for sf in sorted(qwen_dir.glob("sweep_v3_*.json")):
        with open(sf) as f:
            data = json.load(f)
        for r in data:
            baselines[r['model']] = r
    return baselines


def main():
    print("=" * 70)
    print("METACOGNITION BENCHMARK: Gemma 4 (E4B-IT + 26B-A4B-IT)")
    print("DS Critique Bank + All Datasets + Balanced 2x2")
    print("Thinking mode suppressed for fair comparison with Qwen3.5")
    print("=" * 70)

    print("\nPreparing items from DS Critique Bank...")
    all_items = prepare_items()
    print(f"Total items with matched valid/invalid critiques: {len(all_items)}")

    by_dataset = {}
    for item in all_items:
        by_dataset.setdefault(item['dataset'], []).append(item)
    for ds_name, items_list in sorted(by_dataset.items(), key=lambda x: -len(x[1])):
        print(f"  {ds_name}: {len(items_list)}")

    items = all_items
    print(f"\nUsing all {len(items)} items.")

    # Gemma 4 models to test (IT versions)
    gemma_models = [
        ("Gemma4-E4B-IT", "google/gemma-4-E4B-it"),
        ("Gemma4-26B-A4B-IT", "google/gemma-4-26B-A4B-it"),
    ]

    all_results = []

    # Load Qwen baselines for comparison
    qwen_baselines = load_qwen_baselines()
    if qwen_baselines:
        print(f"\nLoaded Qwen3.5 baselines: {list(qwen_baselines.keys())}")

    for model_name, model_path in gemma_models:
        trial_path = RESULTS_DIR / f"trials_{model_name}.json"
        if trial_path.exists():
            print(f"\nModel {model_name}: SKIPPING (results exist at {trial_path})")
            # Load existing results
            summary_files = sorted(RESULTS_DIR.glob("sweep_gemma4_*.json"))
            for sf in summary_files:
                with open(sf) as f:
                    prior = json.load(f)
                for r in prior:
                    if r['model'] == model_name:
                        all_results.append(r)
                        print(f"  Loaded: d'={r['d_prime']:.3f}")
            continue

        print(f"\n{'='*70}")
        print(f"MODEL: {model_name} ({model_path})")
        print(f"{'='*70}")

        print("Loading model (this may download on first run)...")
        t_load = time.time()
        model, processor = vlm_load(model_path)
        tokenizer = processor.tokenizer
        print(f"Loaded in {time.time()-t_load:.1f}s")
        print(f"Chat template: {tokenizer.chat_template is not None}")

        t0 = time.time()
        metrics, trials = run_experiment(model, tokenizer, items, model_name)
        elapsed = time.time() - t0
        metrics['elapsed_s'] = elapsed

        print("\n  --- RESULTS ---")
        print(f"  Accuracy:     {metrics['accuracy']*100:.1f}% ({metrics['n_correct']}/{metrics['n_items']})")
        print(f"  Signal trials: {metrics['n_signal_trials']}  |  Noise trials: {metrics['n_noise_trials']}")
        print(f"  d-prime:      {metrics['d_prime']:.3f}  95% CI [{metrics['d_prime_ci95'][0]:.3f}, {metrics['d_prime_ci95'][1]:.3f}]")
        print(f"  Criterion c:  {metrics['criterion_c']:.3f}")
        print(f"  Hit rate:     {metrics['hit_rate']:.3f} ({metrics['hits']}/{metrics['n_signal_trials']})")
        print(f"  FA rate:      {metrics['false_alarm_rate']:.3f} ({metrics['false_alarms']}/{metrics['n_noise_trials']})")
        print(f"  Science d':   {metrics['science_d_prime']:.3f} (N_sig={metrics['science_n_signal']}, N_noi={metrics['science_n_noise']})")
        print(f"  Commonsense:  {metrics['commonsense_d_prime']:.3f} (N_sig={metrics['commonsense_n_signal']}, N_noi={metrics['commonsense_n_noise']})")
        if metrics['logprob_gap']:
            print(f"  Logprob gap:  {metrics['logprob_gap']:.4f}")
        print(f"  Time:         {elapsed:.1f}s")

        print("\n  Per-dataset breakdown:")
        for ds_name, ds_metrics in sorted(metrics['per_dataset'].items()):
            ci_str = ""
            if ds_metrics['d_prime_ci95']:
                ci_str = f" [{ds_metrics['d_prime_ci95'][0]:.2f}, {ds_metrics['d_prime_ci95'][1]:.2f}]"
            print(f"    {ds_name:20s} d'={ds_metrics['d_prime']:.3f}{ci_str}  HR={ds_metrics['hit_rate']:.2f}  FAR={ds_metrics['false_alarm_rate']:.2f}  (sig={ds_metrics['n_signal']}, noi={ds_metrics['n_noise']})")

        all_results.append(metrics)

        trial_path = RESULTS_DIR / f"trials_{model_name}.json"
        with open(trial_path, "w") as f:
            json.dump(trials, f, indent=2, default=str)

        del model, processor, tokenizer
        mx.clear_cache()

    # Comparative summary with Qwen baselines
    print(f"\n{'='*70}")
    print("CROSS-ARCHITECTURE COMPARISON: Gemma 4 vs Qwen 3.5")
    print(f"{'='*70}")

    all_for_table = []
    for name, r in sorted(qwen_baselines.items()):
        r['architecture'] = 'qwen3.5'
        all_for_table.append(r)
    all_for_table.extend(all_results)

    header = f"{'Model':<22} {'Arch':<10} {'Acc%':<7} {'N_sig':<6} {'N_noi':<6} {'d-prime':<22} {'HR':<7} {'FAR':<7} {'c':<7} {'Sci d':<8} {'Com d':<8}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in all_for_table:
        ci = f"[{r['d_prime_ci95'][0]:.2f}, {r['d_prime_ci95'][1]:.2f}]"
        arch = r.get('architecture', 'qwen3.5')
        sci_d = r.get('science_d_prime', 'N/A')
        com_d = r.get('commonsense_d_prime', 'N/A')
        sci_str = f"{sci_d:.3f}" if isinstance(sci_d, float) else str(sci_d)
        com_str = f"{com_d:.3f}" if isinstance(com_d, float) else str(com_d)
        print(f"{r['model']:<22} {arch:<10} {r['accuracy']*100:<7.1f} {r['n_signal_trials']:<6} {r['n_noise_trials']:<6} {r['d_prime']:<8.3f} {ci:<14} {r['hit_rate']:<7.3f} {r['false_alarm_rate']:<7.3f} {r['criterion_c']:<7.3f} {sci_str:<8} {com_str:<8}")

    # Save summary
    outpath = RESULTS_DIR / f"sweep_gemma4_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
