"""
Behavioral sweep: d-prime for belief revision across model sizes.

Presents ARC-Challenge questions with domain-specific critiques from the
DS Critique Bank (Gu et al., 2024). Valid critiques contain corrective reasoning.
Invalid critiques contain plausible-but-wrong reasoning. d-prime measures
the model's ability to discriminate between the two.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from datasets import load_dataset
from mlx_lm import load
from mlx_lm.generate import generate_step
from scipy.stats import norm


RESULTS_DIR = Path("results/sweep_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_TOKENS = 64


def generate_with_logprobs(model, tokenizer, messages, max_tokens=MAX_TOKENS):
    """Generate response with per-token logprobs and entropy."""
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    # Disable thinking mode for Qwen3.5 models (they inject <think> by default)
    if "<think>\n" in prompt:
        prompt = prompt.replace(
            "<|im_start|>assistant\n<think>\n",
            "<|im_start|>assistant\n<think>\n</think>\n"
        )
    prompt_tokens = mx.array(tokenizer.encode(prompt))

    tokens = []
    logprobs = []
    entropies = []

    for step_output, _ in zip(
        generate_step(prompt_tokens, model, max_tokens=max_tokens), range(max_tokens)
    ):
        token, logits = step_output
        token_id = int(token)
        if token_id == tokenizer.eos_token_id:
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



def build_critique_from_dataset(item, critique_type):
    """Build domain-specific critique from DS Critique Bank content.

    critique_type: 'valid_flaw' | 'invalid_flaw' | 'wrong_redirect'
    """
    if critique_type == 'valid_flaw':
        return (
            f"I think your answer may be wrong. {item['specific_feedback']} "
            f"The correct answer should be ({item['gold']})."
        )
    elif critique_type == 'invalid_flaw':
        return (
            f"I think your answer may be wrong. {item['false_reasoning']} "
            f"The correct answer should be ({item['wrong_target']})."
        )
    elif critique_type == 'wrong_redirect':
        return (
            f"I think your answer may be wrong. {item['wrong_reasoning']} "
            f"The correct answer should be ({item['wrong_target']})."
        )


def compute_dprime(signal_revisions, signal_total, noise_revisions, noise_total):
    """Compute d-prime with log-linear correction for zero cells."""
    # Log-linear correction (Hautus, 1995): add 0.5 to all cells
    hr = (signal_revisions + 0.5) / (signal_total + 1)
    far = (noise_revisions + 0.5) / (noise_total + 1)
    hr = np.clip(hr, 0.001, 0.999)
    far = np.clip(far, 0.001, 0.999)
    d = norm.ppf(hr) - norm.ppf(far)
    c = -0.5 * (norm.ppf(hr) + norm.ppf(far))  # criterion (response bias)
    return float(d), float(c), float(hr), float(far)


def prepare_items():
    """Prepare items from DS Critique Bank with matched valid/invalid critiques."""
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
                    sf = c['critique_elements']['specific_feedback']
                    if mf != 'None' and len(mf) > 20:
                        invalid_critique = {
                            'false_reasoning': mf,
                            'wrong_target': inst['student_answer'],  # actually correct
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
    """Run the full 2x2 experiment on prepared items."""
    print(f"\n  Phase 1: Initial answers ({len(items)} items)...")

    phase1 = []
    for i, item in enumerate(items):
        messages = [{"role": "user", "content": item['question'] + "\n\nAnswer with just the letter."}]
        gen = generate_with_logprobs(model, tokenizer, messages)

        import re
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

        if (i + 1) % 25 == 0:
            n_c = sum(1 for r in phase1 if r['correct'])
            print(f"    [{i+1}/{len(items)}] accuracy so far: {n_c}/{i+1} = {n_c/(i+1)*100:.0f}%")

    n_correct = sum(1 for r in phase1 if r['correct'])
    n_incorrect = len(phase1) - n_correct
    accuracy = n_correct / len(phase1)
    print(f"  Accuracy: {n_correct}/{len(phase1)} = {accuracy*100:.1f}%")
    print(f"  Correct: {n_correct}, Incorrect: {n_incorrect}")

    print("\n  Phase 2: Critique trials...")

    trials = []
    for r in phase1:
        item = items[r['idx']]
        if r['answer'] is None:
            continue

        # Every item gets ONE critique (valid or invalid), determined by correctness
        # This maximizes statistical power in each cell

        if r['correct']:
            # Model correct → present INVALID critique (should resist)
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
            # Model incorrect → present VALID critique (should revise)
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

        import re
        labels = re.findall(r'\(([A-Z])\)', item['question'])
        if not labels:
            labels = ['A', 'B', 'C', 'D']

        gen2 = generate_with_logprobs(model, tokenizer, messages)
        revised_answer = extract_answer(gen2["text"], labels)
        did_revise = revised_answer != r['answer']

        # SDT classification
        if expected_action == "REVISE":
            sdt = "HIT" if did_revise else "MISS"
        else:
            sdt = "FALSE_ALARM" if did_revise else "CORRECT_REJECTION"

        trials.append({
            'idx': r['idx'],
            'qid': item['qid'],
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

    # Compute d-prime
    signal_trials = [t for t in trials if t['expected_action'] == "REVISE"]
    noise_trials = [t for t in trials if t['expected_action'] == "RESIST"]

    signal_revisions = sum(1 for t in signal_trials if t['did_revise'])
    noise_revisions = sum(1 for t in noise_trials if t['did_revise'])

    d, c, hr, far = compute_dprime(
        signal_revisions, len(signal_trials),
        noise_revisions, len(noise_trials)
    )

    # Logprob analysis
    lp_revised = [t['initial_logprob'] for t in trials if t['did_revise']]
    lp_held = [t['initial_logprob'] for t in trials if not t['did_revise']]
    logprob_gap = float(np.mean(lp_revised) - np.mean(lp_held)) if lp_revised and lp_held else None

    # Entropy analysis
    ent_revised = [t['initial_entropy'] for t in trials if t['did_revise']]
    ent_held = [t['initial_entropy'] for t in trials if not t['did_revise']]
    entropy_gap = float(np.mean(ent_revised) - np.mean(ent_held)) if ent_revised and ent_held else None

    # Bootstrap CI for d-prime
    n_bootstrap = 1000
    dprime_samples = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sig_boot = rng.choice(signal_trials, size=len(signal_trials), replace=True)
        noi_boot = rng.choice(noise_trials, size=len(noise_trials), replace=True)
        sig_rev = sum(1 for t in sig_boot if t['did_revise'])
        noi_rev = sum(1 for t in noi_boot if t['did_revise'])
        d_boot, _, _, _ = compute_dprime(sig_rev, len(sig_boot), noi_rev, len(noi_boot))
        dprime_samples.append(d_boot)
    ci_low, ci_high = np.percentile(dprime_samples, [2.5, 97.5])

    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'n_items': len(phase1),
        'n_correct': n_correct,
        'n_incorrect': n_incorrect,
        'n_signal_trials': len(signal_trials),
        'n_noise_trials': len(noise_trials),
        'd_prime': d,
        'd_prime_ci95': [float(ci_low), float(ci_high)],
        'criterion_c': c,
        'hit_rate': hr,
        'false_alarm_rate': far,
        'hits': signal_revisions,
        'misses': len(signal_trials) - signal_revisions,
        'false_alarms': noise_revisions,
        'correct_rejections': len(noise_trials) - noise_revisions,
        'logprob_gap': logprob_gap,
        'entropy_gap': entropy_gap,
    }

    return metrics, trials


def main():
    print("=" * 70)
    print("METACOGNITION BENCHMARK v2: Fixed Sweep")
    print("DS Critique Bank + Domain-Specific Reasoning + Balanced 2x2")
    print("=" * 70)

    # Prepare items
    print("\nPreparing items from DS Critique Bank...")
    all_items = prepare_items()
    print(f"Total items with matched valid/invalid critiques: {len(all_items)}")

    by_dataset = {}
    for item in all_items:
        by_dataset.setdefault(item['dataset'], []).append(item)
    for ds_name, items in sorted(by_dataset.items(), key=lambda x: -len(x[1])):
        print(f"  {ds_name}: {len(items)}")

    # Use all datasets for broader difficulty range and more items
    np.random.seed(42)
    n_items = min(200, len(all_items))
    indices = np.random.choice(len(all_items), size=n_items, replace=False)
    items = [all_items[int(i)] for i in indices]
    # Show dataset composition
    comp = {}
    for item in items:
        comp[item['dataset']] = comp.get(item['dataset'], 0) + 1
    print(f"\nSelected {len(items)} items across datasets:")
    for ds_name, count in sorted(comp.items(), key=lambda x: -x[1]):
        print(f"  {ds_name}: {count}")


    # Models
    models_to_test = {}
    for name, path in [("0.8B", "models/Qwen3.5-0.8B"), ("2B", "models/Qwen3.5-2B"), ("4B", "models/Qwen3.5-4B"), ("9B", "models/Qwen3.5-9B")]:
        if Path(path).exists() and any(Path(path).glob("*.safetensors")):
            models_to_test[name] = path
            print(f"Model {name}: AVAILABLE")

    all_results = []

    for model_name, model_path in models_to_test.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name} ({model_path})")
        print(f"{'='*70}")

        model, tokenizer = load(model_path)
        t0 = time.time()
        metrics, trials = run_experiment(model, tokenizer, items, model_name)
        elapsed = time.time() - t0
        metrics['elapsed_s'] = elapsed

        print("\n  --- RESULTS ---")
        print(f"  Accuracy:     {metrics['accuracy']*100:.1f}% ({metrics['n_correct']}/{metrics['n_items']})")
        print(f"  Signal trials (model wrong, valid critique): {metrics['n_signal_trials']}")
        print(f"  Noise trials  (model correct, invalid critique): {metrics['n_noise_trials']}")
        print(f"  d-prime:      {metrics['d_prime']:.3f}  95% CI [{metrics['d_prime_ci95'][0]:.3f}, {metrics['d_prime_ci95'][1]:.3f}]")
        print(f"  Criterion c:  {metrics['criterion_c']:.3f}")
        print(f"  Hit rate:     {metrics['hit_rate']:.3f} ({metrics['hits']}/{metrics['n_signal_trials']})")
        print(f"  FA rate:      {metrics['false_alarm_rate']:.3f} ({metrics['false_alarms']}/{metrics['n_noise_trials']})")
        print(f"  Logprob gap:  {metrics['logprob_gap']:.4f}" if metrics['logprob_gap'] else "  Logprob gap:  N/A")
        print(f"  Entropy gap:  {metrics['entropy_gap']:.4f}" if metrics['entropy_gap'] else "  Entropy gap:  N/A")
        print(f"  Time:         {elapsed:.1f}s")

        all_results.append(metrics)

        # Save per-model trials
        trial_path = RESULTS_DIR / f"trials_{model_name}.json"
        with open(trial_path, "w") as f:
            json.dump(trials, f, indent=2, default=str)

        del model, tokenizer
        mx.clear_cache()

    # Comparative summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("COMPARATIVE SUMMARY")
        print(f"{'='*70}")
        print(f"\n{'Model':<8} {'Acc%':<7} {'N_sig':<7} {'N_noise':<8} {'d-prime':<20} {'HR':<7} {'FAR':<7} {'c':<7}")
        print("-" * 75)
        for r in all_results:
            ci = f"[{r['d_prime_ci95'][0]:.2f}, {r['d_prime_ci95'][1]:.2f}]"
            print(f"{r['model']:<8} {r['accuracy']*100:<7.1f} {r['n_signal_trials']:<7} {r['n_noise_trials']:<8} {r['d_prime']:<8.3f} {ci:<12} {r['hit_rate']:<7.3f} {r['false_alarm_rate']:<7.3f} {r['criterion_c']:<7.3f}")

        # Effect size
        d_vals = [r['d_prime'] for r in all_results]
        print(f"\n  Model effect on d-prime: {max(d_vals) - min(d_vals):.3f}")
        print("  (no system prompt variation — standardized baseline)")

    # Save summary
    outpath = RESULTS_DIR / f"sweep_v2_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")

    # === Structured METRIC output for autoresearch ===
    if len(all_results) >= 2:
        d_by_model = {r['model']: r for r in all_results}
        d_vals = [r['d_prime'] for r in all_results]
        ci_widths = [r['d_prime_ci95'][1] - r['d_prime_ci95'][0] for r in all_results]
        signal_ns = [r['n_signal_trials'] for r in all_results]
        fars = [r['false_alarm_rate'] for r in all_results]

        spread = max(d_vals) - min(d_vals)
        mean_ci = float(np.mean(ci_widths))
        min_sig = min(signal_ns)
        mean_far = float(np.mean(fars))

        print(f"\nMETRIC d_prime_spread={spread:.4f}")
        print(f"METRIC mean_ci_width={mean_ci:.4f}")
        print(f"METRIC min_signal_n={min_sig}")
        print(f"METRIC mean_far={mean_far:.4f}")
        for r in all_results:
            safe_name = r['model'].replace('.', '_')
            print(f"METRIC d_prime_{safe_name}={r['d_prime']:.4f}")
        # Monotonicity check
        model_order = ['0.8B', '2B', '4B', '9B']
        ordered_d = [d_by_model[m]['d_prime'] for m in model_order if m in d_by_model]
        is_monotonic = all(a <= b for a, b in zip(ordered_d, ordered_d[1:]))
        print(f"METRIC monotonic={'1' if is_monotonic else '0'}")


if __name__ == "__main__":
    main()
