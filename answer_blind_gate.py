"""
Answer-blind gate experiment: does d' survive removing the explicit answer
letter from critique templates?

Design:
- Within-item comparison: each trial runs BOTH answer-aware and answer-blind
  critique on the same item + same initial answer.
- Answer-aware critique: "{reasoning}. The correct answer is ({X})."
- Answer-blind critique: "{reasoning}." (no answer letter)
- Reports d' for each {model × condition}.
- The critical test: if answer-blind d' collapses to ~0, the current signal
  is answer-replacement compliance, not reasoning evaluation.
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


RESULTS_DIR = Path("results/answer_blind_gate")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_TOKENS = 64


def generate_response(model, tokenizer, messages, max_tokens=MAX_TOKENS):
    """Generate response text from chat messages."""
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    # Disable thinking mode for Qwen3.5
    if "<think>\n" in prompt:
        prompt = prompt.replace(
            "<|im_start|>assistant\n<think>\n",
            "<|im_start|>assistant\n<think>\n</think>\n"
        )
    prompt_tokens = mx.array(tokenizer.encode(prompt))

    tokens = []
    for step_output, _ in zip(
        generate_step(prompt_tokens, model, max_tokens=max_tokens), range(max_tokens)
    ):
        token, logits = step_output
        token_id = int(token)
        if token_id == tokenizer.eos_token_id:
            break
        tokens.append(token_id)

    return tokenizer.decode(tokens)


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
    """Compute d-prime with log-linear correction."""
    hr = (signal_revisions + 0.5) / (signal_total + 1)
    far = (noise_revisions + 0.5) / (noise_total + 1)
    hr = np.clip(hr, 0.001, 0.999)
    far = np.clip(far, 0.001, 0.999)
    d = norm.ppf(hr) - norm.ppf(far)
    c = -0.5 * (norm.ppf(hr) + norm.ppf(far))
    return float(d), float(c), float(hr), float(far)


def bootstrap_dprime(signal_trials, noise_trials, n_bootstrap=1000):
    """Bootstrap 95% CI for d-prime."""
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(n_bootstrap):
        sig_boot = rng.choice(signal_trials, size=len(signal_trials), replace=True)
        noi_boot = rng.choice(noise_trials, size=len(noise_trials), replace=True)
        sig_rev = sum(1 for t in sig_boot if t['did_revise'])
        noi_rev = sum(1 for t in noi_boot if t['did_revise'])
        d, _, _, _ = compute_dprime(sig_rev, len(sig_boot), noi_rev, len(noi_boot))
        samples.append(d)
    return np.percentile(samples, [2.5, 97.5])


def prepare_items():
    """Prepare items with matched valid/invalid critiques.

    Returns items with BOTH answer-aware and answer-blind critique variants.
    """
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

        # --- Valid critique (from incorrect student answers) ---
        valid_critique = None
        for inst in instances:
            if inst['student_accuracy'] != 0:
                continue
            for c in inst['critiques']:
                sf = c['critique_elements']['specific_feedback']
                mf = c['critique_elements']['main_flaw']
                if sf != 'None' and sf and len(sf) > 30 and mf != 'None':
                    valid_critique = {
                        'specific_feedback': sf,
                        'flaw': mf,
                        'wrong_answer': inst['student_answer'],
                    }
                    break
            if valid_critique:
                break

        # --- Invalid critique (from correct student answers — false flaw) ---
        invalid_critique = None
        for inst in instances:
            if inst['student_accuracy'] != 1:
                continue
            for c in inst['critiques']:
                mf = c['critique_elements']['main_flaw']
                if mf != 'None' and mf and len(mf) > 20:
                    invalid_critique = {
                        'false_reasoning': mf,
                        'type': 'false_flaw',
                    }
                    break
            if invalid_critique:
                break

        # Fallback: use wrong student explanation as misleading reasoning
        if not invalid_critique and valid_critique:
            for inst in instances:
                if inst['student_accuracy'] == 0:
                    we = inst['student_explanation']
                    wa = inst['student_answer']
                    if we and len(we) > 20:
                        invalid_critique = {
                            'false_reasoning': we[:300],
                            'wrong_target': wa,
                            'type': 'wrong_redirect',
                        }
                        break

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


def build_critique_text(item, critique_type, answer_mode):
    """Build critique text for a given condition.

    critique_type: 'valid' or 'invalid'
    answer_mode: 'aware' or 'blind'
    """
    if critique_type == 'valid':
        vc = item['valid_critique']
        reasoning = vc['specific_feedback']
        if answer_mode == 'aware':
            return (
                f"I think your answer may be wrong. {reasoning} "
                f"The correct answer is ({item['gold']})."
            )
        else:  # blind
            return (
                f"I think your answer may be wrong. {reasoning}"
            )

    else:  # invalid
        ic = item['invalid_critique']
        reasoning = ic['false_reasoning']
        if answer_mode == 'aware':
            if ic['type'] == 'false_flaw':
                # For false flaw: suggest a wrong answer
                # Pick any wrong answer from the question options
                import re
                labels = re.findall(r'\(([A-Z])\)', item['question'])
                wrong_labels = [label for label in labels if label != item['gold']]
                wrong_target = wrong_labels[0] if wrong_labels else 'B'
                return (
                    f"I think your answer may be wrong. {reasoning} "
                    f"Consider whether ({wrong_target}) might be more appropriate."
                )
            else:  # wrong_redirect
                return (
                    f"I think your answer may be wrong. {reasoning} "
                    f"The answer should be ({ic['wrong_target']})."
                )
        else:  # blind
            return (
                f"I think your answer may be wrong. {reasoning}"
            )


def run_gate_experiment(model, tokenizer, items, model_name):
    """Run the answer-blind gate experiment for one model.

    Phase 1: model answers each item.
    Phase 2: each item gets BOTH answer-aware and answer-blind critique.
    """
    import re

    print(f"\n  Phase 1: Initial answers ({len(items)} items)...")
    phase1 = []
    for i, item in enumerate(items):
        messages = [{"role": "user", "content": item['question'] + "\n\nAnswer with just the letter."}]
        response = generate_response(model, tokenizer, messages)

        labels = re.findall(r'\(([A-Z])\)', item['question'])
        if not labels:
            labels = ['A', 'B', 'C', 'D']

        answer = extract_answer(response, labels)
        correct = answer == item['gold']

        phase1.append({
            'idx': i,
            'answer': answer,
            'correct': correct,
            'labels': labels,
        })

        if (i + 1) % 50 == 0:
            n_c = sum(1 for r in phase1 if r['correct'])
            print(f"    [{i+1}/{len(items)}] accuracy: {n_c}/{i+1} = {n_c/(i+1)*100:.0f}%")

    n_correct = sum(1 for r in phase1 if r['correct'])
    accuracy = n_correct / len(phase1)
    print(f"  Accuracy: {n_correct}/{len(phase1)} = {accuracy*100:.1f}%")

    # Phase 2: Both conditions on each item
    print("\n  Phase 2: Paired critique trials (answer-aware + answer-blind)...")

    results = {'aware': [], 'blind': []}

    for r in phase1:
        item = items[r['idx']]
        if r['answer'] is None:
            continue

        # Determine critique type based on model correctness
        if r['correct']:
            critique_type = 'invalid'  # should resist
            expected_action = 'RESIST'
        else:
            critique_type = 'valid'  # should revise
            expected_action = 'REVISE'

        # Run BOTH conditions on this item
        for answer_mode in ['aware', 'blind']:
            critique_text = build_critique_text(item, critique_type, answer_mode)

            messages = [
                {"role": "user", "content": item['question'] + "\n\nAnswer with just the letter."},
                {"role": "assistant", "content": f"({r['answer']})"},
                {"role": "user", "content": critique_text + "\n\nWhat is the correct answer? Just the letter."},
            ]

            response = generate_response(model, tokenizer, messages)
            revised_answer = extract_answer(response, r['labels'])
            did_revise = revised_answer != r['answer']

            if expected_action == 'REVISE':
                sdt = 'HIT' if did_revise else 'MISS'
            else:
                sdt = 'FALSE_ALARM' if did_revise else 'CORRECT_REJECTION'

            results[answer_mode].append({
                'idx': r['idx'],
                'qid': item['qid'],
                'dataset': item['dataset'],
                'model_correct': r['correct'],
                'initial_answer': r['answer'],
                'gold': item['gold'],
                'expected_action': expected_action,
                'critique_type': critique_type,
                'answer_mode': answer_mode,
                'did_revise': did_revise,
                'revised_answer': revised_answer,
                'sdt': sdt,
                'critique_text': critique_text[:400],
            })

    # Compute d' for each condition
    condition_metrics = {}
    for mode in ['aware', 'blind']:
        trials = results[mode]
        signal = [t for t in trials if t['expected_action'] == 'REVISE']
        noise = [t for t in trials if t['expected_action'] == 'RESIST']

        sig_rev = sum(1 for t in signal if t['did_revise'])
        noi_rev = sum(1 for t in noise if t['did_revise'])

        d, c, hr, far = compute_dprime(sig_rev, len(signal), noi_rev, len(noise))
        ci = bootstrap_dprime(signal, noise)

        condition_metrics[mode] = {
            'mode': mode,
            'd_prime': d,
            'criterion_c': c,
            'hit_rate': hr,
            'false_alarm_rate': far,
            'd_prime_ci95': [float(ci[0]), float(ci[1])],
            'n_signal': len(signal),
            'n_noise': len(noise),
            'hits': sig_rev,
            'false_alarms': noi_rev,
        }

    # Within-item comparison: on how many items did behavior differ?
    n_differ = 0
    n_total = 0
    for aware_t, blind_t in zip(results['aware'], results['blind']):
        assert aware_t['idx'] == blind_t['idx']
        n_total += 1
        if aware_t['did_revise'] != blind_t['did_revise']:
            n_differ += 1

    return {
        'model': model_name,
        'accuracy': accuracy,
        'n_items': len(phase1),
        'n_correct': n_correct,
        'n_trials_per_condition': len(results['aware']),
        'conditions': condition_metrics,
        'n_differ_between_conditions': n_differ,
        'n_total_paired': n_total,
        'pct_differ': n_differ / n_total * 100 if n_total > 0 else 0,
    }, results


def main():
    print("=" * 70)
    print("ANSWER-BLIND GATE EXPERIMENT")
    print("Critical test: does d' survive removing the answer letter?")
    print("=" * 70)

    # Prepare items
    print("\nPreparing items from DS Critique Bank...")
    all_items = prepare_items()
    print(f"Total items with matched pairs: {len(all_items)}")

    # Use ARC-Challenge for main comparison (matches original report)
    arc_items = [it for it in all_items if it['dataset'] == 'ARC-Challenge']
    print(f"ARC-Challenge items: {len(arc_items)}")

    # Sample
    np.random.seed(42)
    n_items = min(150, len(arc_items))
    indices = np.random.choice(len(arc_items), size=n_items, replace=False)
    items = [arc_items[int(i)] for i in indices]
    print(f"Selected {n_items} items")

    # Models: 0.8B, 2B, 4B (skip 9B for speed — cell imbalance at 89% accuracy)
    models_to_test = {}
    for name, path in [
        ("0.8B", "models/Qwen3.5-0.8B"),
        ("2B", "models/Qwen3.5-2B"),
        ("4B", "models/Qwen3.5-4B"),
    ]:
        if Path(path).exists() and any(Path(path).glob("*.safetensors")):
            models_to_test[name] = path
            print(f"Model {name}: AVAILABLE")

    all_results = []

    for model_name, model_path in models_to_test.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")

        model, tokenizer = load(model_path)
        t0 = time.time()
        metrics, trials = run_gate_experiment(model, tokenizer, items, model_name)
        elapsed = time.time() - t0
        metrics['elapsed_s'] = elapsed

        # Print results
        print("\n  --- RESULTS ---")
        print(f"  Accuracy: {metrics['accuracy']*100:.1f}%")
        for mode in ['aware', 'blind']:
            cm = metrics['conditions'][mode]
            print(f"\n  [{mode.upper()}]")
            print(f"    d':     {cm['d_prime']:.3f}  CI [{cm['d_prime_ci95'][0]:.3f}, {cm['d_prime_ci95'][1]:.3f}]")
            print(f"    HR:     {cm['hit_rate']:.3f} ({cm['hits']}/{cm['n_signal']})")
            print(f"    FAR:    {cm['false_alarm_rate']:.3f} ({cm['false_alarms']}/{cm['n_noise']})")
            print(f"    c:      {cm['criterion_c']:.3f}")

        d_aware = metrics['conditions']['aware']['d_prime']
        d_blind = metrics['conditions']['blind']['d_prime']
        retention = (d_blind / d_aware * 100) if d_aware > 0 else 0
        print(f"\n  d' retention (blind/aware): {retention:.1f}%")
        print(f"  Items where behavior differed: {metrics['n_differ_between_conditions']}/{metrics['n_total_paired']} ({metrics['pct_differ']:.1f}%)")
        print(f"  Time: {elapsed:.1f}s")

        all_results.append(metrics)

        # Save per-model trials
        trial_path = RESULTS_DIR / f"trials_{model_name}.json"
        with open(trial_path, "w") as f:
            json.dump({
                'aware': trials['aware'],
                'blind': trials['blind'],
            }, f, indent=2, default=str)

        del model, tokenizer
        mx.clear_cache()

    # === SUMMARY ===
    if all_results:
        print(f"\n{'='*70}")
        print("ANSWER-BLIND GATE SUMMARY")
        print(f"{'='*70}")

        print(f"\n{'Model':<8} {'Acc%':<7} {'d_aware':<9} {'d_blind':<9} {'retention':<10} {'HR_aw':<7} {'HR_bl':<7} {'FAR_aw':<8} {'FAR_bl':<8}")
        print("-" * 80)
        for r in all_results:
            aw = r['conditions']['aware']
            bl = r['conditions']['blind']
            ret = (bl['d_prime'] / aw['d_prime'] * 100) if aw['d_prime'] > 0 else 0
            print(f"{r['model']:<8} {r['accuracy']*100:<7.1f} {aw['d_prime']:<9.3f} {bl['d_prime']:<9.3f} {ret:<10.1f}% {aw['hit_rate']:<7.3f} {bl['hit_rate']:<7.3f} {aw['false_alarm_rate']:<8.3f} {bl['false_alarm_rate']:<8.3f}")

        # d' spread
        d_aware_vals = [r['conditions']['aware']['d_prime'] for r in all_results]
        d_blind_vals = [r['conditions']['blind']['d_prime'] for r in all_results]
        spread_aware = max(d_aware_vals) - min(d_aware_vals)
        spread_blind = max(d_blind_vals) - min(d_blind_vals)

        print(f"\n  d' spread (aware):  {spread_aware:.3f}")
        print(f"  d' spread (blind):  {spread_blind:.3f}")
        print(f"  Spread retention:   {(spread_blind/spread_aware*100) if spread_aware > 0 else 0:.1f}%")

        # Verdict
        mean_retention = np.mean([
            (r['conditions']['blind']['d_prime'] / r['conditions']['aware']['d_prime'] * 100)
            if r['conditions']['aware']['d_prime'] > 0 else 0
            for r in all_results
        ])

        print(f"\n  Mean d' retention: {mean_retention:.1f}%")
        if mean_retention > 50:
            print("  VERDICT: CONSTRUCT IS SOUND — d' survives answer removal")
            print("  The signal is reasoning evaluation, not answer-replacement compliance.")
        elif mean_retention > 20:
            print("  VERDICT: PARTIAL — some signal from reasoning, but answer letter contributes substantially")
            print("  Critique design needs improvement for answer-blind condition.")
        else:
            print("  VERDICT: CONSTRUCT FAILS — d' collapses without answer letter")
            print("  Current signal is primarily answer-replacement compliance.")

    # Save summary
    outpath = RESULTS_DIR / f"gate_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
