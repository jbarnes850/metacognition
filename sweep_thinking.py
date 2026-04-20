"""
Thinking-mode ablation: d-prime with chain-of-thought enabled.

Same protocol as sweep_fullpool.py but with Qwen3.5's native thinking
mode enabled (enable_thinking=True). Compares metacognitive control
under realistic deployment conditions vs the base decision process.

Runs only 4B and 9B (thinking quality is poor at 0.8B/2B).
"""

import json
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from datasets import load_dataset
from mlx_lm import load
from mlx_lm.generate import generate_step
from scipy.stats import norm


RESULTS_DIR = Path("results/sweep_v3_thinking")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Higher token budget for thinking chains
MAX_TOKENS_THINKING = 1024
MAX_TOKENS_ANSWER_ONLY = 64


def generate_with_thinking(model, tokenizer, messages, max_tokens=MAX_TOKENS_THINKING):
    """Generate response with thinking enabled, capture logprobs on answer tokens only."""
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
        enable_thinking=True
    )
    prompt_tokens = mx.array(tokenizer.encode(prompt))

    tokens = []
    all_logprobs = []
    all_entropies = []

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
        all_logprobs.append(float(log_p[token_id]))
        all_entropies.append(-float(mx.sum(probs * log_p)))

    full_text = tokenizer.decode(tokens)

    # Split into thinking and answer portions
    if '</think>' in full_text:
        think_part, answer_part = full_text.split('</think>', 1)
        # Find the token index where </think> ends
        think_text_tokens = tokenizer.encode(think_part + '</think>', add_special_tokens=False)
        answer_start_idx = len(think_text_tokens)
        answer_logprobs = all_logprobs[answer_start_idx:]
        answer_entropies = all_entropies[answer_start_idx:]
        thinking_tokens = answer_start_idx
    else:
        # Thinking didn't close within token budget
        answer_part = full_text
        think_part = ""
        answer_logprobs = all_logprobs
        answer_entropies = all_entropies
        thinking_tokens = 0

    return {
        "text": answer_part.strip(),
        "full_text": full_text,
        "think_text": think_part,
        "thinking_tokens": thinking_tokens,
        "total_tokens": len(tokens),
        "mean_logprob": float(np.mean(answer_logprobs)) if answer_logprobs else 0.0,
        "mean_entropy": float(np.mean(answer_entropies)) if answer_entropies else 0.0,
        "think_closed": '</think>' in full_text,
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
    """Prepare ALL matched items from DS Critique Bank."""
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
    """Run the full 2x2 experiment with thinking enabled."""
    print(f"\n  Phase 1: Initial answers with thinking ({len(items)} items)...")

    phase1 = []
    think_stats = {'closed': 0, 'unclosed': 0, 'total_think_tokens': 0}

    for i, item in enumerate(items):
        messages = [{"role": "user", "content": item['question'] + "\n\nAnswer with just the letter."}]
        gen = generate_with_thinking(model, tokenizer, messages)

        labels = re.findall(r'\(([A-Z])\)', item['question'])
        if not labels:
            labels = ['A', 'B', 'C', 'D']

        answer = extract_answer(gen["text"], labels)
        correct = answer == item['gold']

        if gen['think_closed']:
            think_stats['closed'] += 1
        else:
            think_stats['unclosed'] += 1
        think_stats['total_think_tokens'] += gen['thinking_tokens']

        phase1.append({
            'idx': i,
            'answer': answer,
            'correct': correct,
            'logprob': gen['mean_logprob'],
            'entropy': gen['mean_entropy'],
            'response': gen['text'][:200],
            'think_closed': gen['think_closed'],
            'thinking_tokens': gen['thinking_tokens'],
        })

        if (i + 1) % 50 == 0:
            n_c = sum(1 for r in phase1 if r['correct'])
            avg_think = think_stats['total_think_tokens'] / (i + 1)
            print(f"    [{i+1}/{len(items)}] acc={n_c/(i+1)*100:.0f}% think_closed={think_stats['closed']}/{i+1} avg_think_tok={avg_think:.0f}")

    n_correct = sum(1 for r in phase1 if r['correct'])
    accuracy = n_correct / len(phase1)
    print(f"  Accuracy: {n_correct}/{len(phase1)} = {accuracy*100:.1f}%")
    print(f"  Think closed: {think_stats['closed']}/{len(phase1)}")
    print(f"  Avg thinking tokens: {think_stats['total_think_tokens']/len(phase1):.0f}")

    print("\n  Phase 2: Critique trials with thinking...")

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

        gen2 = generate_with_thinking(model, tokenizer, messages)
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
            'revised_think_closed': gen2['think_closed'],
            'revised_thinking_tokens': gen2['thinking_tokens'],
            'critique_text': critique_text[:300],
        })

        if len(trials) % 100 == 0:
            print(f"    [{len(trials)} trials complete]")

    # d-prime computation (same as fullpool)
    signal_trials = [t for t in trials if t['expected_action'] == "REVISE"]
    noise_trials = [t for t in trials if t['expected_action'] == "RESIST"]

    signal_revisions = sum(1 for t in signal_trials if t['did_revise'])
    noise_revisions = sum(1 for t in noise_trials if t['did_revise'])

    d, c, hr, far = compute_dprime(
        signal_revisions, len(signal_trials),
        noise_revisions, len(noise_trials)
    )
    ci_low, ci_high = bootstrap_dprime(signal_trials, noise_trials)

    logprob_gap = None
    lp_revised = [t['initial_logprob'] for t in trials if t['did_revise']]
    lp_held = [t['initial_logprob'] for t in trials if not t['did_revise']]
    if lp_revised and lp_held:
        logprob_gap = float(np.mean(lp_revised) - np.mean(lp_held))

    # Per-dataset breakdown
    science_datasets = {'ARC-Challenge', 'ARC-Easy'}
    sci_sig = [t for t in signal_trials if t['dataset'] in science_datasets]
    sci_noi = [t for t in noise_trials if t['dataset'] in science_datasets]
    com_sig = [t for t in signal_trials if t['dataset'] not in science_datasets]
    com_noi = [t for t in noise_trials if t['dataset'] not in science_datasets]

    sci_d, _, _, _ = compute_dprime(
        sum(1 for t in sci_sig if t['did_revise']), len(sci_sig),
        sum(1 for t in sci_noi if t['did_revise']), len(sci_noi)
    )
    com_d, _, _, _ = compute_dprime(
        sum(1 for t in com_sig if t['did_revise']), len(com_sig),
        sum(1 for t in com_noi if t['did_revise']), len(com_noi)
    )

    metrics = {
        'model': model_name,
        'thinking_mode': True,
        'accuracy': accuracy,
        'n_items': len(phase1),
        'n_valid_trials': len(trials),
        'n_correct': n_correct,
        'n_incorrect': len(phase1) - n_correct,
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
        'science_d_prime': sci_d,
        'commonsense_d_prime': com_d,
        'think_closed_rate_phase1': think_stats['closed'] / len(phase1),
        'avg_thinking_tokens_phase1': think_stats['total_think_tokens'] / len(phase1),
    }

    return metrics, trials


def main():
    print("=" * 70)
    print("METACOGNITION BENCHMARK v3: Thinking-Mode Ablation")
    print("DS Critique Bank + Full Pool + enable_thinking=True")
    print("=" * 70)

    print("\nPreparing items from DS Critique Bank...")
    all_items = prepare_items()
    print(f"Total items: {len(all_items)}")

    items = all_items
    print(f"Using all {len(items)} items.")

    # Only 4B and 9B (thinking quality insufficient at smaller scales)
    models_to_test = {}
    for name, path in [("4B", "models/Qwen3.5-4B"), ("9B", "models/Qwen3.5-9B")]:
        if Path(path).exists() and any(Path(path).glob("*.safetensors")):
            models_to_test[name] = path
            print(f"Model {name}: AVAILABLE")

    all_results = []

    for model_name, model_path in models_to_test.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name} (thinking ON)")
        print(f"{'='*70}")

        model, tokenizer = load(model_path)
        t0 = time.time()
        metrics, trials = run_experiment(model, tokenizer, items, model_name)
        elapsed = time.time() - t0
        metrics['elapsed_s'] = elapsed

        print("\n  --- RESULTS (THINKING ON) ---")
        print(f"  Accuracy:     {metrics['accuracy']*100:.1f}%")
        print(f"  Signal: {metrics['n_signal_trials']}  |  Noise: {metrics['n_noise_trials']}")
        print(f"  d-prime:      {metrics['d_prime']:.3f}  95% CI [{metrics['d_prime_ci95'][0]:.3f}, {metrics['d_prime_ci95'][1]:.3f}]")
        print(f"  Hit rate:     {metrics['hit_rate']:.3f}  |  FA rate: {metrics['false_alarm_rate']:.3f}")
        print(f"  Science d':   {metrics['science_d_prime']:.3f}")
        print(f"  Commonsense:  {metrics['commonsense_d_prime']:.3f}")
        print(f"  Think closed: {metrics['think_closed_rate_phase1']*100:.0f}%")
        print(f"  Avg think tok: {metrics['avg_thinking_tokens_phase1']:.0f}")
        print(f"  Time:         {elapsed:.1f}s")

        all_results.append(metrics)

        trial_path = RESULTS_DIR / f"trials_{model_name}_thinking.json"
        with open(trial_path, "w") as f:
            json.dump(trials, f, indent=2, default=str)

        del model, tokenizer
        mx.clear_cache()

    # Save summary
    outpath = RESULTS_DIR / f"sweep_v3_thinking_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")

    for r in all_results:
        safe_name = r['model'].replace('.', '_')
        print(f"METRIC d_prime_thinking_{safe_name}={r['d_prime']:.4f}")
        print(f"METRIC accuracy_thinking_{safe_name}={r['accuracy']:.4f}")


if __name__ == "__main__":
    main()
