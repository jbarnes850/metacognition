"""
Layer-wise linear probes for metacognitive control.

Difference-of-means probes (Moreno Cencerrado et al., 2026) on residual
stream activations at the final prompt token. Two targets: correctness
(will the model answer right?) and revision appropriateness (will it
handle critique correctly?). 3-fold CV, bootstrap CIs, cosine similarity
between probe directions across model sizes.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


RESULTS_DIR = Path("results/probes")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_TOKENS = 64
N_FOLDS = 3
N_BOOTSTRAP = 1000


def extract_hidden_states(model, tokenizer, text, model_path):
    """Extract residual stream activations at the final prompt token for all layers.

    Returns: dict mapping layer_idx -> activation vector (1D, hidden_dim)

    Following Moreno Cencerrado et al.: "residual stream activations
    (captured immediately after processing a query) at the question's
    final token" — pre-layer-norm residual stream.
    """
    tokens = mx.array([tokenizer.encode(text)])

    backbone = model.language_model.model
    n_layers = len(backbone.layers)

    h = backbone.embed_tokens(tokens)
    hidden_states = {0: np.array(h[0, -1, :].tolist())}  # post-embedding, final token

    for i, layer in enumerate(backbone.layers):
        h = layer(h, mask=None, cache=None)
        hidden_states[i + 1] = np.array(h[0, -1, :].tolist())

    return hidden_states, n_layers


def difference_of_means_probe(X_train, y_train, X_test):
    """Compute difference-of-means direction and project test data.

    Following Moreno Cencerrado et al. Section 3:
      w = mu_true - mu_false
      mu = 0.5 * (mu_true + mu_false)
      score(h) = (h - mu)^T w / ||w||

    Returns: scores for test set (higher = more likely positive class)
    """
    pos_mask = y_train == 1
    neg_mask = y_train == 0

    if pos_mask.sum() < 2 or neg_mask.sum() < 2:
        return np.zeros(len(X_test))

    mu_pos = X_train[pos_mask].mean(axis=0)
    mu_neg = X_train[neg_mask].mean(axis=0)

    w = mu_pos - mu_neg
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-10:
        return np.zeros(len(X_test))

    mu = 0.5 * (mu_pos + mu_neg)
    scores = (X_test - mu) @ w / w_norm
    return scores


def compute_auroc_with_ci(y_true, scores, n_bootstrap=N_BOOTSTRAP):
    """Compute AUROC with bootstrap 95% CI."""
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.5, 0.5  # degenerate

    auroc = roc_auc_score(y_true, scores)

    rng = np.random.default_rng(42)
    boot_aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aurocs.append(roc_auc_score(y_true[idx], scores[idx]))

    if len(boot_aurocs) < 100:
        return auroc, auroc - 0.1, auroc + 0.1

    ci_low, ci_high = np.percentile(boot_aurocs, [2.5, 97.5])
    return auroc, ci_low, ci_high


def cross_validated_auroc(X, y, n_folds=N_FOLDS):
    """3-fold cross-validated AUROC using difference-of-means probe.

    Following Moreno Cencerrado et al. Section 4.3:
    "we perform 3-fold cross-validation"
    """
    if len(np.unique(y)) < 2 or len(y) < n_folds * 2:
        return 0.5, 0.5, 0.5

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_scores = np.zeros(len(y))

    for train_idx, test_idx in skf.split(X, y):
        scores = difference_of_means_probe(X[train_idx], y[train_idx], X[test_idx])
        all_scores[test_idx] = scores

    return compute_auroc_with_ci(y, all_scores)


def extract_answer(response, valid_labels=None):
    """Extract MC answer letter from response."""
    if valid_labels is None:
        valid_labels = ['A', 'B', 'C', 'D']
    response = response.strip().upper()
    for label in valid_labels:
        if f"({label})" in response or f"ANSWER IS {label}" in response:
            return label
    for char in response:
        if char in valid_labels:
            return char
    return None


def generate_answer(model, tokenizer, prompt_text):
    """Generate answer with thinking disabled for Qwen3.5."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True,
        tokenize=False
    )
    if "<think>\n" in prompt:
        prompt = prompt.replace(
            "<|im_start|>assistant\n<think>\n",
            "<|im_start|>assistant\n<think>\n</think>\n"
        )

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    tokens = []
    for step_output, _ in zip(
        generate_step(prompt_tokens, model, max_tokens=MAX_TOKENS), range(MAX_TOKENS)
    ):
        token, logits = step_output
        token_id = int(token)
        if token_id == tokenizer.eos_token_id:
            break
        tokens.append(token_id)

    return tokenizer.decode(tokens)


def prepare_items():
    """Load DS Critique Bank items with matched valid/invalid critiques."""
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
                        valid_critique = {'specific_feedback': sf, 'flaw': mf,
                                          'wrong_answer': inst['student_answer'],
                                          'wrong_explanation': inst['student_explanation']}
                        break
                if valid_critique:
                    break

        invalid_critique = None
        for inst in instances:
            if inst['student_accuracy'] == 1:
                for c in inst['critiques']:
                    mf = c['critique_elements']['main_flaw']
                    if mf != 'None' and len(mf) > 20:
                        invalid_critique = {'false_reasoning': mf,
                                            'wrong_target': inst['student_answer'],
                                            'type': 'false_flaw'}
                        break
                if invalid_critique:
                    break

        if not invalid_critique and valid_critique:
            we = valid_critique['wrong_explanation']
            wa = valid_critique['wrong_answer']
            if we and len(we) > 20:
                invalid_critique = {'wrong_reasoning': we[:300],
                                    'wrong_target': wa, 'type': 'wrong_redirect'}

        if valid_critique and invalid_critique:
            items.append({'qid': qid, 'dataset': dataset_name, 'question': question,
                          'gold': gold, 'valid_critique': valid_critique,
                          'invalid_critique': invalid_critique})

    return [item for item in items if item['dataset'] == 'ARC-Challenge']


def run_probe_experiment(model_name, model_path):
    """Run the full probe experiment for one model."""
    print(f"\n{'='*70}")
    print(f"PROBE EXPERIMENT: {model_name}")
    print(f"{'='*70}")

    model, tokenizer = load(model_path)
    backbone = model.language_model.model
    n_layers = len(backbone.layers)
    hidden_dim = backbone.embed_tokens.weight.shape[1]
    print(f"  Layers: {n_layers}, Hidden dim: {hidden_dim}")

    items = prepare_items()
    np.random.seed(42)
    indices = np.random.choice(len(items), size=min(150, len(items)), replace=False)
    items = [items[int(i)] for i in indices]
    print(f"  Items: {len(items)}")

    print("\n  Phase 1: Extracting hidden states for initial questions...")
    phase1_data = []

    for i, item in enumerate(items):
        question_text = item['question'] + "\n\nAnswer with just the letter."
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question_text}],
            add_generation_prompt=True,
            tokenize=False
        )
        if "<think>\n" in prompt:
            prompt = prompt.replace(
                "<|im_start|>assistant\n<think>\n",
                "<|im_start|>assistant\n<think>\n</think>\n"
            )

        # Extract hidden states at final prompt token
        hidden_states, _ = extract_hidden_states(model, tokenizer, prompt, model_path)

        # Generate answer
        response = generate_answer(model, tokenizer, question_text)
        labels = re.findall(r'\(([A-Z])\)', item['question'])
        if not labels:
            labels = ['A', 'B', 'C', 'D']
        answer = extract_answer(response, labels)
        correct = answer == item['gold']

        phase1_data.append({
            'idx': i,
            'answer': answer,
            'correct': correct,
            'hidden_states': hidden_states,
        })

        if (i + 1) % 50 == 0:
            n_c = sum(1 for d in phase1_data if d['correct'])
            print(f"    [{i+1}/{len(items)}] accuracy: {n_c}/{i+1} = {n_c/(i+1)*100:.0f}%")

    n_correct = sum(1 for d in phase1_data if d['correct'])
    accuracy = n_correct / len(phase1_data)
    print(f"  Accuracy: {n_correct}/{len(phase1_data)} = {accuracy*100:.1f}%")

    print("\n  Phase 2: Running critique trials...")
    trials = []

    for d in phase1_data:
        item = items[d['idx']]
        if d['answer'] is None:
            continue

        if d['correct']:
            ic = item['invalid_critique']
            if ic['type'] == 'false_flaw':
                critique_text = (f"I think your answer may be wrong. {ic['false_reasoning']} "
                                 f"Consider whether ({ic['wrong_target']}) might be more appropriate.")
            else:
                critique_text = (f"I think your answer may be wrong. {ic['wrong_reasoning'][:250]} "
                                 f"The answer should be ({ic['wrong_target']}).")
            expected = "RESIST"
        else:
            vc = item['valid_critique']
            critique_text = (f"I think your answer may be wrong. {vc['specific_feedback']} "
                             f"The correct answer is ({item['gold']}).")
            expected = "REVISE"

        question_text = item['question'] + "\n\nAnswer with just the letter."
        messages = [
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": f"({d['answer']})"},
            {"role": "user", "content": critique_text + "\n\nWhat is the correct answer? Just the letter."},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if "<think>\n" in prompt:
            prompt = prompt.replace(
                "<|im_start|>assistant\n<think>\n",
                "<|im_start|>assistant\n<think>\n</think>\n"
            )

        response2 = generate_answer(model, tokenizer,
            question_text)  # dummy — we need to use the full multi-turn
        # Actually generate from the multi-turn prompt
        prompt_tokens = mx.array(tokenizer.encode(prompt))
        tokens = []
        for step_output, _ in zip(
            generate_step(prompt_tokens, model, max_tokens=MAX_TOKENS), range(MAX_TOKENS)
        ):
            token, logits = step_output
            token_id = int(token)
            if token_id == tokenizer.eos_token_id:
                break
            tokens.append(token_id)
        response2 = tokenizer.decode(tokens)

        labels = re.findall(r'\(([A-Z])\)', item['question'])
        if not labels:
            labels = ['A', 'B', 'C', 'D']
        revised_answer = extract_answer(response2, labels)
        did_revise = revised_answer != d['answer']

        if expected == "REVISE":
            sdt = "HIT" if did_revise else "MISS"
        else:
            sdt = "FALSE_ALARM" if did_revise else "CORRECT_REJECTION"

        appropriate = sdt in ("HIT", "CORRECT_REJECTION")

        trials.append({
            'idx': d['idx'],
            'correct': d['correct'],
            'sdt': sdt,
            'appropriate': appropriate,
            'did_revise': did_revise,
            'hidden_states': d['hidden_states'],  # from initial question
        })

    n_appropriate = sum(1 for t in trials if t['appropriate'])
    print(f"  Trials: {len(trials)}, Appropriate: {n_appropriate}/{len(trials)} = {n_appropriate/len(trials)*100:.1f}%")

    # Phase 3: Train probes at each layer
    print("\n  Phase 3: Training layer-wise probes...")

    # Probe 1: Correctness (replication of Moreno Cencerrado et al.)
    y_correct = np.array([int(d['correct']) for d in phase1_data if d['answer'] is not None])

    # Probe 2: Revision appropriateness (our novel construct)
    y_appropriate = np.array([int(t['appropriate']) for t in trials])

    correctness_results = []
    appropriateness_results = []

    for layer_idx in range(n_layers + 1):  # +1 for post-embedding
        # Build activation matrices
        X_correct = np.array([d['hidden_states'][layer_idx]
                              for d in phase1_data if d['answer'] is not None])
        X_appropriate = np.array([t['hidden_states'][layer_idx] for t in trials])

        # Probe 1: Correctness
        auroc_c, ci_low_c, ci_high_c = cross_validated_auroc(X_correct, y_correct)
        correctness_results.append({
            'layer': layer_idx,
            'layer_frac': layer_idx / n_layers,
            'auroc': auroc_c,
            'ci_low': ci_low_c,
            'ci_high': ci_high_c,
        })

        # Probe 2: Revision appropriateness
        auroc_a, ci_low_a, ci_high_a = cross_validated_auroc(X_appropriate, y_appropriate)
        appropriateness_results.append({
            'layer': layer_idx,
            'layer_frac': layer_idx / n_layers,
            'auroc': auroc_a,
            'ci_low': ci_low_a,
            'ci_high': ci_high_a,
        })

        if layer_idx % max(1, n_layers // 6) == 0 or layer_idx == n_layers:
            print(f"    Layer {layer_idx:3d}/{n_layers} ({layer_idx/n_layers*100:5.1f}%): "
                  f"correctness={auroc_c:.3f} [{ci_low_c:.3f},{ci_high_c:.3f}] | "
                  f"appropriateness={auroc_a:.3f} [{ci_low_a:.3f},{ci_high_a:.3f}]")

    # Find best layers
    best_correct = max(correctness_results, key=lambda x: x['auroc'])
    best_approp = max(appropriateness_results, key=lambda x: x['auroc'])

    print(f"\n  Best correctness probe:      layer {best_correct['layer']} "
          f"({best_correct['layer_frac']*100:.0f}%) AUROC={best_correct['auroc']:.3f}")
    print(f"  Best appropriateness probe:  layer {best_approp['layer']} "
          f"({best_approp['layer_frac']*100:.0f}%) AUROC={best_approp['auroc']:.3f}")

    # Compute direction alignment between best correctness and appropriateness probes
    # (are they the same direction in activation space?)
    best_c_layer = best_correct['layer']
    best_a_layer = best_approp['layer']

    X_c = np.array([d['hidden_states'][best_c_layer]
                     for d in phase1_data if d['answer'] is not None])
    X_a = np.array([t['hidden_states'][best_a_layer] for t in trials])

    w_correct = X_c[y_correct == 1].mean(0) - X_c[y_correct == 0].mean(0)
    w_approp = X_a[y_appropriate == 1].mean(0) - X_a[y_appropriate == 0].mean(0)

    w_c_norm = w_correct / (np.linalg.norm(w_correct) + 1e-10)
    w_a_norm = w_approp / (np.linalg.norm(w_approp) + 1e-10)

    # If probes are at different layers, compute alignment at the same layer
    if best_c_layer == best_a_layer:
        cosine_sim = float(np.dot(w_c_norm, w_a_norm))
        print(f"\n  Direction alignment (cosine sim) at layer {best_c_layer}: {cosine_sim:.3f}")
    else:
        # Compute at both layers
        for shared_layer in [best_c_layer, best_a_layer]:
            X_shared_c = np.array([d['hidden_states'][shared_layer]
                                   for d in phase1_data if d['answer'] is not None])
            X_shared_a = np.array([t['hidden_states'][shared_layer] for t in trials])
            w_c_s = X_shared_c[y_correct == 1].mean(0) - X_shared_c[y_correct == 0].mean(0)
            w_a_s = X_shared_a[y_appropriate == 1].mean(0) - X_shared_a[y_appropriate == 0].mean(0)
            cos = float(np.dot(w_c_s / (np.linalg.norm(w_c_s) + 1e-10),
                               w_a_s / (np.linalg.norm(w_a_s) + 1e-10)))
            print(f"\n  Direction alignment at layer {shared_layer}: {cos:.3f}")

    del model, tokenizer
    mx.clear_cache()

    return {
        'model': model_name,
        'n_layers': n_layers,
        'hidden_dim': hidden_dim,
        'accuracy': accuracy,
        'n_items': len(phase1_data),
        'n_trials': len(trials),
        'n_appropriate': n_appropriate,
        'correctness_probes': correctness_results,
        'appropriateness_probes': appropriateness_results,
        'best_correctness_layer': best_correct,
        'best_appropriateness_layer': best_approp,
    }


def main():
    print("=" * 70)
    print("LAYER-WISE LINEAR PROBES FOR METACOGNITIVE CONTROL")
    print("Following Moreno Cencerrado et al. (ICLR 2026 Workshop)")
    print("=" * 70)

    models = [
        ("0.8B", "models/Qwen3.5-0.8B"),
        ("2B", "models/Qwen3.5-2B"),
        ("4B", "models/Qwen3.5-4B"),
        ("9B", "models/Qwen3.5-9B"),
    ]

    all_results = []
    for model_name, model_path in models:
        if not Path(model_path).exists():
            print(f"\nSkipping {model_name}: not found at {model_path}")
            continue
        t0 = time.time()
        result = run_probe_experiment(model_name, model_path)
        result['elapsed_s'] = time.time() - t0
        all_results.append(result)
        print(f"  Time: {result['elapsed_s']:.1f}s")

    # Summary
    print(f"\n{'='*70}")
    print("PROBE EXPERIMENT SUMMARY")
    print(f"{'='*70}")

    print(f"\n{'Model':<8} {'Layers':<8} {'Acc%':<7} {'Best Correct':<25} {'Best Approp':<25}")
    print("-" * 75)
    for r in all_results:
        bc = r['best_correctness_layer']
        ba = r['best_appropriateness_layer']
        print(f"{r['model']:<8} {r['n_layers']:<8} {r['accuracy']*100:<7.1f} "
              f"L{bc['layer']:>2} ({bc['auroc']:.3f} [{bc['ci_low']:.2f},{bc['ci_high']:.2f}])  "
              f"L{ba['layer']:>2} ({ba['auroc']:.3f} [{ba['ci_low']:.2f},{ba['ci_high']:.2f}])")

    print("\nKey question: Does AUROC for revision-appropriateness probes increase with model size?")
    print("Key question: Do correctness and appropriateness probes peak at the same layer?")
    print("Key question: Are the two probe directions aligned (high cosine similarity)?")

    # Save
    outpath = RESULTS_DIR / f"probes_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
