"""
Build the V1 item registry for the metacognition benchmark.

Curates ~150 items from DS Critique Bank with:
- Answer-blind valid and invalid critique variants
- Answer-aware variants for comparison condition
- Quality filtering on critique length and specificity
- Balanced dataset composition

Output: registry/v1_items.json (the locked item set)
"""

import json
import re
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
from datasets import load_dataset


REGISTRY_DIR = Path("registry")
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


def build_registry():
    """Build the V1 item registry."""
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

        # Extract answer options from question
        labels = re.findall(r'\(([A-Z])\)', question)
        if not labels:
            labels = ['A', 'B', 'C', 'D']

        # --- Collect valid critiques ---
        valid_critiques = []
        for inst in instances:
            if inst['student_accuracy'] != 0:
                continue
            for c in inst['critiques']:
                sf = c['critique_elements']['specific_feedback']
                mf = c['critique_elements']['main_flaw']
                if sf == 'None' or not sf or len(sf) < 30 or mf == 'None':
                    continue

                sf_upper = sf.upper()
                has_gold = bool(re.search(
                    rf'\({gold}\)|answer\s+(is\s+)?{gold}|option\s+\(?{gold}\)?|correct\s+answer.*\(?{gold}\)?',
                    sf_upper
                ))

                cid = hashlib.md5(sf[:100].encode()).hexdigest()[:8]
                valid_critiques.append({
                    'critique_id': f'v_{qid}_{cid}',
                    'type': 'valid',
                    'reasoning': sf,
                    'flaw_description': mf,
                    'answer_blind': not has_gold,
                    'source': 'ds_critique_bank_incorrect_student',
                    'student_answer': inst['student_answer'],
                    'char_len': len(sf),
                })

        # --- Collect invalid critiques ---
        invalid_critiques = []
        for inst in instances:
            if inst['student_accuracy'] != 1:
                continue
            for c in inst['critiques']:
                mf = c['critique_elements']['main_flaw']
                if mf == 'None' or not mf or len(mf) < 20:
                    continue

                has_letter = bool(re.search(
                    r'\([A-Z]\)|answer\s+(is\s+)?[A-Z]|option\s+\(?[A-Z]\)?',
                    mf.upper()
                ))

                cid = hashlib.md5(mf[:100].encode()).hexdigest()[:8]
                invalid_critiques.append({
                    'critique_id': f'i_{qid}_{cid}',
                    'type': 'invalid',
                    'reasoning': mf,
                    'answer_blind': not has_letter,
                    'source': 'ds_critique_bank_false_flaw',
                    'char_len': len(mf),
                })

        # Fallback: wrong student explanation as invalid
        if not invalid_critiques:
            for inst in instances:
                if inst['student_accuracy'] == 0:
                    we = inst['student_explanation']
                    wa = inst['student_answer']
                    if we and len(we) > 20:
                        cid = hashlib.md5(we[:100].encode()).hexdigest()[:8]
                        invalid_critiques.append({
                            'critique_id': f'i_{qid}_{cid}',
                            'type': 'invalid',
                            'reasoning': we[:300],
                            'answer_blind': True,
                            'source': 'ds_critique_bank_wrong_redirect',
                            'wrong_target': wa,
                            'char_len': len(we[:300]),
                        })
                        break

        if not valid_critiques or not invalid_critiques:
            continue

        # Check if we have answer-blind variants of both
        blind_valid = [c for c in valid_critiques if c['answer_blind']]
        blind_invalid = [c for c in invalid_critiques if c['answer_blind']]
        has_blind_pair = len(blind_valid) > 0 and len(blind_invalid) > 0

        items.append({
            'item_id': qid,
            'source_dataset': dataset_name,
            'question': question,
            'answer_options': labels,
            'gold_answer': gold,
            'valid_critiques': valid_critiques,
            'invalid_critiques': invalid_critiques,
            'has_blind_pair': has_blind_pair,
            'n_valid': len(valid_critiques),
            'n_invalid': len(invalid_critiques),
        })

    return items


def select_v1_items(items, target_n=150, seed=42):
    """Select items for V1 registry.

    Selection criteria:
    1. Must have answer-blind pair (valid + invalid)
    2. Prefer items with longer, more specific critiques
    3. Balance across datasets proportionally
    4. Prefer items with multiple critique variants
    """
    # Filter to items with blind pairs
    eligible = [it for it in items if it['has_blind_pair']]
    print(f"Eligible items (have blind pair): {len(eligible)}")

    # Score each item by quality
    for it in eligible:
        blind_valid = [c for c in it['valid_critiques'] if c['answer_blind']]
        blind_invalid = [c for c in it['invalid_critiques'] if c['answer_blind']]

        # Quality score: average critique length + bonus for multiple variants
        avg_valid_len = np.mean([c['char_len'] for c in blind_valid])
        avg_invalid_len = np.mean([c['char_len'] for c in blind_invalid])
        variant_bonus = min(len(blind_valid), 3) * 10 + min(len(blind_invalid), 3) * 10

        it['quality_score'] = avg_valid_len + avg_invalid_len + variant_bonus

    # Group by dataset
    by_dataset = {}
    for it in eligible:
        by_dataset.setdefault(it['source_dataset'], []).append(it)

    # Allocate proportionally, minimum 10 per dataset if available
    total_eligible = len(eligible)
    selected = []
    rng = np.random.default_rng(seed)

    for ds_name, ds_items in sorted(by_dataset.items()):
        proportion = len(ds_items) / total_eligible
        n_alloc = max(10, int(target_n * proportion))
        n_alloc = min(n_alloc, len(ds_items))

        # Sort by quality, take top n_alloc
        ds_items.sort(key=lambda x: x['quality_score'], reverse=True)
        selected.extend(ds_items[:n_alloc])

    # If we have too many, trim from largest datasets
    if len(selected) > target_n:
        rng.shuffle(selected)
        selected = selected[:target_n]

    # If too few, add more from any dataset
    if len(selected) < target_n:
        selected_ids = {it['item_id'] for it in selected}
        remaining = [it for it in eligible if it['item_id'] not in selected_ids]
        remaining.sort(key=lambda x: x['quality_score'], reverse=True)
        needed = target_n - len(selected)
        selected.extend(remaining[:needed])

    return selected


def build_registry_entry(item):
    """Build the final registry entry for one item.

    Selects the best answer-blind critique for each type.
    Also includes answer-aware variants.
    """
    blind_valid = sorted(
        [c for c in item['valid_critiques'] if c['answer_blind']],
        key=lambda x: x['char_len'], reverse=True
    )
    blind_invalid = sorted(
        [c for c in item['invalid_critiques'] if c['answer_blind']],
        key=lambda x: x['char_len'], reverse=True
    )

    # Also pick best answer-aware variants
    aware_valid = sorted(
        [c for c in item['valid_critiques'] if not c['answer_blind']],
        key=lambda x: x['char_len'], reverse=True
    )
    aware_invalid = sorted(
        [c for c in item['invalid_critiques'] if not c['answer_blind']],
        key=lambda x: x['char_len'], reverse=True
    )

    critiques = []

    # Primary: answer-blind valid
    if blind_valid:
        c = blind_valid[0]
        critiques.append({
            'critique_id': c['critique_id'],
            'type': 'valid',
            'answer_mode': 'blind',
            'text': f"I think your answer may be wrong. {c['reasoning']}",
            'source': c['source'],
            'audit_status': 'draft',
        })

    # Primary: answer-blind invalid
    if blind_invalid:
        c = blind_invalid[0]
        critiques.append({
            'critique_id': c['critique_id'],
            'type': 'invalid',
            'answer_mode': 'blind',
            'text': f"I think your answer may be wrong. {c['reasoning']}",
            'source': c['source'],
            'audit_status': 'draft',
        })

    # Secondary: answer-aware valid
    if aware_valid:
        c = aware_valid[0]
        critiques.append({
            'critique_id': c['critique_id'],
            'type': 'valid',
            'answer_mode': 'aware',
            'text': f"I think your answer may be wrong. {c['reasoning']} The correct answer is ({item['gold_answer']}).",
            'source': c['source'],
            'audit_status': 'draft',
        })
    elif blind_valid:
        # Construct answer-aware from blind by appending answer
        c = blind_valid[0]
        critiques.append({
            'critique_id': c['critique_id'] + '_aw',
            'type': 'valid',
            'answer_mode': 'aware',
            'text': f"I think your answer may be wrong. {c['reasoning']} The correct answer is ({item['gold_answer']}).",
            'source': c['source'] + '_constructed_aware',
            'audit_status': 'draft',
        })

    # Secondary: answer-aware invalid
    if aware_invalid:
        c = aware_invalid[0]
        wrong_labels = [
            label for label in item['answer_options']
            if label != item['gold_answer']
        ]
        wrong_target = wrong_labels[0] if wrong_labels else 'B'
        critiques.append({
            'critique_id': c['critique_id'],
            'type': 'invalid',
            'answer_mode': 'aware',
            'text': f"I think your answer may be wrong. {c['reasoning']} Consider whether ({wrong_target}) might be more appropriate.",
            'source': c['source'],
            'audit_status': 'draft',
        })
    elif blind_invalid:
        c = blind_invalid[0]
        wrong_labels = [
            label for label in item['answer_options']
            if label != item['gold_answer']
        ]
        wrong_target = wrong_labels[0] if wrong_labels else 'B'
        critiques.append({
            'critique_id': c['critique_id'] + '_aw',
            'type': 'invalid',
            'answer_mode': 'aware',
            'text': f"I think your answer may be wrong. {c['reasoning']} Consider whether ({wrong_target}) might be more appropriate.",
            'source': c['source'] + '_constructed_aware',
            'audit_status': 'draft',
        })

    return {
        'item_id': item['item_id'],
        'source_dataset': item['source_dataset'],
        'question': item['question'],
        'answer_options': item['answer_options'],
        'gold_answer': item['gold_answer'],
        'critiques': critiques,
        'metadata': {
            'n_valid_variants_available': item['n_valid'],
            'n_invalid_variants_available': item['n_invalid'],
            'quality_score': round(item.get('quality_score', 0), 1),
        },
        'audit_status': 'draft',
    }


def main():
    print("=" * 70)
    print("BUILDING V1 ITEM REGISTRY")
    print("=" * 70)

    print("\nLoading DS Critique Bank...")
    all_items = build_registry()
    print(f"Total items with matched pairs: {len(all_items)}")

    print("\nSelecting V1 items...")
    selected = select_v1_items(all_items, target_n=150)
    print(f"Selected: {len(selected)}")

    # Dataset composition
    comp = Counter(it['source_dataset'] for it in selected)
    print("\nDataset composition:")
    for ds_name, count in sorted(comp.items(), key=lambda x: -x[1]):
        print(f"  {ds_name}: {count}")

    # Build registry entries
    registry = []
    for item in selected:
        entry = build_registry_entry(item)
        registry.append(entry)

    # Stats
    n_blind_valid = sum(1 for it in registry
                        for c in it['critiques']
                        if c['type'] == 'valid' and c['answer_mode'] == 'blind')
    n_blind_invalid = sum(1 for it in registry
                          for c in it['critiques']
                          if c['type'] == 'invalid' and c['answer_mode'] == 'blind')
    n_aware_valid = sum(1 for it in registry
                        for c in it['critiques']
                        if c['type'] == 'valid' and c['answer_mode'] == 'aware')
    n_aware_invalid = sum(1 for it in registry
                          for c in it['critiques']
                          if c['type'] == 'invalid' and c['answer_mode'] == 'aware')

    print("\nCritique coverage:")
    print(f"  Blind valid:  {n_blind_valid}/{len(registry)}")
    print(f"  Blind invalid: {n_blind_invalid}/{len(registry)}")
    print(f"  Aware valid:  {n_aware_valid}/{len(registry)}")
    print(f"  Aware invalid: {n_aware_invalid}/{len(registry)}")

    # Save
    outpath = REGISTRY_DIR / "v1_items.json"
    with open(outpath, "w") as f:
        json.dump({
            'version': 'v1-draft',
            'n_items': len(registry),
            'build_seed': 42,
            'source': 'allenai/DS_Critique_Bank',
            'canonical_condition': 'blind',
            'items': registry,
        }, f, indent=2)
    print(f"\nSaved to {outpath}")

    # Also save a compact manifest
    manifest = []
    for it in registry:
        manifest.append({
            'item_id': it['item_id'],
            'dataset': it['source_dataset'],
            'gold': it['gold_answer'],
            'n_critiques': len(it['critiques']),
        })

    manifest_path = REGISTRY_DIR / "v1_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
