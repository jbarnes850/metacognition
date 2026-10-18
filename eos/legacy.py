"""MLX-free copy of sweep_fullpool.prepare_items (commit 43774c9), used to rebuild
the legacy 969-item critique pool exactly as the published sweep built it."""

from datasets import load_dataset


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
