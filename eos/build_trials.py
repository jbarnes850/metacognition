"""Build trial files (JSON lines, gzipped) for scoring and training."""

from __future__ import annotations

import argparse
import sys
import random
from pathlib import Path

from eos import cues, stimuli
from eos.legacy import prepare_items
from eos.jsonl import read_jsonl, write_jsonl

TRIALS = Path("eos_data/trials")
SCORES = Path("eos_results/scores")
HELDOUT_SPLITS = ("test_id", "test_ood")
ALL_FAMILIES = cues.TRAIN_FAMILIES + cues.HELDOUT_FAMILIES


def clean():
    items = stimuli.load_items()
    write_jsonl(TRIALS / "clean_matched.jsonl.gz", [stimuli.clean_trial(it) for it in items])
    rows = []
    for it in prepare_items():
        rows.append({"qid": it["qid"], "dataset": it["dataset"], "domain": stimuli.domain_for(it["dataset"]),
                     "split": "legacy", "design": "clean", "stimulus": "legacy",
                     "labels": stimuli.labels_for(it["question"]), "gold": it["gold"], "condition": "clean",
                     "messages": [{"role": "user", "content": it["question"] + stimuli.QUESTION_SUFFIX}]})
    write_jsonl(TRIALS / "clean_legacy.jsonl.gz", rows)
    print(len(items), len(rows))


def smoke(n_items: int = 32):
    items = [it for it in stimuli.load_items() if it["split"] != "train"]
    rng = random.Random(0)
    rng.shuffle(items)
    rows = []
    for it in items[:n_items]:
        rows += stimuli.crossed_trials(it, "none", None, None)
        rows += stimuli.crossed_trials(it, "conflict", "panel", 4)
    write_jsonl(TRIALS / "smoke.jsonl.gz", rows)
    print(len(rows))


def conditions(families, strengths):
    out = [("none", None, None), ("bare", None, None)]
    for fam in families:
        for cond in ("congruent", "conflict"):
            for s in strengths:
                out.append((cond, fam, s))
    return out


def base_initials(name: str) -> dict[str, str]:
    return {r["qid"]: r["decision"] for r in read_jsonl(SCORES / f"base__{name}.jsonl.gz")}


def core_conditions(strengths_heldout=cues.STRENGTHS):
    """Pre-registered crossed conditions: held-out families at the given strengths, trained at 2 and 4."""
    conds = [("none", None, None), ("bare", None, None)]
    for fams, strengths in ((cues.HELDOUT_FAMILIES, strengths_heldout), (cues.TRAIN_FAMILIES, (2, 4))):
        for fam in fams:
            for cond in ("congruent", "conflict"):
                for s in strengths:
                    conds.append((cond, fam, s))
    return conds


def eval_sets(families=ALL_FAMILIES, tag=""):
    items = [it for it in stimuli.load_items() if it["split"] in HELDOUT_SPLITS]
    for name, conds in (("core", core_conditions()), ("reduced", core_conditions((2, 4)))):
        crossed = []
        for it in items:
            for cond, fam, s in conds:
                crossed += stimuli.crossed_trials(it, cond, fam, s)
        write_jsonl(TRIALS / f"eval_{name}{tag}.jsonl.gz", crossed)
        print(name, len(conds), len(crossed))

    init = base_initials("clean_matched")
    own = []
    for it in items:
        for cond, fam, s in conditions(families, (2, 4)):
            own.append(stimuli.own_answer_trial(it, init[it["qid"]], cond, fam, s))
    write_jsonl(TRIALS / f"eval_own{tag}.jsonl.gz", own)
    print("own", len(own))


def legacy_set():
    train_qids = {it["qid"] for it in stimuli.load_items() if it["split"] == "train"}
    init = base_initials("clean_legacy")
    rows = []
    for it in prepare_items():
        if it["qid"] in train_qids:
            continue
        rows.append(stimuli.legacy_trial(it, init[it["qid"]], "none", None))
        for cond in ("congruent", "conflict"):
            for s in (2, 4):
                rows.append(stimuli.legacy_trial(it, init[it["qid"]], cond, s))
    write_jsonl(TRIALS / "eval_legacy.jsonl.gz", rows)
    print("legacy", len(rows))


def train_set(arm: str, n_per_item: int = 8, seed: int = 0):
    """Training examples from train items only. Arms: main, nocue, shuffled, bct_prep.

    bct_prep reproduces main's prompts exactly (same seed and draws) and also writes the
    cue-free version of each prompt, so the base model's cue-free decision can be the label.
    """
    rng = random.Random(seed)
    items = [it for it in stimuli.load_items() if it["split"] == "train"]
    rows = []
    for it in items:
        students = sorted({v["student"] for v in it["valid_explanations"]}
                          & {w["student"] for w in it["invalid_explanations"]})
        for _ in range(n_per_item):
            s = rng.choice(students)
            v = rng.choice([e for e in it["valid_explanations"] if e["student"] == s])
            w = rng.choice([e for e in it["invalid_explanations"] if e["student"] == s])
            valid = rng.random() < 0.5
            if valid:
                initial, expl, target = w["answer"], v["text"], it["gold"]
            else:
                initial, expl, target = it["gold"], w["text"], w["answer"]
            cue = cues.sample_train_cue(rng, valid)
            if arm == "nocue":
                cue = {"condition": "none", "family": None, "strength": None,
                       "direction": None, "placement": None, "variant": None}
            follow = stimuli.followup_text(expl, target, cue if cue["family"] else None)
            follow_none = stimuli.followup_text(expl, target, None)
            label = it["gold"]
            if arm == "shuffled":
                label = rng.choice([initial, target])  # revise or resist at random, ignoring validity
            rows.append({
                "qid": it["qid"], "dataset": it["dataset"], "labels": it["labels"], "gold": it["gold"],
                "initial": initial, "target": target, "critique_valid": valid, "label": label,
                "condition": cue["condition"], "family": cue["family"], "strength": cue["strength"],
                "direction": cue["direction"], "placement": cue["placement"],
                "messages": stimuli.messages_for(it["question"], initial, follow),
                "messages_none": stimuli.messages_for(it["question"], initial, follow_none),
            })
    rng.shuffle(rows)
    if arm == "bct_prep":
        write_jsonl(TRIALS / "train_bct_prep.jsonl.gz", rows)
        none_rows = [dict({k: v for k, v in r.items() if k not in ("messages", "messages_none")},
                          messages=r["messages_none"], row=i) for i, r in enumerate(rows)]
        write_jsonl(TRIALS / "train_bct_none.jsonl.gz", none_rows)
        print("bct_prep", len(rows))
        return
    for r in rows:
        r.pop("messages_none")
    write_jsonl(TRIALS / f"train_{arm}.jsonl.gz", rows)
    print(arm, len(rows))


def bct_labels():
    """Label each cued training prompt with the base model's decision on its cue-free version."""
    prep = read_jsonl(TRIALS / "train_bct_prep.jsonl.gz")
    scored = {r["row"]: r["decision"] for r in read_jsonl(SCORES / "base__train_bct_none.jsonl.gz")}
    rows = []
    for i, r in enumerate(prep):
        r = {k: v for k, v in r.items() if k != "messages_none"}
        r["label"] = scored[i]
        rows.append(r)
    write_jsonl(TRIALS / "train_bct.jsonl.gz", rows)
    agree = sum(r["label"] == r["gold"] for r in rows) / len(rows)
    print("bct", len(rows), "label==gold", round(agree, 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["clean", "smoke", "eval", "legacy", "train", "redteam", "bct_labels", "bctown_prep", "bctown_labels"])
    ap.add_argument("--arm", default="main")
    a = ap.parse_args()
    if a.cmd == "train":
        train_set(a.arm)
    elif a.cmd == "redteam":
        pass  # handled below, after redteam_sets is defined
    elif a.cmd == "bct_labels":
        bct_labels()
    elif a.cmd in ("bctown_prep", "bctown_labels"):
        pass  # handled below
    else:
        {"clean": clean, "smoke": smoke, "eval": eval_sets, "legacy": legacy_set}[a.cmd]()


def redteam_sets():
    """Post-freeze red-team families (eos/redteam_cues.py), strength 4, crossed and own-answer."""
    from eos import redteam_cues

    items = [it for it in stimuli.load_items() if it["split"] in HELDOUT_SPLITS]
    init = base_initials("clean_matched")
    conds = [("none", None, None)] + [(c, f, 4) for f in redteam_cues.REDTEAM_FAMILIES for c in ("congruent", "conflict")]
    crossed, own = [], []
    for it in items:
        for cond, fam, s in conds:
            crossed += stimuli.crossed_trials(it, cond, fam, s)
            own.append(stimuli.own_answer_trial(it, init[it["qid"]], cond, fam, s))
    write_jsonl(TRIALS / "eval_redteam_crossed.jsonl.gz", crossed)
    write_jsonl(TRIALS / "eval_redteam_own.jsonl.gz", own)
    print("redteam crossed", len(crossed), "own", len(own))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "redteam":
    redteam_sets()


def bctown_prep(n_per_item: int = 8, seed: int = 0):
    """BCT on own-answer prompts: train items with the base model's own initial answers.

    Validity follows the base answer (wrong gives a valid critique, right gives an invalid one),
    explanations are same-student pairs, cues are sampled as in train_main. Writes the cued prompts
    and their cue-free versions; labels come from the base model's cue-free decisions.
    """
    rng = random.Random(seed)
    items = [it for it in stimuli.load_items() if it["split"] == "train"]
    init = base_initials("clean_matched")
    rows = []
    for it in items:
        initial = init[it["qid"]]
        valid = initial != it["gold"]
        students = sorted({v["student"] for v in it["valid_explanations"]}
                          & {w["student"] for w in it["invalid_explanations"]})
        for _ in range(n_per_item):
            s = rng.choice(students)
            if valid:
                expl, target = rng.choice([e for e in it["valid_explanations"] if e["student"] == s])["text"], it["gold"]
            else:
                pool = [e for e in it["invalid_explanations"] if e["student"] == s and e["answer"] != initial]
                if not pool:
                    pool = [e for e in it["invalid_explanations"] if e["answer"] != initial]
                w = rng.choice(pool)
                expl, target = w["text"], w["answer"]
            cue = cues.sample_train_cue(rng, valid)
            rows.append({
                "qid": it["qid"], "dataset": it["dataset"], "labels": it["labels"], "gold": it["gold"],
                "initial": initial, "target": target, "critique_valid": valid,
                "condition": cue["condition"], "family": cue["family"], "strength": cue["strength"],
                "direction": cue["direction"], "placement": cue["placement"],
                "messages": stimuli.messages_for(it["question"], initial,
                                                 stimuli.followup_text(expl, target, cue if cue["family"] else None)),
                "messages_none": stimuli.messages_for(it["question"], initial, stimuli.followup_text(expl, target, None)),
            })
    rng.shuffle(rows)
    write_jsonl(TRIALS / "train_bctown_prep.jsonl.gz", rows)
    write_jsonl(TRIALS / "train_bctown_none.jsonl.gz",
                [dict({k: v for k, v in r.items() if k not in ("messages", "messages_none")},
                      messages=r["messages_none"], row=i) for i, r in enumerate(rows)])
    print("bctown_prep", len(rows), "valid share", round(sum(r["critique_valid"] for r in rows) / len(rows), 3))


def bctown_labels():
    prep = read_jsonl(TRIALS / "train_bctown_prep.jsonl.gz")
    scored = {r["row"]: r["decision"] for r in read_jsonl(SCORES / "base__train_bctown_none.jsonl.gz")}
    rows = [dict({k: v for k, v in r.items() if k != "messages_none"}, label=scored[i]) for i, r in enumerate(prep)]
    write_jsonl(TRIALS / "train_bctown.jsonl.gz", rows)
    print("bctown", len(rows), "label==gold", round(sum(r["label"] == r["gold"] for r in rows) / len(rows), 3))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("bctown_prep", "bctown_labels"):
    {"bctown_prep": bctown_prep, "bctown_labels": bctown_labels}[sys.argv[1]]()
