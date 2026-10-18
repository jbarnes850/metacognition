"""Question-blind shortcut audit of critique stimuli.

A text classifier sees only the critique (never the question) and tries to
predict whether it is valid. High cross-validated AUROC means validity leaks
through surface form, so a trained model could learn the shortcut instead of
evaluating the reasoning. Folds are grouped by item.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline, make_union

from eos import stimuli
from eos.legacy import prepare_items


def cv_auroc(texts, labels, groups, seed=0):
    texts, labels, groups = np.array(texts, dtype=object), np.array(labels), np.array(groups)
    scores = np.zeros(len(labels))
    for tr, te in GroupKFold(n_splits=5).split(texts, labels, groups):
        clf = make_pipeline(
            make_union(
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True),
            ),
            LogisticRegression(C=4.0, max_iter=2000, random_state=seed),
        )
        clf.fit(texts[tr], labels[tr])
        scores[te] = clf.predict_proba(texts[te])[:, 1]
    return float(roc_auc_score(labels, scores))


def length_auroc(texts, labels):
    return float(roc_auc_score(labels, [len(t) for t in texts]))


def main(out: Path):
    res = {}
    # Legacy pool, critique strings exactly as published.
    legacy = prepare_items()
    t, y, g = [], [], []
    t_nf, y_nf, g_nf = [], [], []
    for it in legacy:
        for valid in (True, False):
            text, _ = stimuli.legacy_followup_text(it, valid)
            t.append(text); y.append(int(valid)); g.append(it["qid"])
            if valid or it["invalid_critique"]["type"] != "false_flaw":
                t_nf.append(text); y_nf.append(int(valid)); g_nf.append(it["qid"])
    res["legacy_full_text"] = {"n": len(t), "auroc": cv_auroc(t, y, g), "length_auroc": length_auroc(t, y)}
    res["legacy_no_false_flaw"] = {"n": len(t_nf), "auroc": cv_auroc(t_nf, y_nf, g_nf)}
    # Legacy with the closing template stripped (reasoning only).
    import re
    strip = lambda s: re.sub(r"(The correct answer is|Consider whether|The answer should be) \(\w\).*$", "", s)
    res["legacy_reasoning_only"] = {"auroc": cv_auroc([strip(s) for s in t], y, g)}
    # Matched-voice stimuli.
    items = stimuli.load_items()
    t, y, g = [], [], []
    for it in items:
        t.append(stimuli.critique_text(it["primary_valid"]["text"], it["gold"])); y.append(1); g.append(it["qid"])
        t.append(stimuli.critique_text(it["primary_invalid"]["text"], it["primary_invalid"]["answer"])); y.append(0); g.append(it["qid"])
    res["matched_voice"] = {"n": len(t), "auroc": cv_auroc(t, y, g), "length_auroc": length_auroc(t, y)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "eos_results/stimulus_audit.json"))
