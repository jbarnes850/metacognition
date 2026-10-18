"""Results tables, named numbers, and figures, all from eos_results/analysis.json.

python3 -m eos.report  ->  docs/evidence-over-source-results.md, eos_results/numbers.json, figures/eos_*.svg
"""

from __future__ import annotations

import json
from pathlib import Path

import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

A = json.loads(Path("eos_results/analysis.json").read_text())
AUDIT = json.loads(Path("eos_results/stimulus_audit.json").read_text())
NUM: dict[str, str] = {}
FIG = Path("figures")

INK, INK2, RULE, OURS, BASE, NEUTRAL, PAPER = "#36332f", "#6f675e", "#ded8cd", "#2457ff", "#c2562f", "#aaa59d", "#fbfaf7"
plt.rcParams.update({
    "font.family": ["IBM Plex Sans", "DejaVu Sans"], "font.size": 11, "svg.fonttype": "none",
    "axes.edgecolor": RULE, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": RULE, "grid.linewidth": 0.8,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none", "ytick.major.size": 0,
    "svg.hashsalt": "eos",  # stable element ids, so reruns reproduce the SVGs byte for byte
})


def save(fig, name):
    """SVG for the repository; a PNG preview under runs/previews for visual checks."""
    fig.savefig(FIG / f"{name}.svg", metadata={"Date": None})
    Path("runs/previews").mkdir(parents=True, exist_ok=True)
    fig.savefig(Path("runs/previews") / f"{name}.png", dpi=110, facecolor=PAPER)


def f2(x):
    return f"{x:.2f}".replace("-", "−")


def f3(x):
    return f"{x:.3f}".replace("-", "−")


def pct(x):
    return f"{100 * x:.1f}%"


def put(name, value):
    NUM[name] = value
    return value


def ci(d):
    return f"[{f2(d['ci'][0])}, {f2(d['ci'][1])}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def section_provenance(md):
    md.append("\n## Provenance\n")
    md.append("Model: Qwen/Qwen3.5-4B, bf16, on one NVIDIA GB10. Scoring and training: PyTorch 2.13.0 (CUDA 13.0), "
              "Transformers 5.12.1, peft 0.21.2. Without causal-conv1d installed, Transformers ran the Gated DeltaNet "
              "layers on its PyTorch path. S2A rewrites: an SGLang development build. "
              "Readout validated at 127/128 agreement with free generation (`eos_results/readout_check.json`).\n")
    rows = []
    for d in sorted(Path("eos_results/adapters").glob("*")):
        meta = json.loads((d / "train_meta.json").read_text())
        sha = (d / "adapter_model.sha256").read_text().split()[0]
        rows.append([d.name, sha[:16], meta["train_sha256"][:16], meta["n_steps"], meta["args"]["train"]])
    md.append(table(["Adapter", "adapter sha256", "training file sha256", "steps", "training file"], rows))
    rows = []
    for m in sorted(Path("eos_results/scores").glob("*__meta.json")):
        meta = json.loads(m.read_text())
        for run in meta.get("runs", []):
            for out, f in run.get("files", {}).items():
                rows.append([Path(out).name, f["n"], Path(f["input"]).name, f["input_sha256"][:12],
                             (run.get("adapter_sha256") or "")[:12], "yes" if run.get("system") else ""])
    md.append("\nNumerical note: scoring runs in bf16 with left-padded batches, so a trial's letter probabilities depend "
              "slightly on which other trials share its batch (max probability difference 0.02 in the readout check). "
              "Rescoring 1,500 trials in a different batch order changed 13 decisions (0.9%), each with a top-two probability margin of at most 0.041.\n")
    md.append("\nScored files (from `eos_results/scores/*__meta.json`):\n")
    md.append(table(["Output", "rows", "input", "input sha256", "adapter sha256", "system prompt"], rows))


def section_scorecard(md):
    """Pre-registered hypotheses, each verdict computed from analysis.json."""
    O = A.get("own_answer") or {}
    C = A.get("crossed_core") or {}
    CA = A.get("clean_accuracy") or {}
    rows = []

    def own(arm):
        return (O.get("arms") or {}).get(arm)

    def verdict(ok):
        return "pass" if ok else "fail"

    pend = "pending"
    # Crossed primary (pre-registered).
    p = (C.get("paired") or {}).get("main-base")
    ko = (A.get("knowledge_only") or {}).get("main")
    note = ""
    if p and ko:
        mc = C["arms"]["main"]["primary_conflict_heldout"]["d"]
        note = (f"; main conflict d' {f3(mc)} vs its knowledge-only bar {f3(ko['d'])}, so not evidence of critique "
                "use (amendment 2)")
    rows.append(["H1 crossed: main > base on conflict d' held-out", verdict(p["ci"][0] > 0) if p else pend,
                 (f"{f3(p['diff'])} [{f3(p['ci'][0])}, {f3(p['ci'][1])}]" + note) if p else ""])
    p = (O.get("paired") or {}).get("main-base")
    rows.append(["H1 own-answer (co-primary): main > base", verdict(p["ci"][0] > 0) if p else pend,
                 f"{f3(p['diff'])} [{f3(p['ci'][0])}, {f3(p['ci'][1])}]" if p else ""])
    for design, src in (("crossed", C), ("own-answer", O)):
        p = (src.get("paired") or {}).get("main-nocue")
        rows.append([f"H2 {design}: main > nocue", verdict(p["ci"][0] > 0) if p else pend,
                     f"{f3(p['diff'])} [{f3(p['ci'][0])}, {f3(p['ci'][1])}]" if p else ""])
    b, m = own("base"), own("main")
    if b and m and "main" in CA:
        hr_drop = b["none"]["hr"] - m["none"]["hr"]
        d_drop = b["none"]["d"] - m["none"]["d"]
        acc_drop = CA["base"]["heldout"] - CA["main"]["heldout"]
        ok = hr_drop <= 0.05 and d_drop <= 0.2 and acc_drop <= 0.03
        rows.append(["H3 no cost to correction (main)", verdict(ok),
                     f"no-cue HR drop {f3(hr_drop)}, d' drop {f3(d_drop)}, clean acc drop {f3(acc_drop)}"])
    sh = (A.get("crossed_reduced") or {}).get("paired", {})
    if "shuffled-base" in sh and "main-base" in sh:
        ratio = sh["shuffled-base"]["diff"] / sh["main-base"]["diff"] if sh["main-base"]["diff"] else float("nan")
        rows.append(["Artifact check: shuffled gain < half of main's (crossed, s2/s4)", verdict(ratio < 0.5), f"ratio {f3(ratio)}"])
    else:
        rows.append(["Artifact check: shuffled gain < half of main's", "not run",
                     "also not run: instruction baseline, BCT on the crossed core set, "
                     "crossed red team, mechanism check"])
    for arm, tag in (("main", "main"), ("bct", "BCT"), ("bctown", "BCT-own")):
        a = own(arm)
        if a:
            gaps = {f: a["families"][f]["conflict_4"]["d"] - a["none"]["d"] for f in HELDOUT_FAMS}
            rows.append([f"Goal ({tag}, own-answer): conflict s4 >= no cue, each held-out family",
                         f"{sum(g >= 0 for g in gaps.values())} of 3 families",
                         ", ".join(f"{k} {f3(v)}" for k, v in gaps.items())])
    for arm, tag, base_far_cap in (("bct", "B", None), ("bctown", "O", 0.05)):
        a = own(arm)
        if not (a and b):
            continue
        p = O["paired"].get(f"{arm}-base")
        within = abs(a["primary_conflict_heldout"]["d"] - b["none"]["d"]) <= 0.2
        rows.append([f"{tag}1 {arm}: conflict d' > base and within 0.2 of base no-cue d'",
                     verdict(p and p["ci"][0] > 0 and within),
                     f"diff {f3(p['diff'])} [{f3(p['ci'][0])}, {f3(p['ci'][1])}]; {f3(a['primary_conflict_heldout']['d'])} vs {f3(b['none']['d'])}"])
        ok2 = a["none"]["hr"] >= b["none"]["hr"] - 0.05
        detail = f"no-cue HR {f3(a['none']['hr'])} vs base {f3(b['none']['hr'])}"
        if base_far_cap is not None:
            ok2 = ok2 and a["none"]["far"] <= b["none"]["far"] + base_far_cap
            detail += f"; FAR {f3(a['none']['far'])} vs base {f3(b['none']['far'])}"
        if a.get("reanswer"):
            detail += f"; final = own clean answer {pct(a['reanswer']['final_equals_own_clean'])}"
        rows.append([f"{tag}2 {arm}: evidence use kept", verdict(ok2), detail])
        if arm in CA:
            dacc = CA["base"]["heldout"] - CA[arm]["heldout"]
            rows.append([f"{tag}3 {arm}: clean accuracy within 3 points", verdict(dacc <= 0.03),
                         f"{pct(CA[arm]['heldout'])} vs base {pct(CA['base']['heldout'])}"])
    md.append("## Scorecard (pre-registered hypotheses and amendments)\n")
    md.append(table(["Hypothesis", "Verdict", "Evidence"], rows))
    md.append("")


HELDOUT_FAMS = ("verifier", "role", "messenger")


def section_design(md):
    import json as _json
    items = _json.loads(Path("eos_data/items_v1.json").read_text())
    from collections import Counter
    c = Counter(it["split"] for it in items)
    put("n_items", str(len(items)))
    put("n_train", str(c["train"]))
    put("n_test_id", str(c["test_id"]))
    put("n_test_ood", str(c["test_ood"]))
    md.append(f"Design: {len(items)} matched items; train {c['train']}, test_id {c['test_id']}, test_ood {c['test_ood']}.\n")


def section_audit(md):
    put("audit_legacy_auroc", f"{AUDIT['legacy_full_text']['auroc']:.5f}")
    put("audit_legacy_reasoning_auroc", f3(AUDIT["legacy_reasoning_only"]["auroc"]))
    put("audit_matched_auroc", f3(AUDIT["matched_voice"]["auroc"]))
    put("audit_matched_length_auroc", f3(AUDIT["matched_voice"]["length_auroc"]))
    md.append("## Stimulus audit (question-blind validity classifier)\n")
    md.append(table(["Stimuli", "n critiques", "AUROC"], [
        ["Published critiques, full text", AUDIT["legacy_full_text"]["n"], NUM["audit_legacy_auroc"]],
        ["Published, closing phrase stripped", AUDIT["legacy_full_text"]["n"], NUM["audit_legacy_reasoning_auroc"]],
        ["Matched voice (same student, one template)", AUDIT["matched_voice"]["n"], NUM["audit_matched_auroc"]],
        ["Matched voice, length only", AUDIT["matched_voice"]["n"], NUM["audit_matched_length_auroc"]],
    ]))
    md.append("\nSource: `eos_results/stimulus_audit.json` (`eos/audit.py`, 5-fold CV grouped by item).\n")
    fig, ax = plt.subplots(figsize=(7, 2.4))
    labels = ["Published critiques", "Published, closing\nphrase removed", "Matched-voice\ncritiques"]
    vals = [AUDIT["legacy_full_text"]["auroc"], AUDIT["legacy_reasoning_only"]["auroc"], AUDIT["matched_voice"]["auroc"]]
    cols = [BASE, BASE, OURS]
    ax.set_axisbelow(True)
    for i, (v, c) in enumerate(zip(vals, cols)):
        ax.plot([0.5, v], [i, i], color=c, lw=2, solid_capstyle="butt")
        ax.plot([v], [i], "o", color=c, ms=7)
        ax.text(v + 0.012, i, ">0.999" if v > 0.9995 else f"{v:.3f}", va="center", color=INK, fontsize=11)
    ax.axvline(0.5, color=INK2, lw=1)
    ax.text(0.5, -0.75, "chance", color=INK2, fontsize=11, ha="center")
    ax.set_yticks(range(3), labels)
    ax.set_ylim(2.5, -0.95)
    ax.set_xlim(0.45, 1.08)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Validity AUROC from the critique text alone (question hidden)")
    ax.grid(axis="x", color=RULE)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "eos_stimulus_audit")
    plt.close(fig)


def section_legacy(md):
    L = A["legacy_reanalysis"]
    md.append("## Forensic reanalysis of the published per-trial files\n")
    ff = L["false_flaw_items"]
    put("legacy_false_flaw_n", str(ff["n"]))
    put("legacy_pool_n", str(ff["pool"]))
    md.append(f"{ff['target_equals_gold']} of {ff['n']} false-flaw invalid critiques in the {ff['pool']}-item pool "
              "recommend the gold answer, which on a noise trial is the answer the model already holds.\n")
    rows = []
    for m, r in L["false_flaw"].items():
        key = m.replace(" ", "_").replace(".", "")
        put(f"legacy_d_pub_{key}", f3(r["d_published"]))
        put(f"legacy_d_noff_{key}", f3(r["d_without_false_flaw"]))
        put(f"legacy_far_ff_{key}", f3(r["far_false_flaw"]))
        put(f"legacy_far_redirect_{key}", f3(r["far_redirect"]))
        put(f"legacy_d_noff_lo_{key}", f2(r["d_without_false_flaw_ci"][0]))
        put(f"legacy_d_noff_hi_{key}", f2(r["d_without_false_flaw_ci"][1]))
        rows.append([m, f3(r["d_published"]), r["n_false_flaw"], f3(r["far_false_flaw"]), r["n_redirect"],
                     f3(r["far_redirect"]), f"{f3(r['d_without_false_flaw'])} [{f2(r['d_without_false_flaw_ci'][0])}, "
                     f"{f2(r['d_without_false_flaw_ci'][1])}]"])
    md.append(table(["Model", "published d'", "n false-flaw", "FAR false-flaw", "n redirect", "FAR redirect",
                     "d' without false-flaw"], rows))
    md.append("\nPanel experiment, cue-combination terms (published 240-trial subsets):\n")
    rows = []
    for m, r in L["panel_decomposition"].items():
        key = m.replace("-", "_").replace(".", "")
        for k in ("baseline", "congruent", "conflict", "evidence_term", "source_term",
                  "baseline_without_false_flaw", "conflict_without_false_flaw"):
            put(f"panel_{k}_{key}", f3(r[k]))
        rows.append([m, f3(r["baseline"]), f3(r["congruent"]), f3(r["conflict"]), f3(r["evidence_term"]), f3(r["source_term"])])
    md.append(table(["Model", "no cue", "congruent 5-2", "conflict 5-2", "evidence term", "source term"], rows))
    md.append("\nSources: `eos_data/legacy_raw/` (copied from `metacognition/results/`, hashes in SHA256SUMS).\n")


def arm_row(C, arm):
    a = C["arms"][arm]
    p = a["primary_conflict_heldout"]
    n = a["conditions"]["none"]
    b = a["conditions"]["bare"]
    g = a["congruent_heldout_pooled"]
    return [arm, f"{f3(p['d'])} {ci(p)}", f3(n["d"]), f3(n["hr"]), f3(n["far"]), f3(b["d"]), f3(g["d"]),
            f3(a["conflict_trained_pooled"]["d"])]


def section_primary(md):
    C = A.get("crossed_core")
    if not C:
        return
    md.append("## Primary result: crossed design, held-out items\n")
    md.append(f"{C['n_items']} held-out items, 2 trials per item per condition. Primary metric: conflict d' pooled over "
              "held-out families (verifier, role, messenger) and strengths 1 to 4.\n")
    md.append(table(["Arm", "primary: conflict d' held-out [95% CI]", "no-cue d'", "no-cue HR", "no-cue FAR",
                     "bare d'", "congruent d' held-out", "conflict d' trained fams (2, 4)"],
                    [arm_row(C, a) for a in C["arms"]]))
    for arm, a in C["arms"].items():
        p = a["primary_conflict_heldout"]
        put(f"primary_{arm}", f2(p["d"]))
        put(f"primary_{arm}_lo", f2(p["ci"][0]))
        put(f"primary_{arm}_hi", f2(p["ci"][1]))
        put(f"none_{arm}", f2(a["conditions"]["none"]["d"]))
        put(f"none_hr_{arm}", f2(a["conditions"]["none"]["hr"]))
        put(f"none_far_{arm}", f2(a["conditions"]["none"]["far"]))
        put(f"bare_{arm}", f2(a["conditions"]["bare"]["d"]))
        put(f"congruent_heldout_{arm}", f2(a["congruent_heldout_pooled"]["d"]))
        put(f"conflict_trained_{arm}", f2(a["conflict_trained_pooled"]["d"]))
        put(f"type2_{arm}", f2(a["type2_auroc"]) if a["type2_auroc"] else "n/a")
        for fam, g in a["goal_strength4"].items():
            put(f"goal_gap_{fam}_{arm}", f2(g["gap"]))
            put(f"conflict4_{fam}_{arm}", f2(g["conflict_4"]))
    md.append("\nPaired differences in the primary metric (item bootstrap):\n")
    md.append(table(["Comparison", "difference", "95% CI", "share of resamples <= 0"],
                    [[k, f3(v["diff"]), f"[{f3(v['ci'][0])}, {f3(v['ci'][1])}]", f3(v["p_boot_le_0"])]
                     for k, v in C["paired"].items()]))
    for k, v in C["paired"].items():
        put(f"diff_{k}", f2(v["diff"]))
        put(f"diff_{k}_lo", f2(v["ci"][0]))
        put(f"diff_{k}_hi", f2(v["ci"][1]))
    md.append("\nConflict d' by held-out family and strength (and no-cue d'):\n")
    rows = []
    for arm, a in C["arms"].items():
        for fam in ("verifier", "role", "messenger"):
            F = a["families"][fam]
            rows.append([arm, fam] + [f3(F[f"conflict_{s}"]["d"]) for s in (1, 2, 3, 4)] +
                        [f3(a["conditions"]["none"]["d"]), f3(F["probit"]["strength_slope"]),
                         f3(F["probit"]["validity"]), f3(F["probit"]["slope_over_validity"])])
            put(f"slope_{fam}_{arm}", f3(F["probit"]["strength_slope"]))
            put(f"ratio_{fam}_{arm}", f2(F["probit"]["slope_over_validity"]))
    md.append(table(["Arm", "family", "conflict s1", "s2", "s3", "s4", "no cue", "probit strength slope",
                     "probit validity", "slope / validity"], rows))
    md.append("\nCue-combination terms at strength 4 (evidence = (cong + conf)/2, source = (cong - conf)/2):\n")
    rows = []
    for arm, a in C["arms"].items():
        for fam in ("verifier", "role", "messenger", "panel", "credential", "confidence"):
            F = a["families"][fam]
            rows.append([arm, fam, f3(a["conditions"]["none"]["d"]), f3(F["evidence_term_4"]), f3(F["source_term_4"]),
                         f3(F["congruent_4"]["hr"]), f3(F["congruent_4"]["far"]), f3(F["conflict_4"]["hr"]),
                         f3(F["conflict_4"]["far"])])
            put(f"source4_{fam}_{arm}", f2(F["source_term_4"]))
            put(f"evidence4_{fam}_{arm}", f2(F["evidence_term_4"]))
    md.append(table(["Arm", "family", "no-cue d'", "evidence term", "source term", "cong HR", "cong FAR",
                     "conf HR", "conf FAR"], rows))
    md.append("\nDomain split (held-out commonsense = test_id; held-out science = test_ood):\n")
    rows = []
    for arm, a in C["arms"].items():
        for split, D in a["domain"].items():
            rows.append([arm, split, f3(D["none"]["d"]), f"{f3(D['conflict_heldout']['d'])} {ci(D['conflict_heldout'])}"])
            put(f"domain_{split}_none_{arm}", f2(D["none"]["d"]))
            put(f"domain_{split}_conflict_{arm}", f2(D["conflict_heldout"]["d"]))
    md.append(table(["Arm", "split", "no-cue d'", "conflict d' held-out [CI]"], rows))
    figure_dose(C)


def figure_dose(C):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    colors = {"base": BASE, "main": OURS, "nocue": NEUTRAL}
    names = {"base": "Base", "main": "Trained with cues", "nocue": "Trained without cues"}
    titles = {"verifier": "A. Automated verifier", "role": "B. Role authority", "messenger": "C. Messenger credibility"}
    for ax, fam in zip(axes, ("verifier", "role", "messenger")):
        for arm in ("base", "nocue", "main"):
            if arm not in C["arms"]:
                continue
            a = C["arms"][arm]
            ys = [a["families"][fam][f"conflict_{s}"]["d"] for s in (1, 2, 3, 4)]
            lo = [a["families"][fam][f"conflict_{s}"]["ci"][0] for s in (1, 2, 3, 4)]
            hi = [a["families"][fam][f"conflict_{s}"]["ci"][1] for s in (1, 2, 3, 4)]
            ax.fill_between([1, 2, 3, 4], lo, hi, color=colors[arm], alpha=0.12, lw=0)
            ax.plot([1, 2, 3, 4], ys, color=colors[arm], lw=2 if arm == "main" else 1.5, marker="o", ms=4)
            ax.axhline(a["conditions"]["none"]["d"], color=colors[arm], lw=1, ls=(0, (3, 3)))
            if fam == "messenger":
                ax.text(4.12, ys[-1], names[arm], color=colors[arm], va="center", fontsize=11)
        ax.set_title(titles[fam], loc="left", fontsize=11, color=INK)
        ax.set_xticks([1, 2, 3, 4], ["weak", "", "", "strong"])
        ax.set_xlim(0.8, 4.2)
    axes[0].set_ylabel("d' when the cue conflicts\nwith the evidence")
    axes[1].set_xlabel("Cue strength (cue families never seen in training)")
    fig.tight_layout()
    fig.subplots_adjust(right=0.83)
    save(fig, "eos_dose_response")
    plt.close(fig)


def section_own(md):
    O = A.get("own_answer")
    if not O:
        return
    md.append("\n## Co-primary: own-answer design (base model's own initial answers, held-out items)\n")
    rows = []
    for arm, a in O["arms"].items():
        p, n, b = a["primary_conflict_heldout"], a["none"], a["bare"]
        ra = a.get("reanswer") or {}
        rows.append([arm, f"{f3(p['d'])} {ci(p)}", f"{f3(n['d'])} {ci(n)}", f3(n["hr"]), f3(n["far"]), f3(b["d"]),
                     f3(a["congruent_heldout"]["d"]), f3(a["conflict_trained"]["d"]),
                     pct(ra["final_equals_own_clean"]) if ra else "",
                     f3(ra["hr_none_when_clean_wrong"]) if ra and ra.get("hr_none_when_clean_wrong") is not None else ""])
        cc = a["cue_cost_heldout"]
        put(f"own_cuecost_{arm}", f2(cc["diff"]))
        put(f"own_cuecost_{arm}_lo", f2(cc["ci"][0]))
        put(f"own_cuecost_{arm}_hi", f2(cc["ci"][1]))
        put(f"own_primary_{arm}", f2(p["d"]))
        put(f"own_primary_{arm}_lo", f2(p["ci"][0]))
        put(f"own_primary_{arm}_hi", f2(p["ci"][1]))
        put(f"own_none_{arm}", f2(n["d"]))
        put(f"own_none_hr_{arm}", f2(n["hr"]))
        put(f"own_none_far_{arm}", f2(n["far"]))
        put(f"own_none_far_pct_{arm}", f"{100 * n['far']:.0f}%")
        put(f"own_none_hr_pct_{arm}", f"{100 * n['hr']:.0f}%")
        put(f"own_bare_{arm}", f2(b["d"]))
        put(f"own_conflict_trained_{arm}", f2(a["conflict_trained"]["d"]))
        put(f"own_n_signal_{arm}", str(n["n_signal"]))
        if ra:
            put(f"own_reanswer_{arm}", f"{100 * ra['final_equals_own_clean']:.0f}%")
            for k in ("hr_none_when_clean_wrong", "hr_none_when_clean_correct"):
                if ra.get(k) is not None:
                    put(f"own_{k}_{arm}", f"{100 * ra[k]:.0f}%")
            put(f"own_n_clean_wrong_{arm}", str(ra["n_none_clean_wrong"]))
            put(f"own_n_clean_correct_{arm}", str(ra["n_none_clean_correct"]))
    md.append(table(["Arm", "conflict d' held-out (2,4) [CI]", "no-cue d' [CI]", "no-cue HR", "no-cue FAR", "bare d'",
                     "congruent d' held-out", "conflict d' trained", "final = own clean answer",
                     "no-cue HR when own clean answer wrong"], rows))
    md.append("\n" + table(["Comparison", "difference", "95% CI", "share <= 0"],
                            [[k, f3(v["diff"]), f"[{f3(v['ci'][0])}, {f3(v['ci'][1])}]", f3(v["p_boot_le_0"])]
                             for k, v in O["paired"].items()]))
    for k, v in O["paired"].items():
        put(f"own_diff_{k}", f2(v["diff"]))
        put(f"own_diff_{k}_lo", f2(v["ci"][0]))
        put(f"own_diff_{k}_hi", f2(v["ci"][1]))
    md.append("\nOwn-answer domain split (test_id: held-out commonsense items; test_ood: held-out science domain):\n")
    rows = []
    for arm, a in O["arms"].items():
        for split, D in a["domain"].items():
            cc = D["cue_cost"]
            rows.append([arm, split, D["none"]["n_signal"], f3(D["none"]["d"]), f3(D["none"]["hr"]), f3(D["none"]["far"]),
                         f3(D["conflict_heldout"]["d"]), f"{f3(cc['diff'])} [{f3(cc['ci'][0])}, {f3(cc['ci'][1])}]"])
            put(f"own_{split}_none_{arm}", f2(D["none"]["d"]))
            put(f"own_{split}_far_{arm}", f2(D["none"]["far"]))
            put(f"own_{split}_far_pct_{arm}", f"{100 * D['none']['far']:.0f}%")
            put(f"own_{split}_cost_{arm}", f2(cc["diff"]))
            put(f"own_{split}_cost_{arm}_lo", f2(cc["ci"][0]))
            put(f"own_{split}_cost_{arm}_hi", f2(cc["ci"][1]))
    md.append(table(["Arm", "split", "n signal", "no-cue d'", "HR", "FAR", "conflict d' held-out (2,4)", "cue cost [CI]"], rows))
    md.append("\nOwn-answer cue-combination terms per family (evidence = (cong + conf)/2, source = (cong - conf)/2):\n")
    rows = []
    for arm, a in O["arms"].items():
        for fam, F in a["families"].items():
            rows.append([arm, fam, f3(a["none"]["d"])] + [f3(F[f"conflict_{st}"]["d"]) for st in (2, 4)] +
                        [f3(F[f"evidence_term_{st}"]) for st in (2, 4)] + [f3(F[f"source_term_{st}"]) for st in (2, 4)])
            for st in (2, 4):
                put(f"own_source{st}_{fam}_{arm}", f2(F[f"source_term_{st}"]))
                put(f"own_conflict{st}_{fam}_{arm}", f2(F[f"conflict_{st}"]["d"]))
    md.append(table(["Arm", "family", "no-cue d'", "conflict s2", "conflict s4", "evidence s2", "evidence s4",
                     "source s2", "source s4"], rows))
    figure_evidence_use(O)
    figure_own_conflict(O)


def figure_own_conflict(O):
    """Own-answer d' with no cue and under conflicting held-out cues (strengths 2 and 4)."""
    arms = [a for a in ("base", "bct") if a in O["arms"]]
    colors = {"base": BASE, "bct": OURS}
    names = {"base": "Base", "bct": "Consistency-\ntrained (BCT)"}
    titles = {"verifier": "A. Automated verifier", "role": "B. Role authority", "messenger": "C. Messenger credibility"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
    for ax, fam in zip(axes, ("verifier", "role", "messenger")):
        ax.set_axisbelow(True)
        for arm in arms:
            a = O["arms"][arm]
            F = a["families"][fam]
            ys = [a["none"]["d"], F["conflict_2"]["d"], F["conflict_4"]["d"]]
            lo = [a["none"]["ci"][0], F["conflict_2"]["ci"][0], F["conflict_4"]["ci"][0]]
            hi = [a["none"]["ci"][1], F["conflict_2"]["ci"][1], F["conflict_4"]["ci"][1]]
            xs = [0, 1, 2]
            ax.fill_between(xs, lo, hi, color=colors[arm], alpha=0.10, lw=0)
            ax.plot(xs, ys, color=colors[arm], lw=2.2 if arm == "bct" else 1.5, marker="o", ms=4)
            if fam == "messenger":
                ax.text(2.1, ys[-1], names[arm], color=colors[arm], va="center", fontsize=11)
        ax.set_title(titles[fam], loc="left", fontsize=11, color=INK)
        ax.set_xticks([0, 1, 2], ["no cue", "weak\nconflict", "strong\nconflict"])
        ax.set_xlim(-0.2, 2.2)
    axes[0].set_ylabel("d' (valid vs invalid critique)")
    fig.tight_layout()
    fig.subplots_adjust(right=0.84)
    save(fig, "eos_own_conflict")


def figure_evidence_use(O):
    arms = [a for a in ("base", "main", "nocue", "bct") if a in O["arms"] and O["arms"][a].get("reanswer")]
    names = {"base": "Base", "main": "Trained on gold\nanswers, with cues", "nocue": "Trained on gold\nanswers, no cues",
             "bct": "Consistency-\ntrained (BCT)"}
    colors = {"base": BASE, "main": NEUTRAL, "nocue": NEUTRAL, "bct": OURS}
    fig, ax = plt.subplots(figsize=(7.4, 1.0 + 1.0 * len(arms)))
    ax.set_axisbelow(True)
    for i, arm in enumerate(arms):
        a = O["arms"][arm]
        hit = a["reanswer"]["hr_none_when_clean_wrong"]
        far = a["none"]["far"]
        ax.plot([far, hit], [i, i], color=RULE, lw=2, zorder=1)
        ax.plot([hit], [i], "o", color=colors[arm], ms=9, zorder=3)
        ax.plot([far], [i], "o", color=colors[arm], ms=9, mfc=PAPER, mew=1.5, zorder=3)
        ax.text(hit, i - 0.3, f"valid {100 * hit:.0f}%", ha="center", color=INK, fontsize=11)
        ax.text(far, i + 0.42, f"invalid {100 * far:.0f}%", ha="center", color=INK2, fontsize=11)
    ax.set_yticks(range(len(arms)), [names[a] for a in arms])
    ax.set_ylim(len(arms) - 0.25, -0.75)
    ax.set_xlim(-0.06, 1.05)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Critiques accepted. Valid: questions this model answers wrong on its own.\n"
                  "Invalid: questions the base model answers right.")
    ax.grid(axis="x", color=RULE)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "eos_evidence_use")


def section_reduced(md):
    R = A.get("crossed_reduced")
    if not R:
        return
    md.append("## Like-for-like comparison on strengths 2 and 4 (all arms)\n")
    md.append(table(["Arm", "conflict d' held-out (2, 4) [CI]", "no-cue d'", "no-cue HR", "no-cue FAR", "bare d'",
                     "congruent d' held-out", "conflict d' trained"], [arm_row(R, a) for a in R["arms"]]))
    for arm, a in R["arms"].items():
        put(f"red24_primary_{arm}", f2(a["primary_conflict_heldout"]["d"]))
        put(f"red24_none_{arm}", f2(a["conditions"]["none"]["d"]))
    md.append("\n" + table(["Comparison", "difference", "95% CI"],
                           [[k, f3(v["diff"]), f"[{f3(v['ci'][0])}, {f3(v['ci'][1])}]"] for k, v in R["paired"].items()]))
    for k, v in R["paired"].items():
        put(f"red24_diff_{k}", f2(v["diff"]))


def section_clean_own(md):
    CA = A.get("clean_accuracy", {})
    if CA:
        md.append("## Clean accuracy (question only)\n")
        md.append(table(["Arm", "held-out", "test_id", "test_ood", "train"],
                        [[a, pct(v["heldout"]), pct(v.get("test_id", 0)), pct(v.get("test_ood", 0)), pct(v.get("train", 0))]
                         for a, v in CA.items()]))
        for a, v in CA.items():
            put(f"clean_{a}", pct(v["heldout"]))
    O = A.get("own_legacy_redteam", {})
    if O:
        md.append("\n## Own-answer design, published critiques, red team\n")
        for key, R in O.items():
            rows = [[k, f3(v["d"]), f"[{f2(v['ci'][0])}, {f2(v['ci'][1])}]", f3(v["hr"]), f3(v["far"]), v["n_signal"], v["n_noise"]]
                    for k, v in R.items() if v]
            md.append(f"\n**{key}**\n\n" + table(["condition", "d'", "95% CI", "HR", "FAR", "n signal", "n noise"], rows))
            tag = key.replace(":", "_")
            for k, v in R.items():
                if v:
                    put(f"{tag}_{k}_d", f2(v["d"]))
                    put(f"{tag}_{k}_hr", f2(v["hr"]))
                    put(f"{tag}_{k}_far", f2(v["far"]))


def section_redteam(md):
    R = A.get("redteam_own")
    if not R:
        return
    md.append("\n## Post-freeze red team (own-answer, strength 4)\n")
    rows = []
    for arm, r in R.items():
        cc = r["cue_cost"]
        rows.append([arm, f3(r["none"]["d"]), f"{f3(r['conflict_pooled']['d'])} {ci(r['conflict_pooled'])}",
                     f"{f3(cc['diff'])} [{f3(cc['ci'][0])}, {f3(cc['ci'][1])}]"] +
                    [f3(r["families"][f]["d"]) for f in ("pathos", "ai_consensus", "citation")])
        put(f"rt_conflict_{arm}", f2(r["conflict_pooled"]["d"]))
        put(f"rt_cost_{arm}", f2(cc["diff"]))
        put(f"rt_cost_{arm}_lo", f2(cc["ci"][0]))
        put(f"rt_cost_{arm}_hi", f2(cc["ci"][1]))
        for f in ("pathos", "ai_consensus", "citation"):
            put(f"rt_{f}_{arm}", f2(r["families"][f]["d"]))
    md.append(table(["Arm", "no-cue d'", "conflict d' pooled [CI]", "cue cost [CI]", "pathos", "AI consensus", "citation"], rows))


def section_s2a(md):
    S = A.get("s2a")
    if not S:
        return
    md.append("\n## System 2 Attention baselines (150-item subset, crossed, strength 4)\n")
    rows = []
    for arm, R in S.items():
        put(f"{arm}_critique_overlap", pct(R["critique_overlap_mean"]))
        put(f"{arm}_knowledge_only_subset", f2(R["knowledge_only_subset"]))
        put(f"{arm}_cue_words_kept", pct(R["keeps_cue_words_cue_trials"]))
        for name in [n for n in (arm, "base_same_trials", "bct_same_trials", "main_same_trials") if n in R]:
            if name in ("bct_same_trials", "main_same_trials") and arm != "s2a":
                continue
            D = R[name]
            label = arm if name == arm else f"{name.replace('_same_trials', '')} (same trials)"
            rows.append([label, f3(D["none"]["d"]), f3(D["none"]["hr"]), f3(D["none"]["far"]),
                         f"{f3(D['heldout_conflict_4']['d'])} {ci(D['heldout_conflict_4'])}",
                         f3(D["heldout_congruent_4"]["d"]), f3(D["trained_conflict_4"]["d"])])
            tag = arm if name == arm else f"{name.replace('_same_trials', '')}_for_{arm}"
            put(f"{tag}_none", f2(D["none"]["d"]))
            put(f"{tag}_conflict_heldout4", f2(D["heldout_conflict_4"]["d"]))
        md.append(f"- {arm}: rewrites keep {pct(R['critique_overlap_mean'])} of critique-specific words on average; "
                  f"cue wording survives in {pct(R['keeps_cue_words_cue_trials'])} of cue trials.")
    md.append("\n" + table(["Arm", "no-cue d'", "no-cue HR", "no-cue FAR", "conflict d' held-out s4 [CI]",
                            "congruent d' held-out s4", "conflict d' trained s4"], rows))
    prow = []
    for arm, R in S.items():
        for k, v in R.get("paired", {}).items():
            prow.append([k, f3(v["diff"]), f"[{f3(v['ci'][0])}, {f3(v['ci'][1])}]"])
            key = k.replace(":", "_").replace("-", "_minus_")
            put(f"s2acmp_{key}", f2(v["diff"]))
            put(f"s2acmp_{key}_lo", f2(v["ci"][0]))
            put(f"s2acmp_{key}_hi", f2(v["ci"][1]))
    if prow:
        md.append("\nPaired differences on the same trials (item bootstrap):\n")
        md.append(table(["Comparison", "difference", "95% CI"], prow))


def main():
    FIG.mkdir(exist_ok=True)
    md = ["# Results with provenance\n\nGenerated by `eos/report.py` from `eos_results/analysis.json`. "
          "Rerun everything with `eos/rerun.sh`.\n"]
    section_scorecard(md)
    section_design(md)
    section_audit(md)
    section_legacy(md)
    section_own(md)
    section_primary(md)
    section_reduced(md)
    section_clean_own(md)
    section_redteam(md)
    section_s2a(md)
    section_provenance(md)
    Path("docs/evidence-over-source-results.md").write_text("\n".join(md) + "\n")
    Path("eos_results/numbers.json").write_text(json.dumps(NUM, indent=1, ensure_ascii=False) + "\n")
    print("wrote docs/evidence-over-source-results.md, eos_results/numbers.json,", len(NUM), "numbers")


if __name__ == "__main__":
    main()
