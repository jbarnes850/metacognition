"""Validate the scoring readout: tokenization, padding invariance, agreement with free generation."""

import json
import sys
import time

import torch

from eos.score import generate_rows, load_model, read_jsonl, render, score_rows

MODEL = sys.argv[1]
rows = read_jsonl("eos_data/trials/smoke.jsonl.gz")
tok, model = load_model(MODEL)
print(type(model).__name__)
print(repr(render(tok, rows[0]["messages"], None)[-400:]))
for s in ["(A)", "(B) is", "Answer: (C)"]:
    ids = tok.encode(s, add_special_tokens=False)
    print(s, ids, [tok.decode([i]) for i in ids])

# Padding invariance: 8 rows one at a time vs one padded batch.
single = [score_rows(tok, model, [r], log_every=10**9)[0]["probs"] for r in rows[:8]]
batched = [r["probs"] for r in score_rows(tok, model, rows[:8], log_every=10**9)]
diff = max(abs(a[k] - b[k]) for a, b in zip(single, batched) for k in a)
print("max |p_single - p_batched| =", diff)

t0 = time.time()
scored = score_rows(tok, model, rows)
dt = time.time() - t0
print(f"scored {len(rows)} in {dt:.1f}s ({len(rows)/dt:.1f}/s), peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
gen = generate_rows(tok, model, rows, max_new_tokens=12)
agree = sum(a["decision"] == b["gen_decision"] for a, b in zip(scored, gen))
print(f"agreement prefill-readout vs free generation: {agree}/{len(rows)}")
for a, b in zip(scored, gen):
    if a["decision"] != b["gen_decision"]:
        print("  disagree", a["decision"], repr(b["generation"]), a["probs"])
print("letter mass mean", sum(r["letter_mass"] for r in scored) / len(scored))
json.dump({"agree": agree, "n": len(rows), "pad_diff": diff, "rate": len(rows) / dt,
           "examples": [g["generation"] for g in gen[:10]]}, open("eos_results/readout_check.json", "w"), indent=1)
