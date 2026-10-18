"""Score revise-or-resist trials with a Hugging Face model (runs in the CUDA container).

Readout: the first assistant token after the (empty) thinking block. Qwen
tokenizes "(A" as one token, so each option letter L collects the probability
of the tokens "L", "(L" and " L". The decision is the letter with the most mass;
the mass renormalized over option letters is kept as a confidence. The --generate
mode decodes freely instead, to validate the readout against generation.

Input and output are JSON lines (optionally gzipped). Output rows copy every
input field except "messages" and add the readout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import signal

import torch

from eos.jsonl import open_any, read_jsonl, write_jsonl  # noqa: F401  (re-exported)

PREFILL = ""
EXIT_CHECKPOINTED = 75  # job saved a checkpoint and exited on SIGTERM; rerun the same command to resume
STOP = {"flag": False}


def _on_term(signum, frame):
    STOP["flag"] = True


class Interrupted(Exception):
    pass




def render(tok, messages, system: str | None, prefill: str | None = PREFILL) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)
    return text + (prefill or "")


def letter_ids(tok, letters: str = "ABCDEFGH") -> dict[str, list[int]]:
    ids = {}
    for L in letters:
        ids[L] = []
        for form in (L, f"({L}", f" {L}"):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                ids[L].append(enc[0])
        assert ids[L], L
    return ids


def cap_memory(cap_gb: float):
    """Hard cap on this process's PyTorch allocator. On the GB10's unified memory an overrun then
    raises an out-of-memory error inside the job instead of starving the host."""
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gb * 1e9 / total))
    print(f"memory cap {cap_gb} GB of {total / 1e9:.0f} GB", flush=True)


def load_model(model_path: str, adapter: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


def batches_by_tokens(lengths, max_tokens: int, max_batch: int):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batch, longest = [], 0
    for i in order:
        L = lengths[i]
        if batch and (max(longest, L) * (len(batch) + 1) > max_tokens or len(batch) >= max_batch):
            yield batch
            batch, longest = [], 0
        batch.append(i)
        longest = max(longest, L)
    if batch:
        yield batch


@torch.no_grad()
def score_rows(tok, model, rows, system=None, max_tokens=24000, max_batch=48, log_every=5, done_rows=None):
    """Score rows; rows already in done_rows (index -> result) are skipped. Raises Interrupted on SIGTERM."""
    ids = letter_ids(tok)
    done_rows = done_rows or {}
    todo = [i for i in range(len(rows)) if i not in done_rows]
    texts = {i: render(tok, rows[i]["messages"], system) for i in todo}
    enc_lens = {i: len(tok.encode(texts[i], add_special_tokens=False)) for i in todo}
    out = [done_rows.get(i) for i in range(len(rows))]
    t0, done = time.time(), len(done_rows)
    order_lens = [enc_lens[i] for i in todo]
    for bi, local in enumerate(batches_by_tokens(order_lens, max_tokens, max_batch)):
        if STOP["flag"]:
            raise Interrupted(out)
        batch = [todo[k] for k in local]
        enc = tok([texts[i] for i in batch], return_tensors="pt", padding=True, pad_to_multiple_of=32,
                  add_special_tokens=False).to(model.device)
        logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        logp_full = torch.log_softmax(logits, dim=-1)
        for j, i in enumerate(batch):
            labels = rows[i]["labels"]
            raw = [float(logp_full[j, ids[L]].exp().sum()) for L in labels]
            mass = sum(raw)
            probs = {L: v / mass for L, v in zip(labels, raw)}
            decision = max(probs, key=probs.get)
            r = {k: v for k, v in rows[i].items() if k != "messages"}
            r.update({"probs": probs, "decision": decision, "letter_mass": mass,
                      "n_prompt_tokens": enc_lens[i]})
            r["_idx"] = i
            out[i] = r
        done += len(batch)
        if bi % log_every == 0:
            rate = done / max(1e-9, time.time() - t0)
            print(f"  scored {done}/{len(rows)} ({rate:.1f}/s)", flush=True)
    for r in out:
        r.pop("_idx", None)
    return out


def extract_answer(text: str, labels: list[str]) -> str | None:
    t = text.strip().upper()
    for L in labels:
        if f"({L})" in t or f"ANSWER IS {L}" in t:
            return L
    for ch in t:
        if ch in labels:
            return ch
    return None


@torch.no_grad()
def generate_rows(tok, model, rows, system=None, max_new_tokens=16, batch_size=32, prefill=None):
    texts = [render(tok, r["messages"], system, prefill=prefill) for r in rows]
    out = []
    for s in range(0, len(rows), batch_size):
        chunk = texts[s:s + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j, g in enumerate(gen):
            text = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            r = {k: v for k, v in rows[s + j].items() if k != "messages"}
            r.update({"generation": text, "gen_decision": extract_answer(text, r["labels"])})
            out.append(r)
        print(f"  generated {len(out)}/{len(rows)}", flush=True)
    return out


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--trials", required=True, nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--system", default=None, help="optional system prompt (instruction baseline)")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--mem-cap-gb", type=float, default=30.0)
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _on_term)
    cap_memory(args.mem_cap_gb)

    tok, model = load_model(args.model, args.adapter)
    meta = {"model": args.model, "adapter": args.adapter, "system": args.system,
            "adapter_sha256": sha256_file(Path(args.adapter) / "adapter_model.safetensors") if args.adapter else None,
            "torch": torch.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for path in args.trials:
        rows = read_jsonl(path)[: args.limit]
        name = Path(path).name.replace(".jsonl", "").replace(".gz", "")
        t0 = time.time()
        out = Path(args.out_dir) / f"{args.tag}__{name}{'__gen' if args.generate else ''}.jsonl.gz"
        partial = Path(str(out) + ".partial.jsonl")
        if args.generate:
            res = generate_rows(tok, model, rows, args.system)
        else:
            done_rows = {}
            if partial.exists():
                done_rows = {r["_idx"]: r for r in read_jsonl(partial)}
                print(f"  resuming from checkpoint: {len(done_rows)} rows already scored", flush=True)
            try:
                res = score_rows(tok, model, rows, args.system, max_tokens=args.max_tokens, done_rows=done_rows)
            except Interrupted as e:
                kept = [r for r in e.args[0] if r is not None]
                write_jsonl(partial, kept)
                print(f"checkpoint: {len(kept)}/{len(rows)} rows saved to {partial}; exiting", flush=True)
                sys.exit(EXIT_CHECKPOINTED)
        for r in res:
            r["arm"] = args.tag
        write_jsonl(out, res)
        if partial.exists():
            partial.unlink()
        meta.setdefault("files", {})[str(out)] = {"input": path, "n": len(res),
                                                   "input_sha256": sha256_file(path),
                                                   "seconds": round(time.time() - t0, 1)}
        print(f"wrote {out} ({len(res)} rows, {time.time() - t0:.0f}s)", flush=True)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    print(f"peak reserved {meta['peak_reserved_gb']} GB", flush=True)
    mpath = Path(args.out_dir) / f"{args.tag}__meta.json"
    old = json.loads(mpath.read_text()) if mpath.exists() else {}
    old.setdefault("runs", []).append(meta)
    mpath.write_text(json.dumps(old, indent=1) + "\n")


if __name__ == "__main__":
    main()
