"""LoRA decision fine-tuning (runs in the CUDA container).

The decision is the first assistant token. The loss is the negative log of the
probability mass on the label letter (tokens "L", "(L", " L"), matching the
scoring readout in eos/score.py. Because the action is a single token with a
known correct answer, this equals maximizing the log-probability of the correct
revise-or-resist action; no sampling is involved.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

from eos.score import letter_ids, load_model, read_jsonl, render, sha256_file

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",            # full-attention layers
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",  # Gated DeltaNet layers
    "gate_proj", "up_proj", "down_proj",                # MLP
]


EXIT_CHECKPOINTED = 75
STOP = {"flag": False}


def _on_term(signum, frame):
    STOP["flag"] = True


def main():
    import shutil
    import signal

    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    signal.signal(signal.SIGTERM, _on_term)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke runs only")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true", help="trade speed for memory")
    ap.add_argument("--mem-cap-gb", type=float, default=40.0)
    args = ap.parse_args()
    from eos.score import cap_memory
    cap_memory(args.mem_cap_gb)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok, model = load_model(args.model)
    from peft import LoraConfig, get_peft_model

    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    targets = [t for t in TARGET_MODULES if t in present]
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                     target_modules=targets, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    rows = read_jsonl(args.train)
    ids = letter_ids(tok)
    texts = [render(tok, r["messages"], None) for r in rows]
    label_ids = [ids[r["label"]] for r in rows]

    n_steps = math.ceil(len(rows) * args.epochs / args.batch)
    if args.max_steps:
        n_steps = min(n_steps, args.max_steps)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    warm = max(1, int(0.05 * n_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / n_steps))))

    order = list(range(len(rows)))
    log, cursor, t0, start = [], 0, time.time(), 0
    ckpt = out / "ckpt" / "state.pt"
    resumed_from = None
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, st["lora"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        random.setstate(st["py_rng"])
        torch.set_rng_state(st["torch_rng"])
        torch.cuda.set_rng_state_all(st["cuda_rng"])
        start, cursor, order, log = st["step"], st["cursor"], st["order"], st["log"]
        resumed_from = start
        print(f"resuming from checkpoint at step {start}", flush=True)
    model.train()
    def save_ckpt(at_step, at_cursor, at_order, rng):
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"step": at_step, "cursor": at_cursor, "order": at_order, "log": log,
                    "py_rng": rng[0], "torch_rng": rng[1], "cuda_rng": rng[2], "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "lora": get_peft_model_state_dict(model)}, ckpt)
        print(f"checkpoint saved at step {at_step}; exiting", flush=True)
        sys.exit(EXIT_CHECKPOINTED)

    for step in range(start, n_steps):
        # Snapshot at the step boundary: weights only change at opt.step(), so a stop mid-step
        # can save this snapshot and resume exactly from the start of the step.
        snap = (cursor, list(order), (random.getstate(), torch.get_rng_state(), torch.cuda.get_rng_state_all()))
        opt.zero_grad(set_to_none=True)
        tot_loss, tot_acc = 0.0, 0.0
        for _ in range(args.batch // args.micro):
            if STOP["flag"]:
                save_ckpt(step, snap[0], snap[1], snap[2])
            if cursor + args.micro > len(order):
                random.shuffle(order)
                cursor = 0
            mb = order[cursor:cursor + args.micro]
            cursor += args.micro
            enc = tok([texts[i] for i in mb], return_tensors="pt", padding=True,
                      pad_to_multiple_of=32, add_special_tokens=False).to(model.device)
            logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            lp_label = torch.stack([torch.logsumexp(logp[j, label_ids[i]], dim=0) for j, i in enumerate(mb)])
            loss = -lp_label.mean() / (args.batch // args.micro)
            loss.backward()
            tot_loss += float(loss.detach()) * (args.batch // args.micro)
            tot_acc += float((lp_label > math.log(0.5)).float().mean())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        k = args.batch // args.micro
        rec = {"step": step + 1, "loss": tot_loss / k, "p_label_gt_half": tot_acc / k,
               "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        rec["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 1)
        log.append(rec)
        if step % 10 == 0 or step == n_steps - 1:
            print(json.dumps(rec), flush=True)
        if STOP["flag"] and step + 1 < n_steps:
            save_ckpt(step + 1, cursor, order, (random.getstate(), torch.get_rng_state(),
                                                torch.cuda.get_rng_state_all()))

    model.save_pretrained(out)
    meta = {"args": vars(args), "targets": targets, "n_rows": len(rows), "n_steps": n_steps,
            "train_sha256": sha256_file(args.train),
            "adapter_sha256": sha256_file(out / "adapter_model.safetensors"),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "resumed_from_step": resumed_from,
            "peak_reserved_gb": max((r.get("peak_reserved_gb") or 0) for r in log) if log else None}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    (out / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    if ckpt.parent.exists():
        shutil.rmtree(ckpt.parent)
    print("saved", out, meta["adapter_sha256"][:16])


if __name__ == "__main__":
    main()
