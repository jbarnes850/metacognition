#!/usr/bin/env bash
# Every GPU job behind eos_results/, in dependency order. Hours on one GPU.
#
#   eos/run_gpu.sh [model]    model defaults to Qwen/Qwen3.5-4B; a local path also works.
#
# Adapters go to runs/<arm>/ and per-trial files to eos_results/scores/. A step whose output
# already exists is skipped, so the script can be restarted. Needs torch, transformers, and
# peft; the S2A rewrites also need sglang.
set -euo pipefail
cd "$(dirname "$0")/.."
M="${1:-Qwen/Qwen3.5-4B}"
S=eos_results/scores
T=eos_data/trials

run() {  # run <output> <command...>
  local out="$1"; shift
  if [ -e "$out" ]; then echo "skip $out"; return; fi
  echo "run $*"; "$@"
}
score() {  # score <tag> <trial-file stem> [--adapter runs/<arm>]
  local tag="$1" stem="$2"; shift 2
  run "$S/${tag}__${stem}.jsonl.gz" python3 -m eos.score --model "$M" --trials "$T/$stem.jsonl.gz" --out-dir "$S" --tag "$tag" "$@"
}
train() {  # train <arm>
  run "runs/$1/adapter_model.safetensors" python3 -m eos.train --model "$M" --train "$T/train_$1.jsonl.gz" --out "runs/$1" --micro 4
}

# 1. Readout check and the base model's answers to the bare questions.
python3 -m eos.build_trials smoke
run eos_results/readout_check.json python3 -m eos.check_readout "$M"
python3 -m eos.build_trials clean
score base clean_matched
score base clean_legacy

# 2. Evaluation and training sets. Own-answer trials start from the base answers above.
python3 -m eos.build_trials eval
python3 -m eos.build_trials redteam
python3 -m eos.build_trials train --arm main
python3 -m eos.build_trials train --arm nocue
python3 -m eos.build_trials train --arm bct_prep
python3 -m eos.build_trials bctown_prep
# Unrewritten S2A-subset prompts, scored by the trained arms for a same-trials comparison.
python3 -c "from eos import s2a; from eos.jsonl import write_jsonl; write_jsonl('$T/eval_s2a_subset.jsonl.gz',
  [{k: v for k, v in t.items() if k not in ('question', 'critique_expl')} for t in s2a.s2a_trials()])"

# 3. Training. Consistency labels are the base model's decisions on the cue-free prompts.
train main
train nocue
score base train_bct_none
python3 -m eos.build_trials bct_labels
train bct
score base train_bctown_none
python3 -m eos.build_trials bctown_labels
train bctown

# 4. Evaluation.
score base eval_core
score base eval_own
score base eval_redteam_own
for arm in main nocue bct bctown; do
  score "$arm" clean_matched --adapter "runs/$arm"
  score "$arm" eval_own --adapter "runs/$arm"
done
score main eval_core --adapter runs/main
score nocue eval_core --adapter runs/nocue
for arm in main bct bctown; do score "$arm" eval_redteam_own --adapter "runs/$arm"; done
for arm in main bct; do score "$arm" eval_s2a_subset --adapter "runs/$arm"; done

# 5. System 2 Attention baselines: rewrite with the base model, then answer from the rewrite.
for v in verbatim evidence; do
  arm=$([ "$v" = verbatim ] && echo s2a || echo s2aev)
  run "runs/s2a/rewrites_$v.jsonl.gz" python3 -m eos.s2a rewrite --variant "$v" --model "$M"
  run "$S/${arm}__eval_s2a.jsonl.gz" python3 -m eos.s2a score --variant "$v" --model "$M"
done
echo "run_gpu: done; now run eos/rerun.sh"
