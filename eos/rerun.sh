#!/usr/bin/env bash
# Recompute every reported number of the evidence-over-source study.
#
#   eos/rerun.sh            CPU only, a few minutes. Stimulus audit, analysis, results report,
#                           and figures from the committed per-trial files (eos_results/scores/)
#                           and the earlier sweep's per-trial files (eos_data/legacy_raw/), then a
#                           check that the recomputed analysis matches the committed copy.
#   eos/rerun.sh --rescore  First rebuilds the items from DS Critique Bank (checked against the
#                           frozen file) and reruns every GPU job with eos/run_gpu.sh. Hours on
#                           one GPU; set MODEL to a local copy of Qwen/Qwen3.5-4B if needed.
set -euo pipefail
cd "$(dirname "$0")/.."

check_sums() {  # check_sums <dir>: verify <dir>/SHA256SUMS without relying on sha256sum or shasum
  python3 - "$1" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
for line in (root / "SHA256SUMS").read_text().splitlines():
    digest, name = line.split(maxsplit=1)
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
        sys.exit(f"hash mismatch: {root / name}")
PY
}

if [ "${1:-}" = "--rescore" ]; then
  frozen=$(python3 -c "import hashlib; print(hashlib.sha256(open('eos_data/items_v1.json','rb').read()).hexdigest())")
  cp eos_data/items_v1.json eos_data/items_v1.frozen.json
  python3 -c "from eos import stimuli; stimuli.save_items(stimuli.build_items())"
  now=$(python3 -c "import hashlib; print(hashlib.sha256(open('eos_data/items_v1.json','rb').read()).hexdigest())")
  if [ "$frozen" != "$now" ]; then
    mv eos_data/items_v1.frozen.json eos_data/items_v1.json
    echo "rebuilt items differ from the frozen file; restored the frozen file"; exit 1
  fi
  rm eos_data/items_v1.frozen.json
  mv eos_results/scores "eos_results/scores.committed.$(date +%s)"
  bash eos/run_gpu.sh "${MODEL:-Qwen/Qwen3.5-4B}"
fi

check_sums eos_data/legacy_raw
python3 -m eos.audit eos_results/stimulus_audit.json > /dev/null
python3 -m eos.analyze
python3 -m eos.report
if [ "${1:-}" != "--rescore" ]; then
  git diff --quiet -- eos_results/analysis.json eos_results/numbers.json eos_results/stimulus_audit.json \
    eos_results/legacy_false_flaw_qids.json \
    || { echo "recomputed results differ from the committed files:"; git diff --stat -- eos_results; exit 1; }
  echo "rerun: analysis, numbers, stimulus audit, and false-flaw list match the committed files"
fi
