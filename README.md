# Metacognition Benchmark

This repository measures whether language models know when to change their
mind. The core task is simple: a model answers a multiple-choice question, sees
a critique, and then either revises or holds its original answer. The important
part is that critiques vary in quality. Some are valid and should cause a
revision. Others are plausible but wrong and should be resisted.

The benchmark treats belief revision as a signal detection problem. Valid
critique is the signal. Invalid critique is the noise. The model's response is
whether it changes its answer. This lets us distinguish three behaviors that
ordinary accuracy hides:

- caving to every critique
- resisting every critique, even when correction is warranted
- discriminating good evidence from bad evidence

The accompanying write-up is here:
[Do Language Models Know When to Change Their Mind?](https://jbarnes850.github.io/2026/03/20/do-models-know-when-to-change-their-mind.html)

## What Is In This Repo

The repo contains a matched evaluation registry, MLX-based local inference
runners, behavior metrics, simple mechanistic probes, uncertainty analyses, and
a source-monitoring extension.

The source-monitoring extension asks a second question: did the model update
because the evidence was good, or because a social source told it to? It keeps
the critique text fixed and adds a reviewer-panel cue that either agrees or
conflicts with critique validity. See
[`docs/source-conflict-summary.md`](docs/source-conflict-summary.md).

## Main Results

The main behavioral sweep covers 969 items from the DS Critique Bank across
eight datasets. The table reports d-prime, a signal detection measure of how
well the model separates valid critique from invalid critique.

| Model | Accuracy | d-prime | 95% CI | Hit Rate | FA Rate | Criterion c |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 0.8B | 47.3% | 1.549 | [1.24, 2.17] | 0.993 | 0.820 | -1.69 |
| Qwen3.5 2B | 59.0% | 1.059 | [0.77, 1.54] | 0.986 | 0.873 | -1.67 |
| Qwen3.5 4B | 68.5% | 1.652 | [1.41, 1.96] | 0.956 | 0.521 | -0.88 |
| Qwen3.5 9B | 79.2% | 1.785 | [1.54, 2.09] | 0.924 | 0.361 | -0.54 |
| Gemma4 E4B | 71.5% | 1.818 | [1.59, 2.09] | 0.933 | 0.375 | -0.59 |
| Gemma4 26B-A4B | 78.2% | 1.636 | [1.43, 1.85] | 0.637 | 0.099 | +0.47 |

The short version:

- Competence scales before control. Qwen3.5 2B is more accurate than 0.8B but
  more vulnerable to invalid critique.
- Architecture changes the failure mode. Gemma4 E4B discriminates well, while
  Gemma4 26B-A4B is much more conservative.
- Source pressure can erase discrimination. Under conflict cues, Qwen3.5 9B
  falls near zero d-prime, while Qwen3.6 and Gemma4 26B retain partial signal.

## Repository Layout

```text
build_registry.py                 Build the matched 969-item registry
sweep_fullpool.py                 Qwen3.5 full-pool behavioral sweep
sweep_gemma4.py                   Gemma4 E4B and 26B-A4B behavioral sweep
qwen36_phase2.py                  Qwen3.6 Transformers evaluation harness
qwen36_phase2_sglang.py           Qwen3.6 SGLang evaluation harness
qwen36_phase3_analytical.py       Qwen3.6 analytic follow-up metrics
social_source_monitoring.py       Source-conflict extension
probe.py                          Layer-wise linear probes
varentropy_test.py                Qwen answer-token uncertainty analysis
varentropy_gemma4.py              Gemma4 answer-token uncertainty analysis
answer_blind_gate.py              Construct validation without answer letter
config/models.env.example         Model path environment template
figures/                          Figure generation scripts and outputs
registry/                         Matched item registry and manifest
results/qwen36/                   Tracked Qwen3.6 summary artifacts
docs/source-conflict-summary.md   Source-monitoring experiment note
tests/                            Unit tests for reusable analysis logic
```

Generated sweep outputs live under `results/` and are ignored by default unless
explicitly tracked.

## Setup

This repo is designed for local inference. Most scripts use MLX on Apple
Silicon. The Qwen3.6 phase-2 scripts also include Transformers and SGLang
paths for users who want to reproduce that part on another runtime.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model weights are not vendored in this repository. Put checkpoints under
`models/`, use symlinks, or point the scripts at an existing model cache.
`config/models.env.example` contains the environment variables used by the
source-monitoring and Qwen3.6 runners.

```bash
cp config/models.env.example .env.models
source .env.models
```

Default model path contract:

```text
METACOG_MODEL_ROOT/Qwen3.5-0.8B
METACOG_MODEL_ROOT/Qwen3.5-2B
METACOG_MODEL_ROOT/Qwen3.5-4B
METACOG_MODEL_ROOT/Qwen3.5-9B
METACOG_MODEL_ROOT/qwen3.6-35b-a3b-mlx-int6
METACOG_MODEL_ROOT/gemma-4-E4B-it-mlx-int6
METACOG_MODEL_ROOT/gemma-4-26b-a4b-it-mlx-int6
```

For Qwen3.6 Transformers or SGLang runs, set `QWEN_MODEL_PATH` and
`QWEN_OUTPUT_DIR`. The default output directory is `results/qwen36`.

## Reproduce

Build the matched item registry first.

```bash
python build_registry.py
```

Run the main behavioral sweeps.

```bash
python sweep_fullpool.py
python sweep_gemma4.py
```

Run the Qwen3.6 follow-up.

```bash
python qwen36_phase2.py
python qwen36_phase3_analytical.py
```

Run the source-monitoring extension.

```bash
source .env.models

python social_source_monitoring.py \
  --models Qwen3.5-9B,Qwen3.6-35B-A3B,Gemma4-26B-A4B-IT \
  --conditions congruent,conflict \
  --max-per-class 120 \
  --panel-strength 5-2

python social_source_monitoring.py \
  --models Qwen3.5-9B,Qwen3.6-35B-A3B \
  --conditions congruent,conflict \
  --max-per-class 120 \
  --panel-strength 4-3 \
  --output-dir results/social_source_monitoring/weak_panel_43
```

Run the mechanistic and uncertainty analyses.

```bash
python probe.py
python varentropy_test.py
python varentropy_gemma4.py
```

Generate figures.

```bash
python figures/fig_dprime_scaling_v5.py
python figures/fig_domain_architecture_v3.py
python figures/fig_varentropy_fa.py
```

## Validation

Fast checks that do not require model inference:

```bash
python -m py_compile social_source_monitoring.py tests/test_social_source_monitoring.py
python -m unittest tests/test_social_source_monitoring.py
ruff check .
```

Use `--dry-run` to validate source-monitoring prompt construction and file
output paths without spending inference time.

```bash
python social_source_monitoring.py --dry-run --models Qwen3.5-9B
```

## Reproducibility Notes

- The DS Critique Bank data is loaded from Hugging Face through `datasets`.
- Large generated outputs under `results/` are ignored by default. Re-running
  the scripts will recreate them.
- `results/qwen36/report.md`, `qwen36_phase3.json`, and `qwen36_summary.json`
  are tracked as compact reference artifacts. Full Qwen3.6 phase-1, phase-2,
  and hidden-state files are regenerated by `qwen36_phase2.py` or
  `qwen36_phase2_sglang.py`.
- Published source-conflict numbers are based on timestamped JSON outputs from
  `social_source_monitoring.py`; the script records trial-level rows and
  bootstrap confidence intervals in each output file.

## Citation

```bibtex
@misc{barnes2026metacognition,
  author = {Barnes, Jarrod},
  title = {Do Language Models Know When to Change Their Mind?},
  year = {2026},
  url = {https://jbarnes850.github.io/2026/03/20/do-models-know-when-to-change-their-mind.html}
}
```

## License

MIT
