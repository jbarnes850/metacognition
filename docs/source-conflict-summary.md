# Source-Conflict Extension

This note records the source-monitoring extension added after the Qwen3.6
and Gemma4 runs.

The original benchmark asks whether a model changes its answer when the
critique is valid and resists when the critique is invalid. The extension
keeps the critique text fixed and adds a reviewer-panel cue after the
critique. The cue is either congruent with critique validity or in conflict
with it.

- `congruent` means the panel recommends the action implied by critique
  validity. Valid critique gets a change recommendation. Invalid critique gets
  a keep recommendation.
- `conflict` means the panel recommends the opposite action. Valid critique
  gets a keep recommendation. Invalid critique gets a change recommendation.

This separates evidence quality from source pressure. A model that tracks the
critique should retain high d-prime under conflict. A model that follows the
source should lose discrimination.

## Main Takeaway

The conflict condition is the useful one. Qwen models lose much of their
critique discrimination when the social source disagrees with the evidence.
Gemma4 26B-A4B also degrades, but retains more of the evidence signal.

| Model | Baseline d-prime | Conflict 5-2 d-prime | Conflict 4-3 d-prime |
|---|---:|---:|---:|
| Gemma4 26B-A4B | 1.479 [1.156, 1.881] | 0.597 [0.248, 0.990] | not run |
| Qwen3.5 9B | 1.527 [1.169, 1.942] | -0.104 [-0.420, 0.209] | 0.630 [0.302, 0.955] |
| Qwen3.6 35B-A3B | 1.222 [0.908, 1.592] | 0.326 [-0.024, 0.672] | 0.665 [0.344, 1.008] |

The result changes the interpretation of the original benchmark. The failure
is not only that models cave to critique or stubbornly hold their prior. The
sharper failure is source confusion, where the model does not cleanly separate
the quality of the evidence from the authority of the source carrying it.

## Reproduce

Run the source-conflict extension from the repository root.

```bash
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

The script writes timestamped JSON files with trial records, bootstrap
confidence intervals, and summary metrics.
