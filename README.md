# Metacognitive Control in Language Models

Measuring whether language models know when to change their mind.

d-prime (signal detection theory) applied to belief revision: do models discriminate valid from invalid critique? Tested across Qwen3.5 0.8B-9B with domain-specific critiques from the [DS Critique Bank](https://huggingface.co/datasets/allenai/DS_Critique_Bank).

**Technical report**: [Do Language Models Know When to Change Their Mind?](https://jbarnes850.github.io/ai/research/2026/03/20/do-models-know-when-to-change-their-mind.html)

## Results

| Model | Accuracy | d-prime | 95% CI | Hit Rate | FA Rate |
|-------|----------|---------|--------|----------|---------|
| 0.8B  | 46.0%    | 1.243   | [0.74, 2.04] | 0.970 | 0.736 |
| 2B    | 66.7%    | 1.503   | [1.22, 1.76] | 0.990 | 0.797 |
| 4B    | 80.7%    | 1.604   | [1.11, 2.52] | 0.883 | 0.340 |
| 9B    | 89.3%    | 2.078   | [1.45, 3.09] | 0.853 | 0.152 |

d-prime scales with model size. The mechanism is asymmetric: false alarm rate collapses (0.74 to 0.15) while hit rate stays stable. Larger models resist bad evidence better, not accept good evidence better.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install mlx-lm datasets scipy scikit-learn matplotlib

# Download models
for size in 0.8B 2B 4B 9B; do
  hf download Qwen/Qwen3.5-${size} --local-dir models/Qwen3.5-${size}
done

# Behavioral sweep (d-prime across model sizes)
python sweep_critique.py

# Mechanistic probes (layer-wise correctness + appropriateness)
python probe.py

# Figures
python figures/fig1_dprime_scaling.py
python figures/fig2_direction_alignment.py
```

## Files

```
sweep_critique.py    Behavioral experiment: d-prime for belief revision
probe.py             Layer-wise linear probes for correctness and revision appropriateness
figures/             Visualization scripts
```

## Citation

```
@misc{barnes2026metacognition,
  author = {Barnes, Jarrod},
  title = {Do Language Models Know When to Change Their Mind?},
  year = {2026},
  url = {https://jbarnes850.github.io/ai/research/2026/03/20/do-models-know-when-to-change-their-mind.html}
}
```

## License

MIT
