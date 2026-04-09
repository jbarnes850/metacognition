# Metacognitive Control in Language Models

Measuring whether language models know when to change their mind.

d-prime (signal detection theory) applied to belief revision: do models discriminate valid from invalid critique? Tested across four Qwen3.5 sizes (0.8B-9B) and two Gemma 4 architectures (E4B, 26B-A4B) with domain-specific critiques from the [DS Critique Bank](https://huggingface.co/datasets/allenai/DS_Critique_Bank).

**Technical report**: [Do Language Models Know When to Change Their Mind?](https://jbarnes850.github.io/2026/03/20/do-models-know-when-to-change-their-mind.html)

## Results (969 items, 8 datasets)

| Model | Accuracy | d-prime | 95% CI | Hit Rate | FA Rate | Criterion c |
|-------|----------|---------|--------|----------|---------|-------------|
| Qwen3.5 0.8B | 47.3% | 1.549 | [1.24, 2.17] | 0.993 | 0.820 | -1.69 |
| Qwen3.5 2B | 59.0% | 1.059 | [0.77, 1.54] | 0.986 | 0.873 | -1.67 |
| Qwen3.5 4B | 68.5% | 1.652 | [1.41, 1.96] | 0.956 | 0.521 | -0.88 |
| Qwen3.5 9B | 79.2% | 1.785 | [1.54, 2.09] | 0.924 | 0.361 | -0.54 |
| Gemma4 E4B | 71.5% | 1.818 | [1.59, 2.09] | 0.933 | 0.375 | -0.59 |
| Gemma4 26B-A4B | 78.2% | 1.636 | [1.43, 1.85] | 0.637 | 0.099 | +0.47 |

Scaling is not monotonic: the 2B model is the worst discriminator (d-prime 1.059), worse than 0.8B. Competence scales before control does. Architecture determines failure mode independent of scale: E4B discriminates, 26B-A4B resists everything.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install mlx-lm mlx-vlm datasets scipy scikit-learn matplotlib

# Download Qwen models
for size in 0.8B 2B 4B 9B; do
  huggingface-cli download Qwen/Qwen3.5-${size} --local-dir models/Qwen3.5-${size}
done

# Build matched item registry (969 items across 8 datasets)
python build_registry.py

# Behavioral sweep: Qwen3.5 (all sizes, full pool)
python sweep_fullpool.py

# Behavioral sweep: Gemma 4 (requires mlx-vlm)
python sweep_gemma4.py

# Mechanistic probes (layer-wise correctness + appropriateness)
python probe.py

# Varentropy analysis
python varentropy_test.py        # Qwen models
python varentropy_gemma4.py      # Gemma 4 E4B

# Figures
python figures/fig_dprime_scaling_v5.py
python figures/fig_domain_architecture_v3.py
python figures/fig_varentropy_fa.py
```

## Files

```
sweep_fullpool.py      Full-pool behavioral sweep (969 items, Qwen3.5)
sweep_gemma4.py        Gemma 4 behavioral sweep (E4B + 26B-A4B)
sweep_critique.py      Original 200-item sweep (superseded by sweep_fullpool.py)
sweep_thinking.py      Thinking-mode ablation (4B, 9B)
probe.py               Layer-wise linear probes for correctness and revision appropriateness
varentropy_test.py     Varentropy analysis (Qwen models, 150 ARC items)
varentropy_gemma4.py   Varentropy analysis (Gemma 4 E4B)
build_registry.py      Builds matched item registry from DS Critique Bank
answer_blind_gate.py   Construct validation: d-prime without answer letter
registry/              Matched item registry (969 items, 8 datasets)
figures/               Figure generation scripts
```

## Citation

```
@misc{barnes2026metacognition,
  author = {Barnes, Jarrod},
  title = {Do Language Models Know When to Change Their Mind?},
  year = {2026},
  url = {https://jbarnes850.github.io/2026/03/20/do-models-know-when-to-change-their-mind.html}
}
```

## License

MIT
