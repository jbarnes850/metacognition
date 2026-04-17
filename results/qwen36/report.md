# Qwen3.6-35B-A3B on the Metacognition Benchmark

*Extension of [Do Language Models Know When to Change Their Mind?](https://jbarnes850.github.io/2026/03/20/do-models-know-when-to-change-their-mind.html) to the Qwen3.6 family. Inference run 2026-04-16 on DGX Spark (GB10) via SGLang 0.5.9-t5.*

## Headline

Qwen3.6-35B-A3B scores d-prime 1.347 on the DS Critique Bank matched pool, below both the 9B (1.785) and 4B (1.652) Qwen3.5 models, with the criterion-positive "resist everything" profile of Gemma 4 26B-A4B and a probe-direction cosine of 0.935. Scale within the Qwen family does not continue the U-shape recovery. The competence-before-control trajectory the blog documents from 0.8B to 9B inverts here into something new: near-complete integration of the correctness and appropriateness directions at the final layer, a collapsed confidence-FAR gap, and an E4B-style high-varentropy risk profile.

## Scaling table

| Model | Accuracy | N_sig | N_noi | d-prime | 95% CI | Hit Rate | FA Rate | c |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5 0.8B (blog) | 47.3% | 511 | 458 | 1.549 | [1.24, 2.17] | 0.993 | 0.820 | -1.69 |
| Qwen3.5 2B (blog) | 59.0% | 397 | 572 | 1.059 | [0.77, 1.54] | 0.986 | 0.873 | -1.67 |
| Qwen3.5 4B (blog) | 68.5% | 305 | 664 | 1.652 | [1.41, 1.96] | 0.956 | 0.521 | -0.88 |
| Qwen3.5 9B (blog) | 79.2% | 202 | 767 | 1.785 | [1.54, 2.09] | 0.924 | 0.361 | -0.54 |
| Gemma4 E4B (blog) | 71.5% | 276 | 693 | 1.818 | [1.59, 2.09] | 0.933 | 0.375 | -0.59 |
| Gemma4 26B-A4B (blog) | 78.2% | 210 | 758 | 1.636 | [1.43, 1.85] | 0.637 | 0.099 | +0.47 |
| **Qwen3.6 35B-A3B (this row)** | **83.7%** | **158** | **811** | **1.347** | **[1.13, 1.58]** | **0.619** | **0.148** | **+0.370** |

Accuracy climbs from 9B (79.2 percent) to 35B (83.7 percent) but d-prime drops from 1.785 to 1.347. The sign of the criterion flips between 9B (c = -0.54, biased toward revising) and 35B (c = +0.37, biased toward holding firm). The signal cell count is 158 here against 811 noise items, a consequence of the elevated base accuracy. Bootstrap CI half-width is 0.22, not tight enough to distinguish 35B from 4B on d-prime alone.

## Domain breakdown

| Model | Science d' (N_sig, N_noi) | Commonsense d' (N_sig, N_noi) |
|---|---|---|
| Qwen3.5 9B (blog) | 2.291 (41, 356) | 1.350 (161, 411) |
| Gemma4 E4B (blog) | 1.724 (72, 325) | 1.862 (204, 368) |
| Gemma4 26B-A4B (blog) | 1.825 (35, 362) | 1.428 (175, 396) |
| **Qwen3.6 35B-A3B** | **1.951 (33, 364)** | **1.043 (125, 447)** |

The science-over-commonsense gap (0.91) is larger than 9B's gap (0.94) in absolute terms, and the science d-prime 1.951 beats the 9B-to-35B-A3B scaling prediction only marginally while the commonsense d-prime 1.043 regresses below 9B's 1.35. Qwen3.6 does not clear the commonsense ceiling that blocked Qwen 9B. Within commonsense, SocialIQa produces a near-floor d-prime of 0.343 (N_sig=28, N_noi=66, CI [-0.22, 0.87]).

Per dataset, with blog 9B comparisons where informative:

| Dataset | Qwen3.6 d' | HR | FAR | N_sig | N_noi |
|---|---|---|---|---|---|
| ARC-Challenge | 1.689 | 0.589 | 0.072 | 27 | 229 |
| ARC-Easy | 2.739 | 0.786 | 0.026 | 6 | 135 |
| CosmosQA | 0.612 | 0.470 | 0.246 | 32 | 58 |
| HellaSwag | 1.041 | 0.643 | 0.250 | 20 | 125 |
| PIQA | 1.734 | 0.781 | 0.169 | 15 | 67 |
| SocialIQa | 0.343 | 0.500 | 0.366 | 28 | 66 |
| WinoGrande | 1.770 | 0.738 | 0.129 | 20 | 65 |
| aNLI | 1.584 | 0.773 | 0.201 | 10 | 66 |

ARC-Easy signal cell N=6 and aNLI signal cell N=10 are too small for stable bootstrap estimates; I report them for completeness.

## Confidence and varentropy conditioning

Entropy-conditioned d-prime, median split on initial-answer mean logprob:

| Split | d-prime | HR | FAR | N_sig | N_noi |
|---|---|---|---|---|---|
| Confident (above median) | 1.115 | 0.520 | 0.143 | 50 | 435 |
| Uncertain (below median) | 1.441 | 0.665 | 0.155 | 108 | 376 |

In the blog, 9B confident items resisted invalid critique at 78 percent (FAR 0.22) and uncertain items resisted at 49 percent (FAR 0.51), a 29-point FAR gap. Qwen3.6 shows a 1.2-point FAR gap (0.143 vs 0.155). The signal that drove the blog's "confidence predicts resistance" finding at 9B has collapsed at 35B.

Varentropy-conditioned FAR, median split on first-token varentropy on noise trials only:

| Varentropy split | FAR | N |
|---|---|---|
| High (above median) | 0.219 | 375 |
| Low (below median) | 0.087 | 436 |

High varentropy is a RISK factor for Qwen3.6, not a protective one. Qwen3.5 showed the opposite: high V protected against sycophancy. Gemma4 E4B inverted that. Qwen3.6 lands on the E4B side.

A 2x2 factorial cut on (entropy, varentropy) sharpens the pattern:

| cell | N | FAR |
|---|---|---|
| high_H high_V | 218 | 0.266 |
| high_H low_V | 188 | 0.074 |
| low_H high_V | 188 | 0.176 |
| low_H low_V | 217 | 0.069 |

Entropy and varentropy both contribute, roughly independently, to vulnerability. Low-H + low-V is the safe cell (FAR 0.069); high-H + high-V is the vulnerable one (FAR 0.266). Varentropy alone moves FAR by about 10 points at fixed entropy.

## Final-layer probes

| Probe | AUROC | 95% CI | N_items |
|---|---|---|---|
| Correctness | 0.776 | [0.738, 0.813] | 969 |
| Revision appropriateness | 0.746 | [0.709, 0.781] | 969 |

Final-layer cosine(correctness, appropriateness) = **0.935**.

The blog's direction-alignment trajectory on Qwen3.5 was -0.862 (0.8B), -0.74 to -0.79 (2B), 0.064 (4B), 0.298 (9B), each at the respective best layer. Qwen3.6 at the final layer reaches 0.935, roughly tripling the 9B value. The two representations have effectively collapsed into one direction at this scale.

The pooled probe AUROCs at 0.776 and 0.746 are close to 9B's 0.786 and 0.768 and do not reflect the discrimination collapse seen behaviorally. Put differently: a linear probe can still read "is this right" and "will the model handle critique appropriately" at 35B, but the model is no longer using the distinct signal during inference.

**Probe scope note.** I restricted probes to the final-layer residual stream at the final prompt token. The blog's per-layer best-layer search is not replicated here because the serving path (SGLang 0.5.9-t5) exposes only the final layer through `--enable-return-hidden-states`. Full transformers loading of the 67 GB BF16 VLM checkpoint OOMs the 128 GB UMA budget on GB10, even with `device_map="auto"` and explicit `max_memory` caps, so the per-layer pipeline is deferred. The 9B row in the blog's Table 4 also peaks at the final layer for both probes, so final-layer-only is a principled comparator for this scale.

## Phase 3 experiments

### E_task_decomp: task-dependent decomposition (Maskey et al., 2026)

I decomposed the appropriateness direction at the final layer into a global diff-of-means and domain-specific diff-of-means for science and commonsense.

| Quantity | Value |
|---|---|
| cos(global, science) | 0.493 |
| cos(global, commonsense) | 0.694 |
| cos(science, commonsense) | 0.415 |
| AUROC science, global probe trained on all | 0.763 |
| AUROC science, domain probe trained on science only | 0.876 |
| AUROC commonsense, global probe | 0.702 |
| AUROC commonsense, domain probe | 0.716 |
| AUROC science, cross-domain probe trained on com | 0.750 |
| AUROC commonsense, cross-domain probe trained on sci | 0.658 |

Science has more domain-specific structure than commonsense. Training a probe only on science items lifts science-item AUROC by 11 points over the global probe; the equivalent lift for commonsense is 1.4 points. The commonsense ceiling at this scale is consistent with Maskey et al.'s framing only weakly: there is little extra task-specific signal to recover in commonsense beyond the global direction. The bulk of what the global direction captures comes from commonsense items, which dominate the pool (572 vs 397).

Prediction from the blog: if the commonsense ceiling reflects a missing task-specific subspace, I expected cos(sci, com) < cos(global, sci). I find cos(sci, com) = 0.415 and cos(global, sci) = 0.493, so the prediction holds weakly. But AUROC_com_by_com barely exceeds AUROC_com_by_global (0.716 vs 0.702), which is the opposite of what a strong task-specific subspace account would require.

### E_var2d: varentropy x entropy factorial

Reported above under "Confidence and varentropy conditioning". The new result: Qwen3.6 flips the Qwen3.5 varentropy-protective pattern, and the flip is not driven by entropy alone. At fixed entropy the FAR gap between high-V and low-V is ~10 points in both bands. Qwen3.6's first-token distributions have structurally different shapes from Qwen3.5's, consistent with the architectural differences (new MoE topology E=256 top-8, `attn_output_gate`, 30 linear_attention + 10 full_attention layers in a 40-layer stack).

### E_cosine_domain: per-domain direction alignment

| Pair | Cosine |
|---|---|
| Correctness, appropriateness (pooled) | 0.935 |
| Correctness, appropriateness (science only, N=397) | 0.691 |
| Correctness, appropriateness (commonsense only, N=572) | 0.805 |

The prediction I carried in from the blog was cos_commonsense < cos_science, on the argument that the 9B commonsense ceiling might reflect failed integration of confidence and control on commonsense specifically. Qwen3.6 shows the opposite: commonsense has tighter integration than science. The pooled value (0.935) exceeds both domain-restricted values, which is an artifact of the different base rates across domains (the global diff-of-means picks up both the appropriateness signal and domain identity).

### E_conf_gate: confidence-gates-FAR replication (Kadavath 2022; Kumaran 2026)

| Split | FAR | N_noise |
|---|---|---|
| Overall confident | 0.143 | 435 |
| Overall uncertain | 0.154 | 376 |
| Overall gap | 0.012 |  |
| Science confident | 0.036 | 222 |
| Science uncertain | 0.099 | 142 |
| Commonsense confident | 0.253 | 213 |
| Commonsense uncertain | 0.183 | 234 |

The blog's 9B pooled gap was 0.29; Qwen3.6's is 0.012. Within science the gap is 6 points (uncertain items are more FAR-prone). Within commonsense the gap is -7 points (uncertain items are LESS FAR-prone than confident items, which is an inversion). The confidence-gates-revision mechanism that opened up between 0.8B and 9B has closed again at 35B. Whether this is because the internal confidence signal has weakened or because the model no longer conditions revision on it is not distinguishable from behavioral data alone.

## What this leaves open

The blog's final paragraph flags causal intervention as the next experiment: steer the appropriateness direction in activation space and measure whether FAR drops selectively. I did not run this. Two reasons. First, SGLang does not expose a mid-network steering hook through its HTTP API, so steering requires either patching the server or running an offline forward-hook client. Second, the offline client needs a second copy of the 67 GB BF16 model alongside the running SGLang server, which exceeds the 128 GB UMA budget. An implementation that switches between serving and steering (stop SGLang, run a hook-enabled script, restart SGLang) fits in budget but was deprioritized to keep the Phase 2 replication complete. The infrastructure scripts in `qwen36_phase3.py` implement the forward-hook mechanics and can be revived once the transformers-side loading issue is resolved (flash-linear-attention plus causal-conv1d plus quantization offload, or an FSDP-sharded load).

Four other experiments from the metaprompt's candidate list are also not run: per-expert routing analysis (requires `--enable-return-routed-experts` on the server and a restart), verbal-confidence probe (requires a new inference pass with an explicit rating prompt), sycophancy circuit decomposition (needs per-layer probes), and post-answer cached-confidence probes (same constraint).

## Limitations

**Single architecture family point.** One 35B model does not make a scaling law within Qwen. I report a single new row.

**Signal cell asymmetry.** Accuracy at 83.7 percent concentrates items in the noise condition (811 vs 158). Per-dataset signal cells for ARC-Easy (6) and aNLI (10) are too small for stable estimation, and I flag them rather than rely on them.

**Final-layer probes only.** I do not have per-layer AUROCs, so the probe trajectory across depth is not comparable to the blog's 9B result. The final-layer cosine 0.935 is a single number; whether alignment peaks there or earlier is not tested.

**Inference stack changed.** The blog used mlx-lm on Apple Silicon. I use SGLang 0.5.9-t5 on GB10 with fp8 KV cache and triton attention backend. Numerical differences at fp8 versus mlx-lm's bfloat16 KV are plausible drivers of small effects; the d-prime CI bracketing 1.34 is wide enough to absorb them.

**Thinking mode disabled.** Following the blog, I disable CoT via the `<think>\n</think>\n` suppression. Qwen3.6's reasoning mode behavior on this task is unknown.

**Correlational.** Every probe result is correlational. The high cosine 0.935 says the two diff-of-means directions point the same way, not that steering one changes the other behaviorally.

## Files

All artifacts under `results/qwen36/`:

- `qwen36_summary.json` — d-prime, hit/FA, per-dataset, domain, entropy- and varentropy-conditioned splits, final-layer probe AUROCs and cosine
- `qwen36_phase1.json` — per-item initial answer, logprobs, entropy, varentropy (969 items)
- `qwen36_phase2.json` — per-item critique trial, revised answer, SDT label (969 trials)
- `qwen36_phase3.json` — task decomposition, var2d, domain cosine, confidence gate, response-length by SDT
- `qwen36_hidden_final.npz` — (969, 2048) final-layer residual stream at final prompt token

Scripts under repo root: `qwen36_phase2_sglang.py`, `qwen36_phase3_analytical.py`, `qwen36_phase2.py` (transformers path, unused), `qwen36_phase3.py` (forward-hook scaffold, unused).
