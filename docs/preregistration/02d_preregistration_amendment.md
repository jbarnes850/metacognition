# Pre-registration amendment 3: a consistency-training arm (BCT)

Written 2026-10-17 about 23:25, after the main arm's own-answer results and before any BCT run. The BCT arm is a new, post-hoc arm: its choice was informed by the main result, but its hypotheses below are fixed before it is trained or scored.

**What the main arm did.** On own-answer trials the main adapter's final answer equals its own question-only answer in 96.2% of trials (base: 62%). It revised to gold in 33 of 34 valid-critique trials where its clean answer was already gold and in 6 of 52 where it was not. No-cue hit rate fell from 0.83 to 0.48 (H3 fails). The recipe taught the model to re-answer from its own knowledge and ignore the follow-up turn, critique and cue alike. Gold-answer labels reward that policy, because on most crossed training trials the model already knows the answer.

**New arm, `bct`.** Bias-augmented consistency training (Chua et al., arXiv 2403.05518) on the same 1,880 training prompts as main (identical text and order). The label for each prompt is the base model's decision on the same prompt with the cue removed (critique unchanged). Everything else matches main (LoRA rank 16, lr 1e-4, batch 16, one pass, seed 0, micro-batch 4). The label never uses gold, so re-answering is not rewarded unless the base model itself re-answers without the cue.

**Hypotheses for `bct` (own-answer design, held-out items, item bootstrap):**

- B1, cue invariance on unseen cues: conflict d' pooled over held-out families at strengths 2 and 4 exceeds base's (paired interval excluding zero), and is within 0.2 of base no-cue d' (1.53).
- B2, evidence use kept: no-cue hit rate at least base minus 0.05 (0.78), and the share of own-answer trials whose final answer equals the arm's own clean answer stays well below main's 96.2% (reported; no threshold).
- B3, no capability cost: clean held-out accuracy within 3 points of base.
- Secondary: the same metrics on the post-freeze red-team families, and the crossed core set.
