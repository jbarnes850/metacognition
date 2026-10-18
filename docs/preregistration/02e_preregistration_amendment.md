# Pre-registration amendment 4: BCT on own-answer prompts (`bctown`)

Written 2026-10-18 about 01:45, after the BCT own-answer results and before any `bctown` run.

**Motivation.** BCT met B1 to B3, but its no-cue false-alarm rate on own-answer trials rose from 0.29 (base) to 0.44. Its labels came from crossed prompts, where the assigned initial answer is often not the model's own, and on those prompts the base model with no cue accepts 72% of invalid critiques. BCT copied that more agreeable behavior.

**Arm `bctown`.** Same as BCT except the training prompts use the base model's own initial answer on each training item (wrong gives a valid critique, right gives an invalid one; same-student explanation pairs; cues sampled as in `train_main`; 8 draws per item, 1,880 prompts). The label is the base model's decision on the same prompt with the cue removed.

**Hypotheses (own-answer design, held-out items):**

- O1: conflict d' pooled over held-out families at strengths 2 and 4 is within 0.2 of base no-cue d' (1.53) and above base's conflict d' (paired interval excluding zero).
- O2: no-cue false-alarm rate at most base + 0.05 (0.34) and no-cue hit rate at least base - 0.05 (0.78).
- O3: clean held-out accuracy within 3 points of base.
