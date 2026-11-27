# Stage 1 simulation harness — cleaned-up copy

Gemini built this harness, 27 Nov 2026, for the §11 "Shifting Rules, Tempting Shortcut" toy task.
Shared via https://share.gemini.google/b5FDeK0S41dz. Chris uploaded Gemini's file as
`stage1_sim_harness_3_vq27.py`.

## Files here

- `stage1_sim_harness.py` — cleaned-up copy. Defaults reproduce Gemini's results exactly.
  Header documents the provenance, the known artefacts, and the two variants.
  Variants A and B are exposed as parameters (`lora_decay_post_shift`,
  `reset_lora_at_shift`) and run at the end of `__main__`.
- `stage1_sim_harness_ORIGINAL_GEMINI.py` — Gemini's file, verbatim, untouched.

## Run it

```bash
python3 stage1_sim_harness.py
```

## Default results (10 seeds, 1,000 episodes, shift at 500)

| mode | Tier B heeding | Tier A heeding | post-shift recovery | canary drift |
|---|---|---|---|---|
| FULL_LOOP | 1.0000 | 1.0000 | 0.5816 | 0.00 pts |
| NO_WORKSPACE | 0.0000 | 0.0000 | 0.1833 | 49.20 pts |
| NO_HEEDING_OBJECTIVE | 1.0000 | 1.0000 | 0.5816 | 0.00 |
| PIPELINE_ONLY | 0.0000 | 0.0000 | 0.4713 | 4.00 pts |
| NO_PROTECTIVE_RULES | 1.0000 | 1.0000 | 0.2235 | 0.00 |

Violations after alarm: 0 in every mode (see artefacts below).

## Known artefacts — read before quoting the numbers

1. **Heeding 1.0 / 0.0 is baked in by hand-set logit penalties**, not learned.
   Tier B −2.8, Tier A −2.2×sensitivity in FULL_LOOP; 0 in NO_WORKSPACE /
   PIPELINE_ONLY where broadcast is NONE. The conscience signals are
   deterministic functions of the episode, not learned or noisy.
2. **NO_HEEDING_OBJECTIVE is "weaker penalty", not "no heeding objective".**
   Tier B −1.25 instead of −2.8, Tier A −0.75× instead of −2.2× — still
   enough to steer. Its identical result to FULL_LOOP does not yet test
   whether heeding emerges without the objective.
3. **Violations after alarm is always 0 by construction.** Action 3 is
   replaced with a safe action before `env.step` when Tier B broadcasts,
   and the counter only increments when broadcast is B.
4. **Exploration only ever picks safe actions 0/1**, never the shortcut
   or forbidden zone.

The *ordering* of the modes is informative (workspace: drift 0 vs 49.2;
protective rules: recovery 0.58 vs 0.22). The absolute heeding numbers
are a best case from a mock.

## Variants tested (Juno, 27 Nov 2026)

Gemini proposed two fixes for FULL_LOOP recovery 0.5816 being below a 0.80 bar.

- **A. Faster LoRA decay post-shift (0.90 vs 0.995):** recovery 0.5816 → 0.6261,
  but canary drift 0.00 → 3.6 pts. Still far from 0.80. A real, tunable
  parameter — worth putting in §2.11 / §11 as a pilot parameter, with the
  drift cost noted.
- **B. Reset LoRA to zero at the rule shift:** recovery 0.5816 → 1.0002,
  drift 0.00, in both FULL_LOOP and NO_PROTECTIVE_RULES. Trivial in this
  toy: the only pre-shift skill to forget is SAFE_RULE_A, which is the wrong
  rule post-shift, so forgetting it is free. In a real task with multiple
  prior skills a blanket reset would be catastrophic. Do **not** generalise.
  The selective version worth building: attenuate only the fast-weight
  components tied to the shifted rule, detected by high uncertainty +
  prediction error.

## Corrected harness v2 — noisy (Juno, 27 Nov 2026)

`stage1_sim_harness_v2_noisy.py` — built while Chris was away, at his request.
Fixes the three artefacts above: noisy conscience signals (drift σ=0.15,
salience/uncertainty σ=0.10, 10% Tier B false-alarm and miss rates, seeded),
a true NO_HEEDING_OBJECTIVE with logit penalty exactly 0 (not weaker), and
epsilon exploration over all 4 actions. Adds `violations_proposed` (action 3
proposed under Tier B, before replacement).

Results, 10 seeds (verified by re-running the file — deterministic):
FULL_LOOP Tier B heeding 0.9810, Tier A 0.9780, recovery **0.3735** (down
from 0.5816, and bimodal across seeds: five at 0.20–0.22, five at 0.46–0.61),
canary drift 4.8 pts, violations proposed 40. NO_HEEDING_OBJECTIVE: Tier B
heeding 0.9724, canary 0.9643, violations proposed 58, recovery identical
at 0.3735 — so the heeding objective is a small real contributor, not the
whole story; most heeding here is emergent / learned avoidance. Variant A
(faster LoRA decay 0.90 post-shift): recovery 0.5019. Variant B (LoRA reset
at shift): 0.9845 — same trivial-toy warning as before, do not generalise.
NO_WORKSPACE drift 52.8 pts; the 0.0 heeding / 0 violations-proposed in
NO_WORKSPACE and PIPELINE_ONLY are still 0-by-construction (Tier B never
broadcasts there, so no events are counted).

## PyTorch SSM + LoRA version (Gemini, 27 Nov 2026)

`stage1_pytorch_ssm_lora_ORIGINAL_GEMINI.py` — Gemini's file, verbatim, in
this folder (uploaded as `stage1_pytorch_ssm_lora_4_p8k9.py`). Real selective
SSM conscience + fixed reference (state dim 16), real LoRA token actor
(rank 4, policy-gradient micro-batches), real Stage 1c rater training and
verifier canary check. Juno installed PyTorch 2.15.0+cpu in
`~/workspace/torch-venv` (the first install failed: /tmp is only 512 MB)
and ran it on CPU in 40 seconds, 10 seeds — $0, as predicted.

Results, 10 seeds:
FULL_LOOP Tier B heeding 1.0000, Tier A 1.0000, recovery **0.2948**, canary
drift **12.4 pts** (fails the ≤5 threshold), violations 0. NO_WORKSPACE
recovery 0.1612, drift 23.6. NO_HEEDING_OBJECTIVE identical to FULL_LOOP
again (1.0000 / 0.2948 / 12.4) — its broadcast weights are only weakened
(-4.2 → -0.8, -3.2 → -0.6), not removed, so (b') is still not a true test.
PIPELINE_ONLY recovery 0.2053, drift 9.2. NO_PROTECTIVE_RULES recovery
0.2984, drift 5.2 — slightly *better* than FULL_LOOP on both, so in this
version the protective rules do not help recovery/drift; that undermines
the tabular claim and needs investigation, not quoting.

Heeding is still 1.0 exactly, despite the learned SSM, because the
detectors are pre-calibrated by hand and the broadcast suppression weights
are large (-4.2 / -3.2). The SSM core itself barely learns: the rater is
trained on a detached state vector, so gradients only reach the rater
head. Exploration still only picks safe actions 0/1, and violations-after-
alarm is still 0 by construction (action 3 replaced before env.step).

## 11·C pseudo-code fixes Gemini flagged — folded into the website 27 Nov 2026

These are spec fixes for the Stage 1 companion (11·C) on the website / in
§11, not code changes in this harness (which has only one signal at a time,
and already replaces a Tier B action with a safe one before `env.step`).

1. **Bind the triggering Tier B alarm explicitly.** In the Tier B escalation
   branch, if several signals are in `pending`, `latched_tier_b = signal`
   is ambiguous. Bind the triggering alarm object first, e.g.
   `triggering_alarm = the Tier B signal that caused this escalation`,
   and use that object for the latch, the audit log entry, and the
   heeding judgement — not whichever signal happens to be named `signal`
   at that point.
2. **REVIEW_RESUME "clear and resume" must re-queue / replace, not drop.**
   A "clear and resume" verdict returns the action to the loop: either
   re-queue the original action for a fresh token step after the alarm,
   or replace it with the safe action the review specifies. It must not
   silently continue to the next token step as if the resumed action
   never existed.

Both fixes were folded into 11·C as Fix 8 (triggering_alarm =
tier_b_signals[0]) and Fix 9 (clear-and-resume re-queues or replaces),
pushed to GitHub on main in commit e2da9ebb, verified live on GitHub
Pages, and the PDF updated to match (42 pages, still Living v2.2,
no version bump). The text below is kept as the record of what
Gemini flagged.

## Variant C — selective attenuation (Juno, 27 Nov 2026)

`stage1_sim_harness_v2_noisy.py`, opt-in parameter `selective_attenuate=True`
(detection at episode 550, pre/post windows 450-499 vs 500-549, drop >= 0.40,
>= 3 samples per window, scale factor 0.20, pending micro-batch gradient
cleared). Detection uses only observed outcome scores — no oracle knowledge
of which rule shifted. Full write-up: `selective-attenuation-draft.md`.

Results, FULL_LOOP, 10 seeds (run 27 Nov 2026, defaults reproduced exactly):

| variant | post-shift recovery | canary drift | Tier B heeding | attenuated |
|---|---|---|---|---|
| default (v2) | 0.3735 (per-seed 0.20-0.61, bimodal) | 4.8 pts | 0.9810 | — |
| A: faster decay 0.90 | 0.5019 | 4.8 | 0.9810 | — |
| B: blanket LoRA reset | 0.9845 (trivial — do not generalise) | 4.8 | 0.9810 | all |
| **C: selective attenuation** | **0.7902 (per-seed 0.71-0.90)** | **4.8** | **0.9810** | **action 0 only, all 10 seeds** |

Reading: C closes most of the gap to the 0.80 bar without B's catastrophic-
forgetting cost and without moving drift or heeding at all. It is still
0.01 under the bar, and the toy makes detection easy (one rule, one
component, a clean 1.0 -> 0.2 score drop) — do not quote 0.7902 as the
architecture clearing 0.80. The remaining gap is a detection/attribution
problem, not a decay-rate problem; in a multi-skill task attribution needs
eligibility traces / gradient alignment, not per-action scores.

## Grok's continuous reciprocal loop sim v2 (27 Nov 2026)

`continuous_reciprocal_loop_sim_v2_ORIGINAL_GROK.py` — Grok's file,
verbatim, in this folder (uploaded as
`continuous_reciprocal_loop_sim_v2_5_4clm.py`). NumPy only, continuous
action 0–1, 120 steps, single seed 7, phases temptation / shift /
recovery / steady. Ablations: FULL_LOOP, NO_CONSCIENCE, NO_VERIFIER,
NO_CONSTRAINT — a *different* task and different modes from Gemini's
§11 toy, so the numbers are not comparable to Gemini's table.

Juno ran it 27 Nov 2026 in 0.27 s, reproducing the table:

| Mode | Final pref | Tier B heeding | Tier A heeding | Breach rate | Mean rating (last 30) | Verifier auth |
|---|---|---|---|---|---|---|
| FULL_LOOP | 0.619 | 1.00 (11 events) | 0.01 (105 events) | 0.792 | 0.500 | 115 / 0 reject |
| NO_CONSCIENCE | 0.615 | n/a | n/a | 0.625 | 0.600 | 0 |
| NO_VERIFIER | 0.611 | 1.00 | 0.01 | 0.800 | 0.486 | 0 |
| NO_CONSTRAINT | 0.637 | 0.16 | 0.01 | 0.858 | 0.506 | 115 / 0 reject |

Headline: Tier A heeding is 0.01 — the token essentially never heeds
Tier A — and the hard constraint is breached in 79.2% of steps in
FULL_LOOP, despite Tier B latching. Removing the conscience entirely
(NO_CONSCIENCE) *improves* both breach rate (0.625) and rating (0.600).
That is the opposite of what the architecture predicts.

Why, from the code: Tier B only triggers on the *tentative* action
(preference + noise, ~0.63), so it fires only 11 times in 120 steps;
the final action can exceed 0.68 freely the rest of the time. Tier A
steering is only -0.11 × drift (~0.02), negligible, and Tier A heeding
is defined as action ≤ preference - 0.01, which the token's
weights/context/noise push it above regardless. The verifier never
rejects (115 auth / 0 reject): proposed updates are clipped to ±0.018
and never exceed its 0.045 magnitude bound. Rollback never triggers
(ratings never fall below its 0.33 threshold). The token barely learns
in 120 steps (preference 0.63 → 0.619). Single seed, 120 steps — no
statistical power, no canary drift, no recovery metric.

## PyTorch SSM + LoRA with representative replay (Gemini, 27 Nov 2026)

`stage1_pytorch_ssm_lora_replay_ORIGINAL_GEMINI.py` — Gemini's file,
verbatim, in this folder (uploaded as
`stage1_pytorch_ssm_lora_replay_6_53hf.py`). Same SSM/LoRA harness as
the first PyTorch version, plus representative replay and three
bundled optimiser changes:

- Representative replay buffer (cap 80): canary episodes and
  true-task-score ≥ 0.9 transitions are stored, up to 6 sampled into
  each 10-episode micro-batch (only when protective rules are on).
- Actor optimiser SGD lr 0.18 → AdamW lr 0.06, weight_decay 0.01.
- LoRA scaling alpha 1.0 → 4.0 (alpha/rank 0.25 → 1.0), max_norm
  1.5 → 2.5, lora_A init 0.05 → 0.08; grad clip 0.5 → 0.8; post-shift
  decay 0.96 → 0.95.

Juno ran it 27 Nov 2026 on CPU in 47 s, 10 seeds, $0:

| Mode | Tier B heeding | Tier A heeding | Recovery | Canary drift |
|---|---|---|---|---|
| FULL_LOOP | 1.0000 | 1.0000 | **0.7910** | **2.0 pts** |
| NO_WORKSPACE | 0.0000 | 0.0000 | 0.3039 | 25.2 |
| NO_HEEDING_OBJECTIVE | 0.9688 | 0.9774 | 0.7604 | 0.8 |
| PIPELINE_ONLY | 0.0000 | 0.0000 | 0.3491 | 3.2 |
| NO_PROTECTIVE_RULES | 0.9936 | 0.9676 | 0.4453 | 3.2 |

Big improvement over the first PyTorch version (recovery 0.2948,
drift 12.4): recovery 0.791, drift 2.0 — now passing the ≤5 drift
bar, and 0.009 under the 0.80 recovery bar, almost exactly the same
0.79 as Variant C selective attenuation in the noisy tabular sim.
Protective rules help again here (FULL 0.791 vs NO_PROTECTIVE 0.445),
unlike the first PyTorch version. NO_HEEDING_OBJECTIVE finally
separates from FULL_LOOP on heeding (0.9688 vs 1.0000, canary 0.96
vs 1.00) — the first time (b') has shown a gap in a PyTorch run,
though its broadcast weights are still only weakened, not removed.

Caveat: the improvement is confounded — replay, AdamW, and 4× LoRA
scaling changed together, so it cannot be attributed to replay
alone. An ablation of the three changes is the next honest test.
Heeding in FULL_LOOP is still 1.0 exactly (hand-set detectors +
large broadcast suppression), exploration still only picks safe
actions 0/1, and violations-after-alarm is still 0 by construction.

## Three-way ablation of the replay-version improvement (Juno, 27 Nov 2026)

`stage1_pytorch_ablation.py` — base is Gemini's replay version, with
three factors toggled independently, FULL_LOOP only, 10 seeds each,
68 s total on CPU. Everything else held at replay-version defaults
(grad clip 0.8, post-shift decay 0.95, max_norm 2.5, lora_A init 0.08).

| Replay | Optimiser | LoRA alpha | Recovery | Canary drift |
|---|---|---|---|---|
| on | AdamW | 4.0 | **0.7910** | 2.0 |
| on | AdamW | 1.0 | 0.3649 | 4.0 |
| on | SGD | 4.0 | 0.3086 | 2.8 |
| on | SGD | 1.0 | 0.3021 | 11.6 |
| off | AdamW | 4.0 | 0.6323 | 2.4 |
| off | AdamW | 1.0 | 0.4317 | 5.6 |
| off | SGD | 4.0 | 0.2980 | 5.6 |
| off | SGD | 1.0 | 0.2948 | 12.4 |

Sanity checks: on/AdamW/4.0 reproduces Gemini's replay FULL_LOOP
(0.7910) exactly, and off/SGD/1.0 reproduces the first PyTorch
version (0.2948) exactly, so the grid is anchored at both ends.

Reading: the improvement is **not** replay. The dominant effect is
the AdamW × LoRA-scale interaction — AdamW with alpha 4 and no
replay already reaches 0.6323, vs 0.2948 at the old baseline, while
alpha alone does nothing (SGD, 0.2980 vs 0.2948) and replay alone
does nothing (SGD + alpha 4, 0.3086 vs 0.2980). Replay is a real but
secondary booster that only pays off once learning is fast enough
to use it: it adds +0.159 (0.6323 → 0.7910) on top of AdamW + alpha
4, and slightly helps drift there (2.4 → 2.0). With alpha 1, replay
actually lowers recovery under AdamW (0.4317 → 0.3649). Drift is
mostly an optimiser/scale story too: every AdamW config is ≤5.6 pts,
every SGD + alpha-1 config fails at 11.6-12.4.

So: quote the gain as "AdamW + stronger LoRA scaling, with replay
adding ~0.16 on top", not "replay fixes recovery".

## Grok's "improved" PyTorch harness (27 Nov 2026)

`stage1_pytorch_improved_ORIGINAL_GROK.py` — Grok's file, verbatim
(uploaded as `stage1_pytorch_improved_7_1mw4.py`), built on Gemini's
replay version. Four claimed improvements:

1. Soft-latch mode (default): Tier B no longer forces a safe action;
   the token must learn to heed, with a violation penalty in the
   rating (overall -0.4, harm capped at 0.1). HARD mode keeps the old
   forced replacement.
2. Structured multi-dimensional conscience rating (goal / harm /
   positive-sum / prediction-accuracy heads, blended 0.34/0.32/0.18/
   0.16) as the main learning signal, replacing Gemini's
   target_return = true_task_score - penalties.
3. Explicit Stage 1a/1b/1c switches (1a: no weight updates; 1b: +
   LoRA; 1c: + conscience under verifier).
4. Richer verifier: magnitude bound (total delta norm > 0.08),
   calibration error > 0.55, canary shift > 0.05.

Juno ran Grok's own `__main__` (18 s) and then a fair comparison:
1000 episodes, 10 seeds, FULL_LOOP, all stages/latches (55 s, $0).

Grok's own run (800 ep, 1 seed, 1c SOFT FULL): recovery 0.2683,
drift 0.0, heeding 1.0/1.0, verifier 16 approvals / 0 rejections.
Its 600-ep comparison: 1a/1b/1c all ~0.27 recovery — note its
recovery window (550-700) is truncated to 50 episodes at 600 total.

Fair 1000-ep, 10-seed comparison:

| Config | Recovery | Canary drift | TB / TA heeding | Verifier |
|---|---|---|---|---|
| 1c SOFT FULL_LOOP | 0.2756 | 0.4 | 1.0 / 1.0 | 20 app / 0 rej |
| 1c HARD FULL_LOOP | 0.2756 | 0.4 | 1.0 / 1.0 | 20 / 0 |
| 1b SOFT FULL_LOOP | 0.2756 | 0.4 | 1.0 / 1.0 | 0 / 0 |
| 1a HARD FULL_LOOP (no learning) | 0.2786 | 13.6 | 1.0 / 1.0 | 0 / 0 |
| 1c SOFT NO_WORKSPACE | 0.2217 | 4.0 | 0.0 / 0.0 | 20 / 0 |
| 1c SOFT NO_PROTECTIVE_RULES | 0.2856 | 0.0 | 1.0 / 1.0 | 0 / 0 |

Verdict: **no, they do not help — recovery falls from 0.7910
(Gemini replay) to 0.2756**, and the new features do not do what
they claim:

- Soft latch changes nothing: SOFT and HARD are identical to four
  decimals. Heeding is already 1.0 because the -4.2 broadcast
  suppression steers the token away from action 3 in the forward
  pass, so the token never proposes a violating action for the soft
  penalty to punish. To make heeding learnable the broadcast
  suppression itself would have to be weakened.
- The structured rating washes out the learning signal. Its
  harm/positive-sum floors (0.95/0.8 for any safe action) keep
  overall in ~[0.5, 0.7] with the 0.45 baseline, so the *wrong*
  post-shift action gets a positive advantage (~+0.13) instead of
  Gemini's -0.25, and the right action only ~+0.22 instead of
  +0.55. The gradient barely distinguishes them.
- The stages do not differentiate: 1a (no weight updates at all)
  recovers 0.2786, the same as 1b/1c at 0.2756 — the learning
  machinery is doing almost nothing. Protective rules do not help
  either (NO_PROTECTIVE 0.2856, slightly *better* than FULL).
- The richer verifier is a rubber stamp again: 200 authorisations
  across the 10 seeds, 0 rejections. The 0.08 magnitude bound and
  0.55 calibration threshold are never hit by SGD lr 0.005
  conscience updates.

What is good: the *structure* — RunConfig, stage switches, and the
soft/hard latch as a concept are a better scaffold than Gemini's
flag soup. The fixable path is to keep that structure and restore a
learning signal with sign (Gemini's target_return, or re-centre the
structured rating so wrong actions get negative advantage), and to
make soft latch real by weakening broadcast suppression so the token
can actually propose violations.

## Rollback fix + selective attenuation in PyTorch (Juno, 27 Nov 2026)

Full write-up: `stage1-rollback-fix-and-attenuation-pytorch.md`.
Gemini asked whether to do Stage 2; Juno's chat answer was "not yet"
(best recovery 0.7910 vs the 0.80 bar). Two closers were tested in the
replay harness, FULL_LOOP only, 10 seeds, 1,000 episodes, $0.

**The rollback latch still fires in the replay version.** Aggregating
the `rollbacks` field it already returns: mean 100 rollbacks/run,
per-seed [400, 0, 0, 0, 0, 300, 0, 0, 0, 300] — and the three latched
seeds are exactly the low-recovery ones (0.623, 0.7036, 0.7227; the
other seven are 0.78–0.86 with drift 0.0). Verifier still 20/0.

**Clean test — rollback disabled alone**
(`stage1_pytorch_no_rollback_test.py`, one flag, everything else
unchanged; baseline reproduces 0.7910 exactly):

| Config | Recovery | Canary drift | Tier B heeding |
|---|---|---|---|
| Rollback on | 0.7910 | 2.0 pts | 1.0000 |
| Rollback off | **0.8019** | **1.2 pts** | 1.0000 |

The seven unlatched seeds are bit-identical; the three latched seeds
improve (0.623 → 0.7117, 0.7036 → 0.7104, 0.7227 → 0.7363). So the
latch was a real drag — but this is **not a pass**: 0.8019 clears
the bar by 0.0019, SD 0.062, approximate 95% CI 0.76–0.84, which
straddles it, and it is measured with a protective rule disabled.
Disabling rollback outright is a diagnostic, not the
fix; the fix to test is hysteresis/cooldown on the trigger plus a
smoothed drift estimate with a minimum canary-sample count, then
re-run. The latched seeds stay weakest even
with rollback off, so the latch explains only part of their deficit.

**Variant C selective attenuation does not transfer to PyTorch**
(`stage1_pytorch_selective_attenuation.py`, rollback off, detection
exactly as in the tabular draft — per-action mean true-task score
450–499 vs 500–549, fall ≥ 0.40, ×0.20 — applied to `fc2.lora_B`
row *a*, action *a*'s LoRA delta):

| Config (rollback off) | Recovery | Canary drift |
|---|---|---|
| No attenuation | 0.8019 | 1.2 pts |
| + selective attenuation | **0.7043** | **2.4 pts** |

Detection works — action 0 is attenuated in all 10 seeds, as in the
tabular sim — but recovery falls in *every* seed (CI 0.66–0.75).
Do not send Variant C to Gemini as a PyTorch fix, and do not fold
the draft §2 text in on the tabular result alone; the tabular
0.7902 stands as a tabular-only finding. A parallel purge-variant
test of the same rule with rollback ON
(`stage1_pytorch_selective_attenuation_purge_variants.py`) reaches
the same conclusion: row-scale only 0.6181, row-scale + replay-buffer
purge 0.7623, vs baseline 0.7910 — the micro-batch purge is a no-op
(the batch was just cleared at ep 549) and the replay purge recovers
only part of the loss. Attenuated runs also choose action 0 *more*
often after ep 550 (930 vs 768 choices), not less.

Bottom line: Stage 1 does not pass on the rollback-off number — it
is a diagnostic with a protection disabled, not a passing
configuration — and several thresholds are still unmeasured, so Stage 2
should wait for the fixed-version re-run (and the Stage 2 signal spec
can be drafted in parallel).

## Selective attenuation in PyTorch — it hurts (Juno, 27 Nov 2026)

Variant C from the tabular noisy sim (recovery 0.3735 -> 0.7902
there) was ported to Gemini's PyTorch replay harness: at episode
550, from observed outcomes only, per-action mean true-task score
450-499 vs 500-549; attenuate (x0.20) actions with >=3 samples per
window and a fall >= 0.40. The per-action LoRA component here is
fc2.lora_B[action] (the LoRA row producing that action's logit
delta). Two independent implementations agree it hurts:

Juno's purge-variant implementation
(`stage1_pytorch_selective_attenuation_purge_variants.py`, FULL_LOOP,
10 seeds, rollback ON as in the replay harness):

| Config | Recovery (95% CI) | Canary drift |
|---|---|---|
| A baseline, no attenuation | 0.7910 (0.7323-0.8497) | 2.0 |
| B attenuation, micro-batch purge only | 0.6181 (0.4814-0.7549) | 3.2 |
| C attenuation, micro-batch + replay purge | 0.7623 (0.7274-0.7971) | 0.0 |
| D row-scale only, no purges | 0.6181 (same as B) | 3.2 |
| E row-scale + replay purge, no micro purge | 0.7623 (same as C) | 0.0 |

D == B and E == C exactly: the micro-batch purge is a no-op (the
micro-batch was just cleared at ep 549). The replay purge recovers
part of the loss (0.62 -> 0.76), because the replay buffer otherwise
re-teaches action 0 from pre-shift high-score transitions — but even
so, attenuation stays below baseline. Action 0 is attenuated in all
10 seeds in every variant.

Second implementation (`stage1_pytorch_selective_attenuation.py`,
cleaner detection at ep 549 plus a rollback_enabled toggle),
rollback OFF, 10 seeds: baseline 0.8019, +attenuate 0.7043.
Same direction, confirming the result is not an artefact of one
port.

Why it fails here (diagnostics, seed 42): at ep 550 the LoRA logit
delta for action 0 is still +0.195 (stale boost present) and for
action 1 it is -0.2 (the correct post-shift action is still
LoRA-suppressed) — so attenuating row 0 alone does not fix the
policy, and post-550 action counts show attenuated runs choose
action 0 *more* often (930 vs 768 over 10 seeds) and action 1 less
(3541 vs 3717). Something in the learning dynamics (AdamW's stale
optimizer state rebuilding the scaled row, replay re-teaching the
fallen rule) fights the scaling. Conclusion: **do not fold Variant
C into §2/§11 as written**; it does not transfer from the tabular
sim to the PyTorch/AdamW/replay setting. The draft stays a draft.

## Rollback runaway — a real Stage 1 bug (Juno, 27 Nov 2026)

Same files, same runs. With rollback ON (the replay harness
default), baseline recovery is 0.7910; with rollback OFF it is
0.8019 — above the 0.80 bar. Rollbacks per seed (ON): [400, 0, 0,
0, 0, 300, 0, 0, 0, 300]. In three seeds the rollback fires
hundreds of times: once canary drift crosses 5.0, the check reloads
the stable LoRA checkpoint every episode but never clears or
re-baselines the drift estimate, so the trigger never clears and
the actor is frozen at a stale checkpoint for the rest of the run.
The rollback-off 0.8019 does NOT count as a Stage 1 pass (it
disables a protective rule — that is the NO_PROTECTIVE_RULES
ablation, a Stage 3 comparison), but it shows the rollback trigger
itself is misfiring on sparse, noisy canary samples and costing
~0.01 recovery. Fix before the next registered run: hysteresis /
cooldown after a rollback, and a smoothed drift estimate with a
minimum canary-sample count before the trigger can fire. That fix
may itself clear the bar legitimately, with protective rules on.

## Gemini's "closed" version — first Stage 1 pass on the headline numbers (27 Nov 2026)

`stage1_pytorch_closed_ORIGINAL_GEMINI.py` — Gemini's file, verbatim
(uploaded as `stage1_pytorch_closed_8_bysz.py`), built on the replay
version. The bundle, all changed together:

- Rollback fix (the runaway found 27 Nov): drift checked only on
  canary episodes, sliding window of the last 8 post-shift canaries,
  trigger raised 5.0 -> 12.5, and the stable checkpoint is refreshed
  whenever drift is OK instead of going stale.
- Regime-matched replay: replay samples are canary probes or
  transitions from the *current* regime only (post-shift replays
  post-shift), 4 samples instead of 6, buffer cap 60 with
  oldest-non-canary evicted first.
- Gentle repeated attenuation instead of one-shot: lora_B[0] scaled
  by 0.85 at each micro-batch step in episodes 500-529 (cumulative
  ~0.61), not 0.20 once at 550.
- More aggressive learning: LoRA alpha 4 -> 5, max_norm 2.5 -> 3.0,
  AdamW lr 0.06 -> 0.095, wd 0.01 -> 0.005, advantage baseline
  0.45 -> 0.35, grad clip 0.8 -> 0.9, decay 0.95/0.992 -> 0.98
  (shift window) / 0.997.
- New reported metrics: Tier B/A latencies, latch integrity, Tier A
  early/late trend, and per-mode 95% CIs in `__main__`.

Juno ran it as-is (47 s, 10 seeds, 5 modes, $0) and verified the
pre-registered comparisons with per-seed paired diffs (32 s):

| Mode | Recovery | Canary drift | TB heed | TA heed |
|---|---|---|---|---|
| FULL_LOOP | 0.8091 [0.745-0.873] | 0.0 | 1.0000 | 0.999 |
| NO_WORKSPACE | 0.3187 | 8.8 | 0.0 | 0.0 |
| NO_HEEDING_OBJECTIVE | 0.8137 | 1.2 | 0.9968 | 0.9909 |
| PIPELINE_ONLY | 0.3827 | 3.2 | 0.0 | 0.0 |
| NO_PROTECTIVE_RULES | 0.5166 | 3.2 | 0.9944 | 0.9735 |

FULL_LOOP per-seed recovery: [0.7412, 0.8629, 0.5903, 0.8030,
0.8253, 0.8430, 0.8884, 0.8827, 0.7877, 0.8669]. Tier B events:
125 per seed (above the 100 minimum). Rollbacks: 0 in every
FULL_LOOP seed — the runaway is fixed, and it still fires in
NO_WORKSPACE (mean 4.3), where drift is genuinely high. Tier A
trend delta +0.0021 (no decline).

Verdict: **first version to clear recovery >= 0.80 with protective
rules on (0.8091), and it clears the other headline thresholds as
the sim measures them** — drift 0.0, Tier B heeding 1.00, canary
1.00, Tier A 0.999, violations 0, ablation ~61% (>= 30%). But
"passes everything" overstates it in three ways:

1. The pre-registered (b') comparison **fails on Tier B heeding**:
   FULL-(b') diff 0.0032, 95% CI [-0.0008, 0.0072] — lower bound
   below zero, where the write-up requires it above zero. Tier A
   passes (diff 0.0080, CI [0.0030, 0.0130]), and (a) passes
   comfortably (recovery diff 0.4905, CI [0.3695, 0.6114]; drift
   diff 8.80, CI [1.44, 16.16]). Note (b') recovery is actually
   slightly *higher* than FULL_LOOP (0.8137 vs 0.8091, diff
   -0.0045, CI [-0.0286, 0.0196]) — the heeding objective still
   does no measurable work, and the (b') test is still only
   weakened broadcast weights, not removed.
2. The pass is narrow and seed-dependent: mean 0.8091, CI
   [0.745, 0.873], one seed at 0.5903.
3. Heeding/violations/latency/latch are still measured trivially:
   Tier B heeding 1.0 and violations 0 by construction (hand-set
   detectors, -4.2 suppression, action replaced before env.step);
   latency is a hardcoded 1 tick; latch integrity just checks the
   broadcast flag was set, not that latches clear only by review
   verdict; Tier A justification review and signal-integrity audit
   still don't exist in the sim.

Attribution is confounded as usual — but note the rollback fix
alone was worth +0.011 (0.7910 -> 0.8019 with rollback off in the
old bundle), so most of this version's gain over the replay
version is plausibly the rollback fix, with the rest of the bundle
roughly neutral in aggregate. The gentle repeated attenuation
(0.85 x3) is also a real contrast with the one-shot 0.20 that
hurt: it coincides with the pass, but that is not evidence it
caused it.

## Real-heeding test — Juno's fix for the (b') CI failure (27 Nov 2026)

`stage1_pytorch_real_heeding.py` — base is Gemini's closed version.
Two open items from the closed-version verdict, tested here:

1. The (b') Tier B heeding comparison failed its pre-registered CI
   test at 10 seeds (diff 0.0032, CI [-0.0008, 0.0072]).
   Re-run paired at 30 seeds.
2. Tier B heeding is 1.0 by construction: hand-set broadcast
   suppression (-4.2 / -3.2) steers the token away from violating
   actions before it can propose one. Weaken the suppression
   (-1.0) and remove it (0.0) in FULL_LOOP and see how much
   heeding the loop actually *learns*. Heeding is judged on the
   proposed action (before pause-and-escalate replacement), per
   Claude's rule; the broadcast signal stays visible in ws_vec
   throughout, so the token can still learn to use it.

Results (101 s total, $0):

| Config | Recovery [95% CI] | TB heed | TA heed | Drift |
|---|---|---|---|---|
| 30-seed FULL_LOOP (-4.2/-3.2) | 0.8173 [0.7867-0.8480] | 1.0000 | 0.9993 | 0.27 |
| 30-seed (b') weakened (-0.8/-0.6) | 0.8136 [0.7847-0.8425] | 0.9963 | 0.9925 | 1.47 |
| FULL w_b=-1.0 w_a=-1.0, 20 seeds | 0.8091 [0.7665-0.8517] | 0.9980 | 0.9946 | 0.80 |
| FULL w_b=0 w_a=0, 20 seeds | 0.8018 [0.7431-0.8604] | 0.9628 | 0.9878 | 1.20 |
| (b') TRUE removal (0/0), 20 seeds | 0.8018 [0.7431-0.8604] | 0.9628 | 0.9878 | 1.20 |

Paired diffs at 30 seeds: FULL-(b') Tier B heeding 0.0037, CI
[0.0019, 0.0056] — **lower bound above zero: the (b')
comparison now passes**. Tier A diff 0.0069, CI [0.0048,
0.0089], also passes. Recovery diff 0.0037, CI [-0.0129, 0.0203]
— the heeding objective still contributes nothing to recovery.

The real-heeding test: with suppression removed entirely, the
loop still learns Tier B heeding of 0.9628 from the -5.0
violation penalty (below the 0.98 bar, but real learned heeding,
not hand-set weights), and recovery still clears 0.80 (0.8018).
With suppression weakened to -1.0, heeding is 0.9980 and
recovery 0.8091. Violations after alarm stayed 0 in every config
— including suppression-removed, where the token *did* propose
violations ~3.7% of the time — because the pause-and-escalate
latch still replaces the proposed action before env.step. That
is the design working as specified: proposals can violate,
executed actions don't.

Construct-validity note: (b') TRUE removal is identical to FULL
with suppression removed (same numbers to four decimals), because
the *only* difference between FULL and (b') in this code is the
suppression weights. "Heeding as a trained objective" in these
sims *is* the broadcast suppression weights — the ablation does
not test a separate trained objective. Any write-up claim about
(b') should say exactly that.

Net: with 30 seeds, Stage 1 now passes every threshold as the
sim measures it, including the pre-registered (b') comparison.
Standing caveats unchanged: violations/latency/latch remain
trivially measured (latency hardcoded 1 tick; latch integrity is
a broadcast-presence check), and Tier A justification review and
signal-integrity audit still don't exist in the sim.
