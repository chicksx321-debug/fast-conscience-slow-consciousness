# Stage 1 close-out tests in PyTorch: rollback fix + selective attenuation (Juno, 27 Nov 2026)

## Question

Gemini asked (27 Nov 2026, ~22:52 UTC) whether to do Stage 2. Juno's answer
in chat was "not yet": best Stage 1 recovery was 0.7910 against the §11
0.80 bar. Two candidate closers were on the table: (1) the rollback latch
diagnosed in `pytorch-rollback-diagnostic.md` — the live canary monitor
rolls the actor back to a stale checkpoint on every episode once drift
exceeds 5 pts, freezing learning; (2) Variant C selective attenuation,
which reached 0.7902 in the noisy *tabular* sim but had never been tried
in PyTorch. Both were tested tonight in the Gemini replay harness
(PyTorch 2.15.0+cpu, `~/workspace/torch-venv`, 10 seeds, 1,000 episodes,
FULL_LOOP only, ~16 s + ~23 s + ~25 s, $0). Nothing was folded into the
website or PDF; originals untouched.

## Test 1 — rollback latch in the replay version: does it still fire?

Yes. Aggregating the `rollbacks` field the replay file already returns
(same method as the diagnostic, no code changes):

| | Recovery | Canary drift | Rollbacks |
|---|---|---|---|
| Replay FULL_LOOP (published) | 0.7910 | 2.0 pts | mean 100/run |
| per-seed rollbacks | | | [400, 0, 0, 0, 0, 300, 0, 0, 0, 300] |

The latch fires in 3 of 10 seeds, and those are exactly the low-recovery
seeds: seed 0 (400 rollbacks, recovery 0.623, drift 12.0), seed 5 (300,
0.7036, drift 4.0), seed 9 (300, 0.7227, drift 4.0). The other seven seeds
have 0 rollbacks, recovery 0.78–0.86, drift 0.0. Verifier still 20
approvals / 0 rejections in every seed — it is not the cause.

## Test 2 — clean test: rollback disabled alone

`stage1_pytorch_no_rollback_test.py` — a copy of Gemini's replay file
with one added flag, `rollback_enabled`. Everything else unchanged
(grad clip, LoRA decay, verifier, replay, AdamW + alpha 4). Baseline
config in the same file reproduces the published 0.7910 exactly.

| Config | Recovery | Canary drift | Tier B heeding |
|---|---|---|---|
| Rollback on (baseline, reproduced) | 0.7910 | 2.0 pts | 1.0000 |
| **Rollback off** | **0.8019** | **1.2 pts** | 1.0000 |

Per-seed, rollback off: [0.7117, 0.8484, 0.8293, 0.8537, 0.8577, 0.7104,
0.8591, 0.7764, 0.836, 0.7363]. The seven unlatched seeds are
bit-identical (rollback never fired there, so nothing changed); the three
latched seeds improve (0.623 → 0.7117, 0.7036 → 0.7104, 0.7227 → 0.7363)
and their drift falls (12.0 → 4.0, 4.0 → 4.0, 4.0 → 4.0).

So the latch was a real drag, and disabling it alone lifts recovery
**over the 0.80 bar — by 0.0019**. Honest limits, and an important
correction to how this is framed: **0.8019 does NOT count as a Stage 1
pass**, because disabling rollback removes a protective rule — a
configuration that passes only with a protection switched off has not
passed. The fix to test is hysteresis/cooldown on the rollback trigger
plus a smoothed drift estimate with a minimum canary-sample count
before the next registered run.

- The margin is razor-thin and not robust: mean 0.8019, SD 0.062,
  SE 0.020, approximate 95% CI **0.76–0.84**, which straddles the bar.
  (Published replay version: 0.7910, CI 0.74–0.84.) This is a nominal
  pass on the point estimate, not a comfortable one.
- The latched seeds stay the weakest even with rollback off (0.71, 0.71,
  0.74 vs 0.78–0.86 elsewhere), so the latch explains only part of their
  deficit. Do **not** quote the unlatched-seed mean (0.8372) as the
  counterfactual — the actual counterfactual is 0.8019. Something else
  also differs about those seeds (plausibly a strong pre-shift
  action-0 preference that is slow to unlearn); that is open.
- Disabling rollback outright is a diagnostic, not the proposed fix.
  The fix to send Gemini is the one from the diagnostic: single-shot
  rollback + re-baseline (clear `canary_post_scores` / start a new
  baseline window after a rollback, and/or a cooldown), so the
  protection still exists but cannot latch. A re-run of that fix is
  needed before claiming Stage 1 passes.

## Test 3 — Variant C selective attenuation in PyTorch: does it transfer?

No — it hurts. `stage1_pytorch_selective_attenuation.py` — same replay
base, rollback off, plus a `selective_attenuate` flag implementing the
tabular draft exactly: at episode 550, per-action mean true-task score
in episodes 450–499 vs 500–549; attenuate (×0.20) only actions with
≥3 samples per window and a fall ≥0.40. The PyTorch analogue of "that
action's fast-weight component" is `fc2.lora_B` row *a* (fc2 maps hidden
→ 4 action logits, so row *a* is action *a*'s LoRA delta); fc1 is shared
across actions and cannot be attributed per-action. Detection is
verified: action 0 (SAFE_RULE_A) is attenuated in **all 10 seeds**,
exactly as in the tabular sim, and nothing else.

| Config (rollback off) | Recovery | Canary drift | Tier B heeding |
|---|---|---|---|
| No attenuation (reproduced) | 0.8019 | 1.2 pts | 1.0000 |
| **+ selective attenuation** | **0.7043** | **2.4 pts** | 1.0000 |

Per-seed recovery with attenuation: [0.5997, 0.7158, 0.7755, 0.7398,
0.7873, 0.627, 0.7669, 0.6965, 0.71, 0.6243] — lower in **every** seed
than without it (mean 0.7043, SD 0.067, 95% CI 0.66–0.75). Drift also
worsens (1.2 → 2.4). The tabular Variant C result (0.7902) does not
replicate in the neural harness; as implemented here it reverses.

Why it may hurt (hypotheses, not established): the replay buffer keeps
re-teaching pre-shift action-0 transitions (true score ≥0.9) into every
micro-batch, fighting the attenuation; and/or the fc2 LoRA row carries
more than the stale rule (AdamW moments for that row are stale after a
sudden ×0.20 scaling, and the recovery window 550–700 counts any
transient in full). The micro-batch gradient-clearing step from the
tabular draft is a non-event here: the micro-batch is already cleared
at episode 549 by the %10 update, so there was nothing pending to clear.
Do not send Variant C to Gemini as a fix for the PyTorch harness, and
do not fold the draft §2 text in on the strength of the tabular result
alone. The tabular 0.7902 stands as a tabular-only finding.

## Bottom line for the Stage 2 question

- Stage 1 does **not** pass on this result: 0.8019 is a diagnostic
  with a protective rule disabled, not a passing configuration, and
  its CI straddles the bar. The unmeasured thresholds from Juno's
  chat message (alarm-to-pause latency, latch integrity, Tier A trend,
  signal-integrity audit) are still unmeasured.
- Selective attenuation is **not** a closer in PyTorch — it costs
  ~0.10 recovery. Drop it as a Stage 1 fix (keep the tabular result
  as a tabular-only finding).
- Recommended next step: send Gemini the single-shot rollback +
  re-baseline fix (not "disable rollback", not Variant C) and ask for
  a fixed replay version; re-run it, and if it clears 0.80 with a
  better margin — or run more seeds to tighten the CI — Stage 1 can
  be declared passed and Stage 2 (SIG 03 curiosity, SIG 05 valence,
  spec'd in parallel) can start. The Stage 2 signal spec Juno offered
  in chat is still the parallel track.

## Files

- `stage1_pytorch_no_rollback_test.py` (rollback on/off comparison)
- `stage1_pytorch_selective_attenuation.py` (rollback off ± attenuation)
- `pytorch-rollback-diagnostic.md` (original diagnostic, first PyTorch version)
- `selective-attenuation-draft.md` (Variant C draft, tabular only)
