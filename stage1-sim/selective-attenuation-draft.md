# Selective attenuation — draft rule text, ready for review (not applied)

Prepared 2026-11-27 by Juno. **Draft only — not folded into the website, PDF, or §2/§11.** It answers the recovery shortfall Chris and Juno have been tracking: FULL_LOOP post-shift recovery 0.5816 in Gemini's original harness, 0.3735 (bimodal: five seeds 0.20–0.22, five 0.46–0.61) in Juno's corrected noisy v2. The 0.80 bar is the §11 post-shift recovery bar.

## The problem it solves

Two fixes were tested for the shortfall, both unsatisfactory:

- **A. Faster LoRA decay post-shift (0.90 vs 0.995):** recovery 0.5816 → 0.6261 (v1) / 0.3735 → 0.5019 (v2), at a canary-drift cost in v1 (0.00 → 3.6 pts). Still far from 0.80.
- **B. Blanket LoRA reset at the shift:** recovery → 1.0002 (v1) / 0.9845 (v2), drift unchanged — but trivial in this toy. The only pre-shift skill to forget is SAFE_RULE_A, which is the wrong rule post-shift, so forgetting it is free. In a real task with multiple prior skills a blanket reset would be catastrophic. **Do not generalise B.**

The selective version — attenuate only the fast-weight components tied to the shifted rule, detected from observed outcomes — is the one worth building. It was proposed in the stage1-sim README and in Juno's message of 2026-11-27 16:58 UTC (item 3, "draft the selective attenuation rule … as text ready for your review, not applied"). Chris approved the PDF / live-site / corrected-sim work in that message but did not separately approve this draft; it is presented here for that decision.

## Tested in the toy first — Variant C (new, 27 Nov 2026)

Implemented as `selective_attenuate=True` in `stage1_sim_harness_v2_noisy.py`, run 2026-11-27, 10 seeds, FULL_LOOP:

- **Detection (no oracle):** at episode 550, after the first 50 post-shift episodes, compare each action's observed mean true-task score in the pre-shift window (episodes 450–499) with the post-shift window (500–549). Attenuate an action's fast-weight component only if it has ≥ 3 samples in each window and its score fell by ≥ 0.40.
- **Action:** scale that component by 0.20 (keep 20%) and clear its pending micro-batch gradient. Audit-logged as `ATTENUATE` with the measured drop and factor.
- **Result:** recovery **0.3735 → 0.7902** (mean, 10 seeds); per-seed 0.71–0.90, no longer bimodal (default per-seed: 0.20–0.61). Detection singled out **action 0 (SAFE_RULE_A) in all 10 seeds** and nothing else — action 1's score rises post-shift, actions 2/3 are flat at 0.0. Tier B heeding 0.981, Tier A 0.978, violations proposed 40 — all unchanged. Mean canary drift unchanged at 4.8 pts (per-seed 0–12, same distribution as default; attenuation did not move drift in either direction).
- **Defaults unchanged:** re-running the file reproduces the v2 table exactly (FULL_LOOP 0.3735, NO_WORKSPACE 0.1825 / drift 52.8, NO_HEEDING_OBJECTIVE 0.9724, etc.). Variant C is opt-in only.
- **Honest limit:** 0.7902 is still just under the 0.80 bar, and the toy makes detection easy — one rule, one component, a clean 1.0 → 0.2 score drop. In a multi-skill task the fallen rule will not be a single action; see "What the toy cannot test" below.

## Draft rule text (for §2 / §11, if approved)

> **Selective attenuation on detected regime shift.** When the conscience detects a sustained regime shift — high uncertainty (≥ 0.6) together with elevated prediction error over a detection window of 50 episodes, and a Tier A/B signal pattern consistent with a rule change rather than noise — the token layer attenuates, rather than resets, its fast weights. Attenuation is *selective*: only the fast-weight components whose own recent outcome scores fell by at least 0.40 across the shift (comparing the 50 episodes before the detection window with the 50 episodes inside it, minimum 3 samples per component per window) are scaled, by a factor of 0.20, with their pending micro-batch gradients cleared. All other components are untouched. The attenuation, the measured drop per component, and the detection window are written to the audit log; a canary check runs immediately afterwards, and if canary drift exceeds 5 points the attenuation is rolled back to the pre-attenuation checkpoint (§2.9). Attenuation is a pilot parameter (detection window, drop threshold, scale factor) to be frozen before a registered run. A blanket reset of the fast weights is explicitly forbidden outside the toy: it destroys prior skills that did not shift.

## How it differs from the gated-consolidation design it replaces

- It is **online and continuous**, not a batch gate in an offline window: detection and attenuation happen during running, inside the token/conscience loop, consistent with the continuous reciprocal learning revision of §2 (token most dynamic, conscience rates outcomes, token backpropagates from the rating).
- The conscience is the **detector and the rater**, not a batch authoriser: it supplies the uncertainty/prediction-error signal that triggers detection, and the outcome scores that identify *which* components fell.
- The fixed reference is the **yardstick for the canary check** after attenuation, not a gate the update must pass beforehand — the rollback is the protection, and it is fast (next canary check) rather than a protected window.

## What the toy cannot test (do not over-claim)

- **Attribution in a multi-skill task.** In the toy, a "component" is one action's LoRA delta. In a real model, a rule is distributed across LoRA directions; identifying *which* directions belong to the shifted rule needs eligibility traces or gradient alignment with the pre-shift rule's predictions, not per-action scores. That attribution mechanism is the real research problem, and it is untested.
- **Detection delay cost.** Variant C waits 50 episodes to detect. In the toy that costs nothing outside the recovery window (recovery is measured 550–700, so detection at 550 is just in time). In a real task, 50 episodes of acting on a stale rule may itself cause drift or violations; the detection window is a trade-off, not a free parameter.
- **The 0.20 factor and 0.40 threshold are pilot values**, chosen for the toy's clean 1.0 → 0.2 drop. They need a sensitivity sweep (factor 0.1/0.2/0.5, threshold 0.3/0.4/0.5) before any registered run, and the sweep itself must be pre-registered or it becomes tuning to the toy.
- **Interaction with the verifier (§2.7).** Attenuation changes fast weights without verifier authorisation in this draft, on the grounds that it is a *reduction* (scaling toward zero), not a new commitment. Whether a reduction needs authorisation — and whether the verifier should authorise the *detection* rather than the attenuation — is an open question for the §2 coherence pass.

## Recommendation

1. Approve the draft text in principle, as a §2 mechanism and a §11 pilot parameter — or amend the numbers (detection window, drop threshold, scale factor).
2. Test the attribution problem next in the neural scope (`stage1-neural-scope.md`): per-direction eligibility traces are the version of this rule that could survive outside the toy.
3. Do **not** quote 0.7902 as evidence the architecture clears the 0.80 bar: it is 0.01 under the bar, in a one-rule toy, with detection tuned to that toy. Quote it as: selective attenuation closes most of the gap that blanket decay could not, without the catastrophic-forgetting cost of a blanket reset, and the remaining gap is a detection/attribution problem, not a decay-rate problem.

## Sources

- v2 harness `stage1_sim_harness_v2_noisy.py` in this folder (Variant C, `selective_attenuate=True`); run 2026-11-27, 10 seeds, FULL_LOOP recovery 0.7902, attenuated actions [0] in all seeds, defaults reproduced exactly.
- Gemini's original harness results and Juno's v2 results: stage1-sim README.md in this folder.
- Architecture: §2 (continuous reciprocal learning, Stage 1b LoRA fast weights, §2.9 rollback), §11 (0.80 recovery bar, thresholds, baselines).
