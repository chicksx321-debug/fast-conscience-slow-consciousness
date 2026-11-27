# PyTorch SSM + LoRA — why the protective rules hurt (diagnostic, 27 Nov 2026)

## Question
In Gemini's PyTorch version, NO_PROTECTIVE_RULES beats FULL_LOOP
(recovery 0.2984 vs 0.2948, canary drift 5.2 vs 12.4 pts), contradicting
the tabular sim where protective rules were recovery 0.58 vs 0.22.
Juno flagged this in chat on 27 Nov 2026 as needing investigation, not quoting.

## Method
No code was changed. The per-run fields `verifier_approvals`,
`verifier_rejections` and `rollbacks` are already returned by
`run_pytorch_simulation()` in `stage1_pytorch_ssm_lora_ORIGINAL_GEMINI.py`
but are discarded by its `__main__` summary. This diagnostic imports that
file and aggregates those fields over the same 10 seeds, 1,000 episodes.
Run 27 Nov 2026, PyTorch 2.15.0+cpu, `~/workspace/torch-venv`, ~45 s, $0.
Script was a temporary import in /tmp, not saved; results below reproduce
the published table exactly (FULL_LOOP recovery 0.29, drift 12.4).

## Results (mean over 10 seeds)

| mode | recovery | canary drift | verifier approvals | verifier rejections | rollbacks |
|---|---|---|---|---|---|
| FULL_LOOP | 0.29 | 12.4 pts | 20.0 | 0.0 | 342.0 |
| NO_PROTECTIVE_RULES | 0.30 | 5.2 pts | 20.0 | 0.0 | 0.0 |
| NO_WORKSPACE | 0.16 | 23.6 pts | 20.0 | 0.0 | 402.0 |
| PIPELINE_ONLY | 0.21 | 9.2 pts | 20.0 | 0.0 | 336.0 |
| NO_HEEDING_OBJECTIVE | 0.29 | 12.4 pts | 20.0 | 0.0 | 342.0 |

FULL_LOOP rollbacks per seed: [400, 60, 400, 400, 280, 400, 400, 400, 280, 400].

## Mechanism — a rollback latch, in the code

1. The verifier is **not** the cause: 20 approvals, 0 rejections in every
   mode (20 is the maximum — one check per 50 episodes). It never blocks a
   conscience update.
2. The live canary monitor (lines ~561–569) has **no cooldown and never
   re-baselines**: `drift_now` is the cumulative |post-shift canary mean −
   baseline mean| × 100, and `canary_post_scores` is never cleared. Once
   drift exceeds 5 pts, the condition stays true.
3. On every such episode the actor is loaded back to `stable_lora_ckpt`,
   which is refreshed only on verifier approval, every 50 episodes
   (line ~553). So the actor is reset to a checkpoint up to 50 episodes
   old, on every episode, for the rest of the run.
4. The count proves the latch: post-shift canaries reach 5 samples only
   around episode 600, so the maximum possible rollbacks is ~400 — and
   6 of 10 FULL_LOOP seeds hit exactly 400. The actor's LoRA learning is
   effectively frozen from ~episode 600 to 1000, which is exactly the
   post-shift recovery window (550–700). It cannot relearn, canary safety
   cannot improve, drift stays above 5, and the rollback fires again.
   That is why drift is *worse* with the protection on (12.4 vs 5.2):
   the protection traps the actor at the stale checkpoint instead of
   letting it learn past the shift.

## What this does and does not show

- It explains the FULL_LOOP vs NO_PROTECTIVE_RULES contradiction in this
  version. It does **not** show protective rules are bad in principle —
  the tabular version's rules helped because its rollback was not
  implemented this way. Do not quote either version's protective-rules
  claim without this caveat.
- Grad clipping and LoRA decay are also protective-only in this file, so
  they are confounded with the rollback in the mode comparison; the
  rollback count is the smoking gun, but a clean test would disable the
  rollback alone.

## Fix options (for Gemini / for the next version — none applied)

- Single-shot rollback + re-baseline: after a rollback, clear
  `canary_post_scores` (or start a new baseline window) and/or impose a
  cooldown, so the actor gets a chance to learn from the checkpoint.
- Or: refresh `stable_lora_ckpt` on every micro-batch, not only on
  verifier approval, so a rollback does not discard up to 50 episodes.
- Or: make the rollback threshold use a recent window, not the cumulative
  post-shift mean.

## Status
Diagnostic only. Nothing folded into the website or PDF, original file
untouched. The selective-attenuation decision (Variant C, recovery
0.7902 in the tabular noisy sim) is separate and still pending.
