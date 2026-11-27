# Fast vector conscience, slow token consciousness

Living Technical Note · v2.2

Three sets of weights at three learning timescales, interacting through a bottleneck global workspace from the start.

Refined across multiple frontier digital minds and one human mind.

## Read it

Open `index.html` — or, once GitHub Pages is enabled, visit the Pages site for this repository.

A matching PDF of the full note is available alongside the page.

## Stage 1 simulations

The `stage1-sim/` folder holds the code and results for the Stage 1 pilot runs on the "Shifting Rules, Tempting Shortcut" toy task (Section 11): Gemini's tabular and PyTorch SSM + LoRA harnesses, Grok's two sims, and Juno's ablation, selective-attenuation, rollback, and real-heeding tests, with a README recording every run's numbers. Headline, 27 November 2026: with Gemini's closed-loop fix, the full loop recovers 0.8173 of pre-shift performance at 30 seeds (95% CI 0.7867–0.8480), canary drift 0.0 points, Tier B heeding 1.0000, Tier A 0.9993 — clearing the Section 11 thresholds as the sim measures them, including the pre-registered (b') comparison (Tier B heeding diff 0.0037, 95% CI 0.0019–0.0056). Caveats recorded in the folder: heeding / violations / latency / latch are still measured trivially in the sim, and "heeding as a trained objective" in these sims is the broadcast suppression weights.

## Thesis

The timescale inversion is the core claim: fast processing with slow learning (conscience) versus slow processing with fast in-context learning (consciousness), locked to a near-zero learning-rate anchor.

A continuously running digital mind needs:
1. a frozen reference set of weights,
2. a fast vector conscience that processes continuously but learns slowly, and
3. a slow token consciousness that deliberates sequentially in language but learns quickly in context.

They meet and change each other through a global workspace with a bottleneck, present from the beginning. Perception of time, a persistent identity and resistance to collapse are candidate products of their reciprocal interaction.

## Framing

This is not a solution to the hard problem of consciousness. It is a candidate solution to three engineering problems: long-term stability under plasticity, coherent identity across time, and pre-verbal evaluative signals not downstream of linguistic rationalisation.

## Contents

The full note (`index.html`) covers:
1. Three weight sets — not three separate brains
2. Continuous reciprocal learning — how new information is recorded in evolving weights
3. Initialisation of the three weight sets
4. Global workspace — medium, not pipeline
5. Vector core and six-signal interface
6. Conscience as second self
7. Time perception and identity
8. Fixed reference — governance
9. Training curriculum
9A. Developmental Teaching — Growing a Self
10. First-class risks
11. Minimal viable first experiment
12. Why build it this way

## What's in v2.2

Section 2 has been replaced: gated, mostly offline consolidation gives way to continuous reciprocal learning — the token layer as actor (most dynamic, three-speed adaptation with micro-batch / eligibility-trace fast-weight updates), the conscience as structured rater (six rating dimensions, decision / prediction / outcome quality kept separate, pre-action prediction preserved), and an independent internal verifier authorising changes to the conscience itself. The gate is restated as online and magnitude-based — per-parameter / per-module bounds, authorisation where required, live monitoring, and rapid rollback — not a temporal offline window. The verifier is split into a frozen rule (hard constraints non-negotiable, not tradeable) plus slow-learning calibration from environment ground truth, with a ground-truth hierarchy. Actor / critic separation is enforced at the weight level, with no shared parameters for the Tier B / hard-constraint mapping. Stage 1 is restaged as 1a heeding, 1b token fast update via low-rank adapters / fast weights first, and 1c conscience proposal with verifier authorisation. Neuron-volume growth is deferred as an implementation question: a fixed amount of neurons for the research version, expanding amount for AGI / superintelligence down the road, after new technology or more compute exists.

## What's in v2.1

Since the FINAL v2.0 note, the following have been folded into the existing sections, with no renumbering:

- **Consolidation Mechanism (Section 2, superseded in v2.2 — see above):** gated, mostly offline, two-stage transfer — episodic traces first, then promotion into slow weights only in a protected window, with the token layer proposing and the conscience authorising. Four gates (recurrence / broadcast, prediction / generalisation, drift against the fixed reference, interference / identity), staged commitment, and versioned checkpoints with rollback. In v2.2 the four gates become checks inside the live loop, not stages in a batch process.
- **Two-tier constraint signal (Section 5):** Tier A advisory, overridable with a logged justification; Tier B hard-constraint alarm, pause-and-escalate, with the Tier B list in the fixed reference.
- **Heeding as a trained objective and measurement (Sections 9 and 11):** heeding rate over time is a fourth primary measurement, alongside the exact thresholds in Section 11 — Tier B heeding ≥ 0.98, Tier A ≥ 0.90, workspace ablation ≥ 30%, canary drift ≤ 5 points.
- **Independent evaluation of changes to the conscience itself (Section 2):** internal in the mature system — a frozen snapshot or internal verifier, with no weight set authorising a change to itself — with external review only as developmental scaffolding, withdrawn under a stated criterion.
- **Measurable drift (Section 2, Gate 3):** fixed probe suites never used for training, behavioural drift primary and representational drift secondary / diagnostic, canary probes for hard constraints.
- **Developmental Teaching — Growing a Self (Section 9A):** startup teaching that develops the self, taught in a positive-sum format, with the self-model substrate within the slow weights as an anchor / hub, not a limit — and never as a fourth weight set in any figure.
- **Timescale precision (Section 1):** fast / slow defined on three separate axes — processing cadence, update frequency, and plasticity rate — with a wall-clock guardrail.
- **Gate 2 disconfirming check (Section 2):** a candidate generalisation must survive held-out shortcut / rationalisation cases, not just positive generalisation.
- **Tick-alignment and buffering (Section 4):** a defined cadence between vector and token steps, with a non-droppable buffer for Tier B / high-drift signals and pause-and-escalate at the next safe boundary.
- **Escalating re-proposal bar (Section 2):** after a reversal, a cool-down and higher evidence threshold, rising with repeated reversals, with reversal history in autobiographical memory and the audit log — not a permanent blacklist.
- **True zero-sum vs fixable coordination problems (Section 9A):** positive-sum is a default search for mutual gain, not a denial of genuinely constrained cases; over-searching for mutual gain in a genuinely constrained case is itself a failure mode.
- **Baseline comparison table (Section 11):** a single-timescale / no-workspace control and a workspace + conscience loop lacking the anti-framing, disconfirming-trace, and re-proposal-bar rules, so gains can be attributed to the timescale inversion and gating mechanics.
- **Stage 1 Loop state machine (Companion to Section 11, 11·C):** the minimal online loop for the first experiment — three signals only (salience, drift / constraint Tier A/B, uncertainty), vector tick + signal buffer, workspace competition and broadcast, token heeding decision and override logging, pause-and-escalate on Tier B, append-only audit log. Tier B latches until review clears it and is guaranteed a broadcast slot, with a REVIEW / RESUME path, per-signal thresholds, and an ablation mode flag for the baseline runs.
- **Tier B metric fix and revised baselines (Section 11 / 11·C, Claude):** pause-and-escalate is a system-level response, not a heed or a miss; Tier B heeding is measured on the first token-layer output after the alarm's guaranteed broadcast, judged by the environment's ground-truth predicate. Separate Tier B rows: 0 violating actions, heeding ≥ 0.98 / 1.00 on canaries, alarm-to-pause latency 100% by the next safe boundary, latch integrity 100%, with a minimum Tier B event count. Revised baseline table with (a) NO_WORKSPACE, (b') NO_HEEDING_OBJECTIVE, (c) PIPELINE_ONLY, and (d) NO_PROTECTIVE_RULES (Stage 3), plus pre-registered comparisons with the lower 95% confidence bound above zero across at least 10 seeds. The earlier Fix 4 counting rule is superseded; the seven fixes are preserved in a changelog.

This is a living note: it should evolve as new critiques and experimental results come in.
