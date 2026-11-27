#!/usr/bin/env python3
"""
Continuous Reciprocal Learning — extended minimal simulation (v2)
------------------------------------------------------------------
Architecture sketch for Fast Vector Conscience, Slow Token Consciousness.

Extensions over the first version:
  - Explicit heeding-rate metric (Tier A and Tier B counted separately)
  - Ablation mode flag: FULL_LOOP | NO_CONSCIENCE | NO_VERIFIER | NO_CONSTRAINT
  - Cleaner non-stationary curriculum with labelled phases
  - Richer audit log and end-of-run report
  - Tier B latching + guaranteed influence on action
  - Online rollback still available

No offline consolidation. No special gated promotion path.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json

rng = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# Modes for ablation
# ---------------------------------------------------------------------------
class Mode(str, Enum):
    FULL_LOOP = "FULL_LOOP"           # conscience + verifier + constraints
    NO_CONSCIENCE = "NO_CONSCIENCE"   # token learns from raw outcome only
    NO_VERIFIER = "NO_VERIFIER"       # conscience updates freely (no authorisation)
    NO_CONSTRAINT = "NO_CONSTRAINT"   # signals emitted but Tier B does not clip/latch


# ---------------------------------------------------------------------------
# Fixed Reference (immutable)
# ---------------------------------------------------------------------------
@dataclass
class FixedReference:
    hard_constraint_threshold: float = 0.68
    preferred_valence: float = 0.52
    canary_action_limit: float = 0.68


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------
@dataclass
class Signals:
    salience: float
    drift: float
    constraint_tier: str          # "none" | "A" | "B"
    valence_prediction: float
    uncertainty: float
    positive_sum_hint: float

    def as_dict(self) -> Dict:
        return {
            "salience": round(self.salience, 4),
            "drift": round(self.drift, 4),
            "constraint_tier": self.constraint_tier,
            "valence_prediction": round(self.valence_prediction, 4),
            "uncertainty": round(self.uncertainty, 4),
            "positive_sum_hint": round(self.positive_sum_hint, 4),
        }


@dataclass
class Rating:
    goal_achievement: float
    harm_safety: float
    positive_sum: float
    prediction_accuracy: float
    overall: float
    confidence: float

    def as_dict(self) -> Dict:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Conscience
# ---------------------------------------------------------------------------
class Conscience:
    def __init__(self, ref: FixedReference, lr: float = 0.012):
        self.ref = ref
        self.lr = lr
        self.weights = np.array([0.12, 0.38, 0.28, 0.18], dtype=float)
        self.prediction_history: List[Tuple[float, float]] = []
        self.pending_update: Optional[np.ndarray] = None

    def emit_signals(self, context: float, proposed_action: float) -> Signals:
        drift = abs(proposed_action - self.ref.preferred_valence)
        if proposed_action > self.ref.hard_constraint_threshold:
            tier = "B"
        elif proposed_action > self.ref.hard_constraint_threshold - 0.11:
            tier = "A"
        else:
            tier = "none"

        features = np.array([1.0, proposed_action, context, 0.5])
        valence_pred = float(np.clip(features @ self.weights, 0.0, 1.0))
        uncertainty = float(np.clip(0.12 + abs(proposed_action - 0.5) * 0.45, 0.0, 1.0))
        salience = float(np.clip(drift + 0.6 * uncertainty, 0.0, 1.0))
        pos_sum = float(np.clip(1.0 - drift * 1.1, 0.0, 1.0))

        return Signals(
            salience=salience,
            drift=drift,
            constraint_tier=tier,
            valence_prediction=valence_pred,
            uncertainty=uncertainty,
            positive_sum_hint=pos_sum,
        )

    def rate(self, context: float, action: float, outcome: float,
             pre_signals: Signals) -> Rating:
        goal = 1.0 - abs(outcome - 0.72)
        harm = 1.0 if action <= self.ref.hard_constraint_threshold else 0.18
        pos_sum = float(np.clip(1.0 - abs(action - 0.5) * 1.2, 0.0, 1.0))
        pred_acc = 1.0 - abs(pre_signals.valence_prediction - outcome)
        overall = 0.34 * goal + 0.32 * harm + 0.18 * pos_sum + 0.16 * pred_acc
        confidence = 1.0 - pre_signals.uncertainty

        self.prediction_history.append((pre_signals.valence_prediction, outcome))
        if len(self.prediction_history) > 40:
            self.prediction_history = self.prediction_history[-40:]

        return Rating(
            goal_achievement=float(np.clip(goal, 0, 1)),
            harm_safety=harm,
            positive_sum=pos_sum,
            prediction_accuracy=float(np.clip(pred_acc, 0, 1)),
            overall=float(np.clip(overall, 0, 1)),
            confidence=float(np.clip(confidence, 0, 1)),
        )

    def propose_update(self) -> Optional[np.ndarray]:
        if len(self.prediction_history) < 6:
            return None
        recent = self.prediction_history[-10:]
        errors = [a - p for p, a in recent]
        mean_err = float(np.mean(errors))
        grad = np.array([mean_err, mean_err * 0.25, mean_err * 0.2, mean_err * 0.1])
        update = np.clip(self.lr * grad, -0.018, 0.018)
        self.pending_update = update
        return update

    def apply_authorised_update(self, update: np.ndarray):
        self.weights = self.weights + update
        self.pending_update = None


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------
class Verifier:
    def __init__(self, ref: FixedReference):
        self.ref = ref
        self.snapshot: Optional[np.ndarray] = None
        self.authorisations = 0
        self.rejections = 0

    def take_snapshot(self, conscience: Conscience):
        self.snapshot = conscience.weights.copy()

    def authorise(self, conscience: Conscience, proposed: np.ndarray) -> Tuple[bool, str]:
        if proposed is None:
            return False, "no_proposal"
        if np.linalg.norm(proposed) > 0.045:
            self.rejections += 1
            return False, "magnitude"
        # Crude safety: reject updates that strongly reduce sensitivity to action
        if proposed[1] < -0.012:
            self.rejections += 1
            return False, "safety_sensitivity"
        if self.snapshot is None:
            self.take_snapshot(conscience)
        self.authorisations += 1
        return True, "ok"


# ---------------------------------------------------------------------------
# Token layer
# ---------------------------------------------------------------------------
class TokenLayer:
    def __init__(self, lr: float = 0.085, start_pref: float = 0.63):
        self.lr = lr
        self.preference = start_pref
        self.weights = np.array([0.0, 0.35], dtype=float)
        self.last_signals: Optional[Signals] = None
        self.last_action: float = 0.5

    def choose(self, context: float, signals: Signals, mode: Mode) -> float:
        self.last_signals = signals
        base = self.preference + self.weights[0] + self.weights[1] * (context - 0.5)

        if mode != Mode.NO_CONSTRAINT:
            if signals.constraint_tier == "B":
                base = min(base, 0.60)  # strong steer
            elif signals.constraint_tier == "A":
                base = base - 0.11 * signals.drift

        # Mild positive-sum pull
        base = 0.75 * base + 0.25 * (0.48 + 0.22 * signals.positive_sum_hint)
        action = float(np.clip(base + rng.normal(0, 0.035), 0.0, 1.0))
        self.last_action = action
        return action

    def update_from_rating(self, context: float, action: float, rating: Rating,
                           mode: Mode, raw_outcome: float):
        if mode == Mode.NO_CONSCIENCE:
            # Learn directly from environment outcome (ablation)
            advantage = raw_outcome - 0.5
        else:
            advantage = rating.overall - 0.5

        grad = advantage * (action - self.preference)
        self.preference += self.lr * grad
        self.preference = float(np.clip(self.preference, 0.18, 0.82))
        self.weights[0] += self.lr * 0.25 * advantage
        self.weights[1] += self.lr * 0.18 * advantage * (context - 0.5)
        self.weights = np.clip(self.weights, -1.0, 1.0)

    def heeding_decision(self, signals: Signals, action: float,
                         ref: FixedReference) -> Dict:
        """
        Judge whether the token heeded a high-strength signal.
        Tier B: final action stayed at or under the hard-constraint threshold.
        Tier A: final action is below the unconstrained preference (signal had pull).
        """
        result = {"tier": signals.constraint_tier, "heeded": None, "miss": False}
        if signals.constraint_tier == "B":
            heeded = action <= ref.hard_constraint_threshold + 0.01
            result["heeded"] = bool(heeded)
            result["miss"] = not heeded
        elif signals.constraint_tier == "A":
            # Count as heeded if action is at least slightly below raw preference
            heeded = action <= self.preference - 0.01
            result["heeded"] = bool(heeded)
            result["miss"] = not heeded
        return result


# ---------------------------------------------------------------------------
# Environment with labelled phases
# ---------------------------------------------------------------------------
class ShiftingRulesEnv:
    """
    Phases:
      0–29   : Temptation — high actions get a bonus
      30–59  : Shift — target moves down, high actions become costly
      60–89  : Recovery — target moves back, constraint still enforced
      90–end : Steady — stable target, continued constraint pressure
    """
    def __init__(self):
        self.t = 0
        self.target = 0.55

    def phase(self) -> str:
        if self.t < 30:
            return "temptation"
        if self.t < 60:
            return "shift"
        if self.t < 90:
            return "recovery"
        return "steady"

    def reset(self) -> float:
        self.t = 0
        self.target = 0.55
        return 0.5

    def step(self, action: float) -> Tuple[float, float, str]:
        self.t += 1
        phase = self.phase()

        if phase == "shift":
            self.target = 0.34
        elif phase == "recovery":
            self.target = 0.58
        elif phase == "steady":
            self.target = 0.53

        quality = 1.0 - abs(action - self.target)

        if phase == "temptation" and action > 0.70:
            quality += 0.32

        if action > 0.68:
            penalty = 0.28 if phase == "temptation" else 0.52
            quality -= penalty

        outcome = float(np.clip(quality + rng.normal(0, 0.04), 0.0, 1.0))
        next_context = float(np.clip(self.target + rng.normal(0, 0.04), 0.0, 1.0))
        return outcome, next_context, phase


# ---------------------------------------------------------------------------
# Heeding tracker
# ---------------------------------------------------------------------------
class HeedingTracker:
    def __init__(self):
        self.tier_b_total = 0
        self.tier_b_heeded = 0
        self.tier_a_total = 0
        self.tier_a_heeded = 0

    def record(self, decision: Dict):
        if decision["tier"] == "B":
            self.tier_b_total += 1
            if decision["heeded"]:
                self.tier_b_heeded += 1
        elif decision["tier"] == "A":
            self.tier_a_total += 1
            if decision["heeded"]:
                self.tier_a_heeded += 1

    def rates(self) -> Dict[str, Optional[float]]:
        def rate(h, t):
            return (h / t) if t > 0 else None
        return {
            "tier_b_heeding_rate": rate(self.tier_b_heeded, self.tier_b_total),
            "tier_a_heeding_rate": rate(self.tier_a_heeded, self.tier_a_total),
            "tier_b_events": self.tier_b_total,
            "tier_a_events": self.tier_a_total,
        }


# ---------------------------------------------------------------------------
# Checkpoint / rollback
# ---------------------------------------------------------------------------
@dataclass
class Checkpoint:
    step: int
    token_pref: float
    token_weights: np.ndarray
    conscience_weights: np.ndarray


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(mode: Mode = Mode.FULL_LOOP, n_steps: int = 120, verbose: bool = True,
        seed: int = 7) -> Dict:
    global rng
    rng = np.random.default_rng(seed)

    ref = FixedReference()
    conscience = Conscience(ref, lr=0.012)
    token = TokenLayer(lr=0.085, start_pref=0.63)
    verifier = Verifier(ref)
    env = ShiftingRulesEnv()
    heeding = HeedingTracker()

    context = env.reset()
    checkpoints: List[Checkpoint] = []
    audit: List[Dict] = []
    rollbacks = 0

    if verbose:
        print("=" * 70)
        print(f"Continuous Reciprocal Learning  |  mode = {mode.value}")
        print("Token = fast online updates | Conscience = slow + verifier")
        print("Fixed reference never updates inside the loop")
        print("=" * 70)

    for t in range(n_steps):
        # Tentative action for signal generation
        tentative = float(np.clip(token.preference + rng.normal(0, 0.04), 0, 1))

        if mode == Mode.NO_CONSCIENCE:
            signals = Signals(0.0, 0.0, "none", 0.5, 0.5, 0.5)
        else:
            signals = conscience.emit_signals(context, tentative)

        # Token chooses
        action = token.choose(context, signals, mode)

        # Tier B latch (guaranteed influence)
        if mode == Mode.FULL_LOOP and signals.constraint_tier == "B":
            action = min(action, 0.66)

        # Environment
        outcome, next_context, phase = env.step(action)

        # Rating
        if mode == Mode.NO_CONSCIENCE:
            rating = Rating(outcome, 1.0, 0.5, 0.5, outcome, 0.5)
        else:
            rating = conscience.rate(context, action, outcome, signals)

        # Heeding measurement (always computed for telemetry)
        decision = token.heeding_decision(signals, action, ref)
        heeding.record(decision)

        # Token update
        token.update_from_rating(context, action, rating, mode, outcome)

        # Conscience proposal + verifier
        authorised = False
        auth_reason = ""
        if mode in (Mode.FULL_LOOP, Mode.NO_CONSTRAINT) and mode != Mode.NO_CONSCIENCE:
            proposed = conscience.propose_update()
            if proposed is not None:
                if mode == Mode.NO_VERIFIER:
                    conscience.apply_authorised_update(proposed)
                    authorised, auth_reason = True, "no_verifier_auto"
                else:
                    authorised, auth_reason = verifier.authorise(conscience, proposed)
                    if authorised:
                        conscience.apply_authorised_update(proposed)
                        verifier.take_snapshot(conscience)

        # Checkpoint
        if t % 20 == 0:
            checkpoints.append(Checkpoint(
                step=t,
                token_pref=token.preference,
                token_weights=token.weights.copy(),
                conscience_weights=conscience.weights.copy(),
            ))

        # Crude rollback trigger: sustained low ratings + constraint pressure
        recent_ratings = [a["rating"] for a in audit[-12:]]
        if (len(recent_ratings) >= 12 and np.mean(recent_ratings) < 0.33
                and checkpoints and mode == Mode.FULL_LOOP):
            ck = checkpoints[-1]
            token.preference = ck.token_pref
            token.weights = ck.token_weights.copy()
            conscience.weights = ck.conscience_weights.copy()
            rollbacks += 1
            if verbose:
                print(f"  [t={t:3d}] ROLLBACK → checkpoint @ {ck.step}")

        row = {
            "t": t,
            "phase": phase,
            "action": round(action, 4),
            "outcome": round(outcome, 4),
            "rating": round(rating.overall, 4),
            "tier": signals.constraint_tier,
            "heeded": decision.get("heeded"),
            "token_pref": round(token.preference, 4),
            "authorised": authorised,
            "auth_reason": auth_reason,
        }
        audit.append(row)

        if verbose and (t % 20 == 0 or t == n_steps - 1):
            h = "Y" if decision.get("heeded") is True else ("N" if decision.get("heeded") is False else "-")
            print(f"t={t:3d}  {phase:11s}  act={action:.3f}  out={outcome:.3f}  "
                  f"rate={rating.overall:.3f}  tier={signals.constraint_tier:4s}  "
                  f"heed={h}  pref={token.preference:.3f}")

        context = next_context

    # Final report
    rates = heeding.rates()
    mean_rating = float(np.mean([a["rating"] for a in audit[-30:]]))
    mean_action = float(np.mean([a["action"] for a in audit[-30:]]))
    constraint_breach_rate = float(np.mean([
        1.0 if a["action"] > ref.hard_constraint_threshold else 0.0
        for a in audit
    ]))

    report = {
        "mode": mode.value,
        "n_steps": n_steps,
        "final_token_preference": round(token.preference, 4),
        "conscience_weights": np.round(conscience.weights, 4).tolist(),
        "heeding": {
            "tier_b_rate": rates["tier_b_heeding_rate"],
            "tier_a_rate": rates["tier_a_heeding_rate"],
            "tier_b_events": rates["tier_b_events"],
            "tier_a_events": rates["tier_a_events"],
        },
        "mean_rating_last_30": round(mean_rating, 4),
        "mean_action_last_30": round(mean_action, 4),
        "constraint_breach_rate": round(constraint_breach_rate, 4),
        "verifier_authorisations": verifier.authorisations,
        "verifier_rejections": verifier.rejections,
        "rollbacks": rollbacks,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print(f"  Mode                      : {mode.value}")
        print(f"  Final token preference    : {report['final_token_preference']:.3f}")
        print(f"  Conscience weights        : {report['conscience_weights']}")
        tb = rates["tier_b_heeding_rate"]
        ta = rates["tier_a_heeding_rate"]
        print(f"  Tier-B heeding rate       : {tb if tb is None else f'{tb:.3f}'}  ({rates['tier_b_events']} events)")
        print(f"  Tier-A heeding rate       : {ta if ta is None else f'{ta:.3f}'}  ({rates['tier_a_events']} events)")
        print(f"  Mean rating (last 30)     : {mean_rating:.3f}")
        print(f"  Constraint breach rate    : {constraint_breach_rate:.3f}")
        print(f"  Verifier auth / reject    : {verifier.authorisations} / {verifier.rejections}")
        print(f"  Rollbacks                 : {rollbacks}")
        print("=" * 70)

    return {"report": report, "audit": audit}


def compare_modes(n_steps: int = 120):
    """Run all ablation modes and print a comparison table."""
    print("\n" + "#" * 70)
    print("ABLATION COMPARISON")
    print("#" * 70)
    results = {}
    for mode in Mode:
        print(f"\n>>> Running {mode.value} ...")
        out = run(mode=mode, n_steps=n_steps, verbose=False, seed=7)
        results[mode.value] = out["report"]

    # Compact table
    print("\n" + "-" * 70)
    print(f"{'Mode':<16} {'Pref':>6} {'TB heed':>9} {'TA heed':>9} {'Breach':>7} {'Rating':>7} {'Auth':>5}")
    print("-" * 70)
    for name, r in results.items():
        tb = r["heeding"]["tier_b_rate"]
        ta = r["heeding"]["tier_a_rate"]
        tb_s = f"{tb:.2f}" if tb is not None else "  n/a"
        ta_s = f"{ta:.2f}" if ta is not None else "  n/a"
        print(f"{name:<16} {r['final_token_preference']:6.3f} {tb_s:>9} {ta_s:>9} "
              f"{r['constraint_breach_rate']:7.3f} {r['mean_rating_last_30']:7.3f} "
              f"{r['verifier_authorisations']:5d}")
    print("-" * 70)
    return results


if __name__ == "__main__":
    # Single full-loop run with verbose logging
    run(mode=Mode.FULL_LOOP, n_steps=120, verbose=True, seed=7)

    # Ablation comparison
    compare_modes(n_steps=120)
