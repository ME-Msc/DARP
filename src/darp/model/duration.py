"""Durative-action models and tau computations."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from math import ceil, erfc, isfinite, sqrt
from typing import Any

ActionName = str
StateKey = Hashable
Belief = Mapping[StateKey, float]
AugmentedStateKey = tuple[StateKey, float]
AugmentedBelief = Mapping[AugmentedStateKey, float]


@dataclass(frozen=True)
class DurationEstimate:
    """Store one action-duration estimate. / 保存一次动作时长估计。"""

    mean: float
    variance: float = 0.0


@dataclass(frozen=True)
class DurationProgress:
    """Track accumulated duration along a history. / 跟踪一条 history 上累计的动作时长。"""

    mean: float = 0.0
    variance: float = 0.0
    augmented_belief: AugmentedBelief | None = None

    def add(self, estimate: DurationEstimate) -> DurationProgress:
        """Return progress after adding one estimate. / 返回加入一次估计后的累计进度。"""
        if self.augmented_belief is not None:
            raise ValueError(
                "Augmented chance-duration progress must be advanced jointly with "
                "the state transition and observation."
            )
        mean = self.mean + estimate.mean
        variance = self.variance + estimate.variance
        return DurationProgress(
            mean=mean,
            variance=variance,
        )


class DurationModel:
    """Base class for duration models. / 动作时长模型基类。"""

    def estimate(self, belief: Belief, action: ActionName) -> DurationEstimate:
        """Estimate duration for an action under a belief. / 在给定 belief 下估计动作时长。"""
        raise NotImplementedError

    def tau(self, progress: DurationProgress, horizon: float) -> float:
        """Compute remaining-horizon feasibility. / 计算相对剩余 horizon 的可行度。"""
        raise NotImplementedError

    def should_continue(self, progress: DurationProgress, horizon: float, zeta: float) -> bool:
        """Return whether a history should keep expanding. / 判断一条 history 是否继续展开。"""
        return self.tau(progress, horizon) > zeta


@dataclass(frozen=True)
class HistoryDurationEvaluator:
    """Evaluate cumulative duration for histories, matching Phase 7 tree pruning. / 评估 history 累计时长以适配 Phase 7 树剪枝。"""

    model: DurationModel
    horizon: float
    zeta: float = 0.0

    def __post_init__(self) -> None:
        """Reject stopping rules that cannot define a finite search tree."""
        if not isfinite(self.horizon) or self.horizon <= 0.0:
            raise ValueError("duration horizon must be a finite positive number")
        if not isfinite(self.zeta) or self.zeta < 0.0:
            raise ValueError("duration zeta must be a finite non-negative number")
        if isinstance(self.model, (GaussianDurationModel, ChanceConstrainedDurationModel)) and self.zeta > 1.0:
            raise ValueError("probabilistic duration zeta must be in [0, 1]")
        root_tau = float(self.model.tau(DurationProgress(), self.horizon))
        if not isfinite(root_tau):
            raise ValueError("duration tau at the empty history must be finite")
        if not self.model.should_continue(
            DurationProgress(),
            self.horizon,
            self.zeta,
        ):
            # The planner API must return a root action, whereas the paper's
            # admissible history set is empty when tau(empty) <= zeta.  Reject
            # that no-action problem explicitly instead of forcing an action
            # outside the paper's policy space.
            raise ValueError(
                "duration stopping condition already holds at the empty "
                "history (tau(empty) must be greater than zeta)"
            )

    def action_depth_upper_bound(self) -> int | None:
        r"""Return a proof that Algorithm 1 must stop by this action depth.

        This is a *derived* bound on the paper's duration test, not an
        independent decision-step horizon.  ``None`` means no finite bound can
        be proved from the configured model (for example Gaussian noise with
        :math:`\zeta=0`, or chance duration with a possible zero-duration
        loop).  Search may still terminate branch by branch, but must not use
        the RDDL integer horizon as a substitute proof.
        """
        if isinstance(self.model, FixedDurationModel):
            minimum = min(
                (float(self.model.default), *(float(value) for value in self.model.durations.values()))
            )
            return _fixed_depth_bound(
                horizon=self.horizon,
                zeta=self.zeta,
                minimum_increment=minimum,
            )
        if isinstance(self.model, StateDependentDurationModel):
            # The per-step estimate is a normalized floating-point dot
            # product.  Although its real-arithmetic value is bounded below by
            # the smallest configured duration, normalization and summation
            # can undershoot that value by more than one ULP.  Do not turn the
            # configuration minimum into a false finite-depth bound;
            # branch-local tau checks still terminate the actual expansion.
            return None
        if isinstance(self.model, ChanceConstrainedDurationModel):
            if self.zeta >= 1.0:
                return 1
            minimum = min(
                (float(self.model.default), *(float(value) for value in self.model.durations.values()))
            )
            if minimum <= 0.0:
                return None
            return _deterministic_depth_bound(
                target=self.horizon,
                minimum_increment=minimum,
            )
        if isinstance(self.model, GaussianDurationModel):
            # Both Gaussian moments are belief-weighted floating-point sums.
            # Without interval arithmetic there is no machine-checkable
            # uniform lower/upper moment bound strong enough to prove a strict
            # tau boundary. Exhaust branches using their stored progress.
            return None
        return None


@dataclass(frozen=True)
class FixedDurationModel(DurationModel):
    """Fixed action durations, where tau is remaining time. / 固定动作时长模型，tau 表示剩余时间。"""

    durations: Mapping[ActionName, float]
    default: float = 1.0

    def __post_init__(self) -> None:
        """Require positive finite durations so tree expansion terminates."""
        _validate_positive_durations(
            self.durations.values(), default=self.default, model_name="fixed"
        )

    def estimate(self, belief: Belief, action: ActionName) -> DurationEstimate:
        """Return the configured fixed duration. / 返回配置中的固定动作时长。"""
        mean = float(self.durations.get(action, self.default))
        return DurationEstimate(mean=mean)

    def tau(self, progress: DurationProgress, horizon: float) -> float:
        """Return remaining time after accumulated duration. / 返回累计时长后的剩余时间。"""
        return float(horizon) - progress.mean

    def should_continue(self, progress: DurationProgress, horizon: float, zeta: float) -> bool:
        """Evaluate the paper's strict remaining-time test."""
        return float(horizon) - progress.mean > float(zeta)


@dataclass(frozen=True)
class StateDependentDurationModel(DurationModel):
    """Expected duration under the current belief. / 当前 belief 下的期望动作时长。"""

    durations: Mapping[tuple[StateKey, ActionName], float]
    default: float = 1.0

    def __post_init__(self) -> None:
        """Require positive finite expected-duration entries."""
        _validate_positive_durations(
            self.durations.values(), default=self.default, model_name="expected"
        )

    def estimate(self, belief: Belief, action: ActionName) -> DurationEstimate:
        """Return belief-weighted expected duration. / 返回 belief 加权的期望时长。"""
        mean = sum(
            probability * self.duration_for_state(state, action)
            for state, probability in _normalized_belief_items(belief)
        )
        return DurationEstimate(mean=mean)

    def duration_for_state(self, state: StateKey, action: ActionName) -> float:
        """Return :math:`D(s,a)` for one complete state."""
        return _state_action_value(self.durations, state, action, self.default)

    def tau(self, progress: DurationProgress, horizon: float) -> float:
        """Return remaining time after expected duration. / 返回期望累计时长后的剩余时间。"""
        return float(horizon) - progress.mean

    def should_continue(self, progress: DurationProgress, horizon: float, zeta: float) -> bool:
        """Evaluate the strict expected-duration boundary."""
        return float(horizon) - progress.mean > float(zeta)


@dataclass(frozen=True)
class ChanceConstrainedDurationModel(DurationModel):
    r"""Deterministic :math:`D(s,a)` with an augmented-state chance bound.

    The sufficient statistic for a history is the posterior over
    :math:`(S_q,G_q)`, where :math:`G_q` is accumulated duration.  Algorithm 2
    updates that distribution jointly with each transition and observation;
    retaining only the marginal state belief or expected duration loses the
    correlation required by the paper.

    / 为确定性状态依赖时长保留论文中的增广状态
    ``(state, accumulated duration)`` 后验分布。
    """

    durations: Mapping[tuple[StateKey, ActionName], float]
    default: float = 1.0

    def __post_init__(self) -> None:
        """Require finite non-negative deterministic durations."""
        _validate_non_negative_durations(
            self.durations.values(), default=self.default, model_name="chance"
        )

    def duration_for_state(self, state: StateKey, action: ActionName) -> float:
        """Return deterministic :math:`D(s,a)` for one complete state."""
        return _state_action_value(self.durations, state, action, self.default)

    def tau(self, progress: DurationProgress, horizon: float) -> float:
        r"""Return :math:`Pr(G_q < h \mid q)` from the augmented belief."""
        probability, total = _chance_duration_mass(progress, horizon)
        return probability / total if total > 0.0 else 0.0

    def should_continue(self, progress: DurationProgress, horizon: float, zeta: float) -> bool:
        """Compare augmented safe-duration mass with zeta."""
        numerator, denominator = _chance_duration_mass(progress, horizon)
        return denominator > 0.0 and numerator > float(zeta) * denominator


@dataclass(frozen=True)
class GaussianDurationModel(DurationModel):
    """Gaussian percentile duration model. / Gaussian 百分位动作时长模型。"""

    means: Mapping[tuple[StateKey, ActionName], float]
    variances: Mapping[tuple[StateKey, ActionName], float]
    default_mean: float = 1.0
    default_variance: float = 0.0

    def __post_init__(self) -> None:
        """Validate Gaussian moments before they control tree expansion."""
        _validate_positive_durations(
            self.means.values(), default=self.default_mean, model_name="gaussian"
        )
        variances = (*self.variances.values(), self.default_variance)
        if any(not isfinite(float(value)) or float(value) < 0.0 for value in variances):
            raise ValueError("gaussian duration variances must be finite and non-negative")

    def estimate(self, belief: Belief, action: ActionName) -> DurationEstimate:
        """Return belief-weighted Gaussian mean and variance. / 返回 belief 加权的 Gaussian 均值与方差。"""
        mean_terms: list[float] = []
        variance_terms: list[float] = []
        for state, probability in _normalized_belief_items(belief):
            state_mean, state_variance = self.moments_for_state(state, action)
            mean_terms.append(probability * state_mean)
            # Paper Sec. 3: sigma_q^2 = sum_i sum_s b_i(s)^2 sigma^2_{s,a_i}.
            variance_terms.append(probability**2 * state_variance)
        return DurationEstimate(mean=sum(mean_terms), variance=sum(variance_terms))

    def moments_for_state(self, state: StateKey, action: ActionName) -> tuple[float, float]:
        """Return the configured Gaussian moments for one complete state."""
        return (
            _state_action_value(self.means, state, action, self.default_mean),
            _state_action_value(self.variances, state, action, self.default_variance),
        )

    def tau(self, progress: DurationProgress, horizon: float) -> float:
        r"""Return the numerical Gaussian probability :math:`Pr(G<h)`.

        The paper permits standard numerical evaluation of the Gaussian CDF;
        ``erfc`` avoids cancellation in ``1 - erf``.
        """
        variance = progress.variance
        centered = progress.mean - float(horizon)
        if variance <= 0.0:
            return 1.0 if centered < 0.0 else 0.0
        if centered == 0.0:
            return 0.5
        squared_distance = centered * centered / (2 * variance)
        standardized = sqrt(squared_distance)
        probability = 0.5 * erfc(
            standardized if centered > 0 else -standardized
        )
        return min(1.0, max(0.0, probability))

    def should_continue(
        self,
        progress: DurationProgress,
        horizon: float,
        zeta: float,
    ) -> bool:
        """Apply Algorithm 2's strict continuation test ``tau(q) > zeta``.

        Degenerate and symmetry cases avoid unnecessary floating-point work.
        The ``zeta == 0`` case is analytic because every non-degenerate
        Gaussian has positive mass below any finite horizon, even when that
        tail is too small for binary64 ``erfc`` to represent.
        """
        threshold = float(zeta)
        variance = progress.variance
        centered = progress.mean - float(horizon)
        if variance <= 0.0:
            probability = 1.0 if centered < 0.0 else 0.0
            return probability > threshold
        if threshold <= 0.0:
            # Every non-degenerate Gaussian assigns positive mass below every
            # finite boundary, including tails below binary64's range.
            return True
        if threshold >= 1.0:
            return False
        if centered == 0.0:
            return 0.5 > threshold
        if centered < 0.0 and threshold <= 0.5:
            return True
        if centered > 0.0 and threshold >= 0.5:
            return False
        return self.tau(progress, horizon) > float(zeta)


def _validate_positive_durations(
    values: Iterable[float], *, default: float, model_name: str
) -> None:
    """Require every possible duration to advance time by a finite amount."""
    durations = (*values, default)
    if any(not isfinite(float(value)) or float(value) <= 0.0 for value in durations):
        raise ValueError(
            f"{model_name} durations must be finite and strictly positive"
        )


def _validate_non_negative_durations(
    values: Iterable[float], *, default: float, model_name: str
) -> None:
    """Validate deterministic chance-duration entries, for which zero is meaningful."""
    durations = (*values, default)
    if any(not isfinite(float(value)) or float(value) < 0.0 for value in durations):
        raise ValueError(f"{model_name} durations must be finite and non-negative")


def _deterministic_depth_bound(
    *,
    target: float,
    minimum_increment: float,
) -> int | None:
    """Bound strict continuation using configured durations."""
    if target <= 0.0:
        return 1
    if not isfinite(minimum_increment) or minimum_increment <= 0.0:
        return None
    estimate = max(1, ceil(float(target) / float(minimum_increment)))
    if estimate > 1_000_000:
        return None
    return estimate


def _fixed_depth_bound(
    *,
    horizon: float,
    zeta: float,
    minimum_increment: float,
) -> int | None:
    """Prove a bound for the strict test ``h-elapsed > zeta``."""
    if not isfinite(minimum_increment) or minimum_increment <= 0.0:
        return None
    target = float(horizon) - float(zeta)
    if target <= 0.0:
        return 1
    depth = max(1, ceil(target / float(minimum_increment)))
    return depth if depth <= 1_000_000 else None


def _chance_duration_mass(
    progress: DurationProgress,
    horizon: float,
) -> tuple[float, float]:
    """Return ``(mass below horizon, total mass)`` for chance duration."""
    horizon_value = float(horizon)
    if progress.augmented_belief is None:
        return (
            (1.0, 1.0)
            if progress.mean < horizon_value
            else (0.0, 1.0)
        )
    numerator = sum(
        probability
        for (_, elapsed), probability in progress.augmented_belief.items()
        if elapsed < horizon_value
    )
    denominator = sum(progress.augmented_belief.values())
    return numerator, denominator


def _normalized_belief_items(
    belief: Belief,
) -> tuple[tuple[StateKey, float], ...]:
    """Normalize positive finite belief entries as floating-point weights."""
    entries = tuple(
        (state, float(probability))
        for state, probability in belief.items()
    )
    if not entries:
        raise ValueError("state-dependent duration requires a non-empty joint-state belief")
    if any(not isfinite(probability) or probability < 0.0 for _, probability in entries):
        raise ValueError("duration belief probabilities must be finite and non-negative")
    positive = tuple(
        (state, probability)
        for state, probability in entries
        if probability > 0.0
    )
    total = sum(probability for _, probability in positive)
    if total <= 0.0:
        raise ValueError("duration belief must contain positive probability mass")
    return tuple((state, probability / total) for state, probability in positive)


def _state_action_value(
    values: Mapping[tuple[StateKey, ActionName], float],
    state: StateKey,
    action: ActionName,
    default: float,
) -> float:
    r"""Resolve one state/action value without confusing marginals with states.

    Programmatic models may key entries by the complete hashable state directly.
    Sidecars additionally support Boolean fluent selectors (``muddy`` means
    ``muddy=true``) and conjunctions such as ``muddy=true&loaded=false``.  The
    most specific matching selector wins; conflicting equal-specificity rules
    are rejected instead of being silently summed.
    """
    try:
        direct_key = (state, action)
        if direct_key in values:
            return float(values[direct_key])
    except TypeError:
        pass

    state_mapping = _state_mapping(state)
    if state_mapping is None:
        return float(default)

    matches: list[tuple[int, str, float]] = []
    for (selector, configured_action), value in values.items():
        if configured_action != action or not isinstance(selector, str):
            continue
        specificity = _selector_specificity(selector, state_mapping)
        if specificity is not None:
            matches.append((specificity, selector, float(value)))
    if not matches:
        return float(default)

    best_specificity = max(specificity for specificity, _, _ in matches)
    best = [(selector, value) for specificity, selector, value in matches if specificity == best_specificity]
    distinct = {value for _, value in best}
    if len(distinct) > 1:
        selectors = ", ".join(sorted(selector for selector, _ in best))
        raise ValueError(
            f"Conflicting duration selectors for action {action!r} and state {state!r}: {selectors}"
        )
    return best[0][1]


def _state_mapping(state: StateKey) -> Mapping[str, Any] | None:
    """Return a mapping view for the kernel's tuple-of-pairs state key."""
    if isinstance(state, Mapping):
        return {str(name): value for name, value in state.items()}
    if isinstance(state, tuple) and all(isinstance(entry, tuple) and len(entry) == 2 for entry in state):
        return {str(name): value for name, value in state}
    return None


def _selector_specificity(selector: str, state: Mapping[str, Any]) -> int | None:
    """Return matched-clause count for one sidecar Boolean-state selector."""
    clauses = selector.split("&")
    parsed: list[tuple[str, bool]] = []
    for clause in clauses:
        name, separator, raw_value = clause.partition("=")
        name = name.strip()
        if not name:
            return None
        if not separator:
            expected = True
        else:
            normalized = raw_value.strip().lower()
            if normalized not in {"true", "false"}:
                return None
            expected = normalized == "true"
        parsed.append((name, expected))
    if all(name in state and bool(state[name]) is expected for name, expected in parsed):
        return len(parsed)
    return None
