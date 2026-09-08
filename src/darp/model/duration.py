"""Durative-action models and tau computations."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from math import ceil, erfc, isfinite, sqrt

ActionName = str
StateKey = Hashable
Belief = Mapping[StateKey, float]
AugmentedStateKey = tuple[StateKey, float]
AugmentedBelief = Mapping[AugmentedStateKey, float]
StateDuration = Callable[[StateKey, ActionName], float]
StateDurationMoments = Callable[[StateKey, ActionName], "DurationEstimate"]


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
            return None
        if isinstance(self.model, ChanceConstrainedDurationModel):
            if self.zeta >= 1.0:
                return 1
            return None
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

    duration: StateDuration

    def estimate(self, belief: Belief, action: ActionName) -> DurationEstimate:
        """Return belief-weighted expected duration. / 返回 belief 加权的期望时长。"""
        mean = sum(
            probability * self.duration_for_state(state, action)
            for state, probability in _normalized_belief_items(belief)
        )
        return DurationEstimate(mean=mean)

    def duration_for_state(self, state: StateKey, action: ActionName) -> float:
        """Evaluate and validate :math:`D(s,a)` for one complete state."""
        value = float(self.duration(state, action))
        _validate_positive_duration(value, model_name="expected")
        return value

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

    duration: StateDuration

    def duration_for_state(self, state: StateKey, action: ActionName) -> float:
        """Evaluate deterministic :math:`D(s,a)` for one complete state."""
        value = float(self.duration(state, action))
        if not isfinite(value) or value < 0.0:
            raise ValueError("chance durations must be finite and non-negative")
        return value

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

    moments: StateDurationMoments

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
        """Evaluate and validate Gaussian moments for one complete state."""
        estimate = self.moments(state, action)
        mean = float(estimate.mean)
        variance = float(estimate.variance)
        _validate_positive_duration(mean, model_name="gaussian")
        if not isfinite(variance) or variance < 0.0:
            raise ValueError("gaussian duration variances must be finite and non-negative")
        return mean, variance

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


def _validate_positive_duration(value: float, *, model_name: str) -> None:
    """Validate one lazily evaluated duration value."""
    if not isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{model_name} durations must be finite and strictly positive"
        )


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
