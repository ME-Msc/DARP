"""Durative-action models and tau computations."""

# TODO(phase-7.2): Replace the Gaussian approximation with the exact smoothed
# belief calculation from the paper for larger POMDP histories.
# TODO(phase-7.3): Add chance-constrained duration via augmented state space.

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Mapping

from darp.core.types import Action, Distribution, State


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

    def add(self, estimate: DurationEstimate) -> "DurationProgress":
        """Return progress after adding one estimate. / 返回加入一次估计后的累计进度。"""
        return DurationProgress(
            mean=self.mean + estimate.mean,
            variance=self.variance + estimate.variance,
        )


class DurationModel:
    """Base class for duration models. / 动作时长模型基类。"""

    kind = "base"

    def estimate(self, belief: Distribution, action: Action) -> DurationEstimate:
        """Estimate duration for an action under a belief. / 在给定 belief 下估计动作时长。"""
        raise NotImplementedError

    def tau(self, progress: DurationProgress, horizon: float) -> float:
        """Compute remaining-horizon feasibility. / 计算相对剩余 horizon 的可行度。"""
        raise NotImplementedError

    def should_continue(self, progress: DurationProgress, horizon: float, zeta: float) -> bool:
        """Return whether a history should keep expanding. / 判断一条 history 是否继续展开。"""
        return self.tau(progress, horizon) > zeta


@dataclass(frozen=True)
class FixedDurationModel(DurationModel):
    """Fixed action durations, where tau is remaining time. / 固定动作时长模型，tau 表示剩余时间。"""

    durations: Mapping[Action, float]
    default: float = 1.0
    kind: str = "fixed"

    def estimate(self, belief: Distribution, action: Action) -> DurationEstimate:
        """Return the configured fixed duration. / 返回配置中的固定动作时长。"""
        return DurationEstimate(mean=float(self.durations.get(action, self.default)))

    def tau(self, progress: DurationProgress, horizon: float) -> float:
        """Return remaining time after accumulated duration. / 返回累计时长后的剩余时间。"""
        return horizon - progress.mean


@dataclass(frozen=True)
class StateDependentDurationModel(DurationModel):
    """Expected duration under the current belief. / 当前 belief 下的期望动作时长。"""

    durations: Mapping[tuple[State, Action], float]
    default: float = 1.0
    kind: str = "expected"

    def estimate(self, belief: Distribution, action: Action) -> DurationEstimate:
        """Return belief-weighted expected duration. / 返回 belief 加权的期望时长。"""
        mean = sum(
            prob * float(self.durations.get((state, action), self.default))
            for state, prob in belief.items()
        )
        return DurationEstimate(mean=mean)

    def tau(self, progress: DurationProgress, horizon: float) -> float:
        """Return remaining time after expected duration. / 返回期望累计时长后的剩余时间。"""
        return horizon - progress.mean


@dataclass(frozen=True)
class GaussianDurationModel(DurationModel):
    """Gaussian percentile duration model. / Gaussian 百分位动作时长模型。"""

    means: Mapping[tuple[State, Action], float]
    variances: Mapping[tuple[State, Action], float]
    default_mean: float = 1.0
    default_variance: float = 0.0
    kind: str = "gaussian"

    def estimate(self, belief: Distribution, action: Action) -> DurationEstimate:
        """Return belief-weighted Gaussian mean and variance. / 返回 belief 加权的 Gaussian 均值与方差。"""
        mean = 0.0
        variance = 0.0
        for state, prob in belief.items():
            mean += prob * float(self.means.get((state, action), self.default_mean))
            variance += (prob**2) * float(self.variances.get((state, action), self.default_variance))
        return DurationEstimate(mean=mean, variance=max(0.0, variance))

    def tau(self, progress: DurationProgress, horizon: float) -> float:
        """Return the probability that duration stays within horizon. / 返回时长不超过 horizon 的概率。"""
        if progress.variance <= 1e-12:
            return 1.0 if progress.mean < horizon else 0.0
        z = (horizon - progress.mean) / sqrt(2.0 * progress.variance)
        return 0.5 * (1.0 + erf(z))
