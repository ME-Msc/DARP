"""Durative-action models and tau computations."""

# TODO(phase-9.3): Replace the Gaussian approximation with the exact smoothed
# belief calculation from the paper for larger POMDP histories.
# TODO(phase-9.3): Add chance-constrained duration via augmented state space.

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Hashable
from math import erf, sqrt
from typing import Iterable, Mapping

from darp.model.and_or_tree import History

ActionName = str
StateKey = Hashable
Belief = Mapping[StateKey, float]


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


@dataclass(frozen=True)
class HistoryDurationRecord:
    """Store one action's duration contribution along a history. / 保存 history 中一次 action 的 duration 贡献。"""

    action: ActionName
    belief: Belief
    estimate: DurationEstimate
    progress: DurationProgress


class DurationModel:
    """Base class for duration models. / 动作时长模型基类。"""

    kind = "base"

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
    default_belief: Belief = field(default_factory=lambda: {"__default__": 1.0})

    def records_for_actions(
        self,
        actions: Iterable[ActionName],
        beliefs: Iterable[Belief] | None = None,
    ) -> tuple[HistoryDurationRecord, ...]:
        """Return duration records for an action sequence. / 返回一串 action 的 duration 记录。"""
        belief_sequence = tuple(beliefs or ())
        progress = DurationProgress()
        records: list[HistoryDurationRecord] = []
        for index, action in enumerate(actions):
            belief = belief_sequence[index] if index < len(belief_sequence) else self.default_belief
            estimate = self.model.estimate(belief, action)
            progress = progress.add(estimate)
            records.append(
                HistoryDurationRecord(
                    action=action,
                    belief=belief,
                    estimate=estimate,
                    progress=progress,
                )
            )
        return tuple(records)

    def progress_for_actions(
        self,
        actions: Iterable[ActionName],
        beliefs: Iterable[Belief] | None = None,
    ) -> DurationProgress:
        """Return cumulative duration progress for actions. / 返回 action 序列的累计 duration progress。"""
        records = self.records_for_actions(actions, beliefs)
        return records[-1].progress if records else DurationProgress()

    def progress_for_history(
        self,
        history: History,
        beliefs: Iterable[Belief] | None = None,
    ) -> DurationProgress:
        """Return cumulative duration progress for a DARP history. / 返回 DARP history 的累计 duration progress。"""
        return self.progress_for_actions(history.actions, beliefs)

    def tau_for_history(self, history: History, beliefs: Iterable[Belief] | None = None) -> float:
        """Return tau for a DARP history. / 返回 DARP history 对应的 tau。"""
        return self.model.tau(self.progress_for_history(history, beliefs), self.horizon)

    def should_expand(self, history: History, beliefs: Iterable[Belief] | None = None) -> bool:
        """Return whether Phase 7 search should expand the history. / 返回 Phase 7 搜索是否应继续展开该 history。"""
        return self.model.should_continue(self.progress_for_history(history, beliefs), self.horizon, self.zeta)

    def elapsed_for_history(self, history: History, beliefs: Iterable[Belief] | None = None) -> float:
        """Return expected elapsed duration, like duration_model(q). / 返回期望累计时长，类似 duration_model(q)。"""
        return self.progress_for_history(history, beliefs).mean


@dataclass(frozen=True)
class FixedDurationModel(DurationModel):
    """Fixed action durations, where tau is remaining time. / 固定动作时长模型，tau 表示剩余时间。"""

    durations: Mapping[ActionName, float]
    default: float = 1.0
    kind: str = "fixed"

    def estimate(self, belief: Belief, action: ActionName) -> DurationEstimate:
        """Return the configured fixed duration. / 返回配置中的固定动作时长。"""
        return DurationEstimate(mean=float(self.durations.get(action, self.default)))

    def tau(self, progress: DurationProgress, horizon: float) -> float:
        """Return remaining time after accumulated duration. / 返回累计时长后的剩余时间。"""
        return horizon - progress.mean


@dataclass(frozen=True)
class StateDependentDurationModel(DurationModel):
    """Expected duration under the current belief. / 当前 belief 下的期望动作时长。"""

    durations: Mapping[tuple[StateKey, ActionName], float]
    default: float = 1.0
    kind: str = "expected"

    def estimate(self, belief: Belief, action: ActionName) -> DurationEstimate:
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

    means: Mapping[tuple[StateKey, ActionName], float]
    variances: Mapping[tuple[StateKey, ActionName], float]
    default_mean: float = 1.0
    default_variance: float = 0.0
    kind: str = "gaussian"

    def estimate(self, belief: Belief, action: ActionName) -> DurationEstimate:
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
