"""Paper Algorithm 2 over grounded finite transition kernels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from darp.adapter.kernel import ObservationKey, RDDLKernel, StateKey
from darp.model.and_or_tree import ANDORNode, ANDORSearchInterface
from darp.model.duration import (
    ChanceConstrainedDurationModel,
    DurationProgress,
    FixedDurationModel,
    HistoryDurationEvaluator,
)
from darp.planning.preprocess import FrontierItem


@dataclass(frozen=True)
class ExpansionMetrics:
    r"""Store Algorithm 2 utility and CC-POMDP risk constants.

    ``chance_risk`` is Lemma 3.3's safe-flow first-entry coefficient.
    / ``chance_risk`` 是 Lemma 3.3 的安全概率流首次进入风险集系数。
    """

    utility: float
    chance_risk: float


@dataclass(frozen=True)
class ExpandedAction:
    """Store one expanded action node and child frontiers. / 保存展开后的 action 节点和子 frontier。"""

    child_frontier: tuple[FrontierItem, ...]
    metrics: ExpansionMetrics
    observation_frontiers: tuple[ObservationFrontier, ...] = ()


@dataclass(frozen=True)
class ObservationFrontier:
    """Store one qao observation branch and its child actions. / 保存一个 qao observation 分支及其子 action。"""

    child_frontier: tuple[FrontierItem, ...]
    should_expand: bool


def evaluate_frontier_leaf_metrics(
    item: FrontierItem,
    interface: ANDORSearchInterface,
    *,
    utility: float,
) -> ExpansionMetrics:
    """Combine a leaf utility with its risk without materializing children.

    The caller supplies the frontier utility (normally :math:`h_q`); this
    function computes only the active constraint coefficient. Observation
    posteriors, duration branches, AND-OR nodes, one-step utility and child
    actions are deliberately postponed until the incumbent selects the leaf.

    / 调用方提供 frontier utility（通常为 :math:`h_q`）；这里只计算
    risk，observation、duration、一步 utility 和 children 均延迟。
    """
    kernel = interface.kernel
    if kernel is None:
        raise ValueError("Paper frontier evaluation requires a finite kernel.")
    action = item.node.assignment
    if action is None:
        raise ValueError("AND-OR action node has no action assignment.")

    chance_risk = kernel.safe_constraint_coefficient_for_mass(
        item.constraint_mass,
        action,
    )

    return ExpansionMetrics(
        utility=utility,
        chance_risk=chance_risk,
    )


# Paper Algorithm 2: Expand.
# 论文 Algorithm 2：Expand。
def expand_frontier_item(
    item: FrontierItem,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
) -> ExpandedAction:
    r"""Implement paper Algorithm 2 with ordinary and safe-prefix flows.

    Line correspondence:

    - Lines 1-4 compute ordinary observation support and CC-POMDP constants,
      keeping the ordinary and safe-conditioned probability flows distinct:
        $$
            b_{qa}(s')=\sum_sT(s,a,s')b_q(s),\quad
            u_{qa}=\rho(q)\,\mathbb E_{b_q}[U(s,a)],\quad
            r_{qa}=\tilde\rho(q)\,r(b^{\mathrm{safe}}_q,a).
        $$
      DARP evaluates these values from pyRDDLGym grounded CPFs through its
      finite kernel; ordinary beliefs support smoothing, while safe
      beliefs support the chance constraint.

    - Lines 5-9 compute observation branches and their occurrence probability:

      $$\rho(qao)=\rho(q)Pr(o\mid q,a)$$

      for utility, and

      $$\tilde\rho(qao)=\tilde\rho(q)(1-r(b^{\mathrm{safe}}_q,a))
        Pr(o\mid q,a,\mathrm{safe})$$

      for chance risk.

      DARP enumerates all finite observation outcomes and posterior beliefs.

    - Lines 10-20 compute backward messages, smoothed beliefs, and the
      durative stopping value $$\tau(qao)$$.

    - Line 21 returns the ILP constants and child histories.

    / 显式实现论文 Algorithm 2：从 grounded CPF 枚举 transition 与
    observation；函数会计算 full-ILP 所需的 $$u_q$$、$$r_q$$、$$\rho(qao)$$、$$\tilde\rho(qao)$$ 与 $$\tau(qao)$$
    """
    kernel = interface.kernel
    if kernel is None:
        raise ValueError("Paper Expand requires a finite kernel.")

    b_q = item.belief
    ordinary_mass_q = item.ordinary_mass
    constraint_mass_q = item.constraint_mass
    a_q = item.node.assignment
    if a_q is None:
        raise ValueError("AND-OR action node has no action assignment.")
    u_qa = kernel.utility_coefficient_for_mass(ordinary_mass_q, a_q)
    ordinary_mass_qa = kernel.expand_ordinary_mass(ordinary_mass_q, a_q)
    constraint_qa = kernel.expand_safe_constraint_mass(
        constraint_mass_q, a_q
    )
    r_qa = constraint_qa.coefficient

    constraint_outcomes = {
        outcome.observation: outcome for outcome in constraint_qa.observations
    }

    # Lines 5-20: enumerate every qao branch and attach the next action frontier.
    # 第 5-20 行：枚举每个 $$qao$$ 分支，分别传播普通/安全概率、计算 smoothed belief 和 $$\tau(qao)$$。
    branches: list[ObservationFrontier] = []
    next_frontier: list[FrontierItem] = []
    for ordinary_outcome in ordinary_mass_qa.observations:
        observation = ordinary_outcome.observation
        ordinary_mass_qao = ordinary_outcome.state_mass
        b_qao = kernel.constraint_mass_belief(ordinary_mass_qao)
        constraint_outcome = constraint_outcomes.get(observation)
        constraint_mass_qao = (
            constraint_outcome.state_mass if constraint_outcome is not None else {}
        )
        observation_keys_qao = item.observation_keys + (
            observation,
        )  # 完整观测序列 o_1..o_k。
        ordinary_mass_trace_qao = item.ordinary_mass_trace + (ordinary_mass_qao,)
        qao_node = interface.observation_node(item.node, ordinary_outcome.label)

        # Lines 10-20 after the backward messages: compute duration from
        # smoothed action-start beliefs.  For action a_i, D(S_i,a_i) uses
        # $$Pr(S_i | qao)$$, not just the forward belief before observing $$q_{>i}$$.
        # 第 10-20 行后半段：用 smoothed action-start belief 计算 duration；
        # 对动作 $$a_i$$，应使用 $$Pr(S_i | qao)$$，即已吸收未来观测信息后的 belief。
        if isinstance(duration_evaluator.model, FixedDurationModel):
            # Fixed duration is history-independent, so Algorithm 2's
            # backward smoothing cannot change tau. Carry one sufficient
            # statistic instead of recomputing the whole history per outcome.
            # 固定时长只需 O(1) 累加，无需为每个 observation 重跑整段 backward smoothing。
            duration_qao = item.duration_progress.add(
                duration_evaluator.model.estimate(
                    b_q,
                    item.action_label,
                )
            )
        elif isinstance(duration_evaluator.model, ChanceConstrainedDurationModel):
            # Paper Sec. 3 chance-constrained duration: propagate the
            # posterior over augmented states (s, g), g being elapsed duration.
            # State marginals or a scalar expected duration cannot preserve the
            # correlation needed by Pr(G_q < h | q).
            duration_qao = _advance_augmented_duration_belief(
                kernel=kernel,
                model=duration_evaluator.model,
                progress=item.duration_progress,
                current_state_mass=ordinary_mass_q,
                action_label=item.action_label,
                action_assignment=a_q,
                observation=observation,
            )
        else:
            actions_qa = item.node.history.actions
            action_assignments_qa = _action_assignments_for_history(
                interface, actions_qa
            )
            smoothed_beliefs_qao = _algorithm2_backward_and_smoothed_beliefs(
                kernel=kernel,
                actions=actions_qa,
                action_assignments=action_assignments_qa,
                observations=observation_keys_qao,
                filtered_masses=ordinary_mass_trace_qao,
            )
            duration_qao = _algorithm2_duration_from_smoothed_beliefs(
                actions=actions_qa,
                smoothed_beliefs=smoothed_beliefs_qao,
                duration_evaluator=duration_evaluator,
            )
        expand_qao = duration_evaluator.model.should_continue(
            duration_qao,
            duration_evaluator.horizon,
            duration_evaluator.zeta,
        )
        # Reuse the normalized posterior for terminal and action callbacks.
        callback_belief_qao: Mapping[Any, Any] = b_qao
        should_expand_qao = (
            expand_qao
            and bool(ordinary_mass_qao)
            and not interface.belief_is_terminal(callback_belief_qao)
        )
        child_actions = _child_frontier(
            observation_node=qao_node,
            interface=interface,
            should_expand=should_expand_qao,
            belief=b_qao,
            action_belief=callback_belief_qao,
            ordinary_mass=ordinary_mass_qao,
            constraint_mass=constraint_mass_qao,
            ordinary_mass_trace=ordinary_mass_trace_qao,
            observation_keys=observation_keys_qao,
            duration_progress=duration_qao,
        )
        branches.append(
            ObservationFrontier(
                child_frontier=child_actions,
                should_expand=should_expand_qao,
            )
        )
        next_frontier.extend(child_actions)

    metrics = ExpansionMetrics(
        utility=u_qa,
        chance_risk=r_qa,
    )
    return ExpandedAction(
        child_frontier=tuple(next_frontier),
        metrics=metrics,
        observation_frontiers=tuple(branches),
    )


def _algorithm2_backward_and_smoothed_beliefs(
    *,
    kernel: RDDLKernel,
    actions: Sequence[str],
    action_assignments: Sequence[Mapping[str, Any]],
    observations: Sequence[ObservationKey],
    filtered_masses: Sequence[Mapping[StateKey, float]],
) -> tuple[Mapping[StateKey, float], ...]:
    r"""Compute Algorithm 2 backward messages and smoothed beliefs.

    For a concrete branch $$qao = (a_1,o_1,\ldots,a_k,o_k)$$,
    Algorithm 2 line 10 iterates backward:

    $$
       f_k(s_k)=1,\qquad
       f_i(s_i)=\sum_{s_{i+1}} f_{i+1}(s_{i+1})
          O(o_{i+1},s_{i+1},a_{i+1})
          T(s_i,a_{i+1},s_{i+1}).
    $$

    Then the smoothed belief used by duration formulas is:

    $$
       \bar b^i_{qao}(s_i)
       = \alpha_i\,\tilde b^i_{qao}(s_i) f_i(s_i).
    $$

    / 真实实现论文 Algorithm 2 第 10 行的 backward message，并用它计算
    smoothed belief，而不是只做 forward belief 累计。
    """

    if len(actions) != len(action_assignments):
        raise ValueError(
            "Action labels and action assignments must have the same length."
        )
    if len(actions) != len(observations):
        raise ValueError("A complete qao branch must have one observation per action.")
    if len(filtered_masses) != len(actions) + 1:
        raise ValueError(
            "Mass trace must contain b0 plus one mass per observation."
        )

    messages: list[dict[StateKey, float]] = [{} for _ in filtered_masses]
    messages[-1] = {state: 1.0 for state in filtered_masses[-1]}
    for index in range(len(actions) - 1, -1, -1):
        messages[index] = dict(
            kernel.backward_message(
                filtered_masses[index],
                messages[index + 1],
                action_assignments[index],
                observations[index],
            )
        )

    smoothed: list[Mapping[StateKey, float]] = []
    for index, filtered_mass in enumerate(filtered_masses):
        unnormalized = {
            state: float(probability) * messages[index].get(state, 0.0)
            for state, probability in filtered_mass.items()
            if float(probability) > 0.0
        }
        total = sum(unnormalized.values())
        if total <= 0:
            raise ValueError(
                "Algorithm 2 smoothing produced zero probability for a qao branch."
            )
        smoothed.append(
            {
                state: probability / total
                for state, probability in unnormalized.items()
                if probability > 0.0
            }
        )

    return tuple(smoothed)


def _algorithm2_duration_from_smoothed_beliefs(
    *,
    actions: Sequence[str],
    duration_evaluator: HistoryDurationEvaluator,
    smoothed_beliefs: Sequence[Mapping[StateKey, float]],
) -> DurationProgress:
    r"""Compute fixed/stochastic duration formulas from smoothed beliefs.

    The paper's duration formulas use $$\bar b^i_{qao}(s)$$ for each
    *complete* action-start state.  Sidecar fluent names are state selectors,
    not independent probability atoms; converting the joint distribution into
    fluent marginals would lose probability mass for all-false states and
    double-count states with multiple true fluents.

    / 用完整状态的 smoothed belief 计算 expected/Gaussian duration；
    sidecar 中的 fluent 名是状态选择器，不是独立概率原子。
    """

    progress = DurationProgress()
    for index, action_label in enumerate(actions):
        # Duration contribution for action $$a_i$$:

        # fixed: $$\sum_s \bar b_i(s) c_{a_i}$$

        # stochastic: $$\sum_s \bar b_i(s) \mu_{s,a_i}$$, variance analogously.

        # 动作 $$a_i$$ 的持续时间贡献由 smoothed action-start belief $$\bar b_i$$ 加权得到

        estimate_i = duration_evaluator.model.estimate(
            smoothed_beliefs[index],
            action_label,
        )
        progress = progress.add(estimate_i)
    return progress


def _advance_augmented_duration_belief(
    *,
    kernel: RDDLKernel,
    model: ChanceConstrainedDurationModel,
    progress: DurationProgress,
    current_state_mass: Mapping[StateKey, float],
    action_label: str,
    action_assignment: Mapping[str, Any],
    observation: ObservationKey,
) -> DurationProgress:
    r"""Apply the paper's deterministic chance-duration state augmentation.

    For each source augmented state :math:`(s,g)`, this computes

    .. math::

       T'((s,g),a,(s',g')) = T(s,a,s')
       \quad\text{when }g'=g+D(s,a),

    multiplies by :math:`O(o\mid s',a)`, and normalizes on the observed
    history.  The returned distribution is therefore
    :math:`Pr(S_{qao},G_{qao}\mid qao)` and directly supports
    :math:`\tau(qao)=Pr(G_{qao}<h\mid qao)`.
    """
    if progress.augmented_belief is None:
        state_weights = {
            state: float(probability)
            for state, probability in current_state_mass.items()
            if float(probability) > 0.0
        }
        total = sum(state_weights.values())
        if total <= 0:
            raise ValueError(
                "Chance-duration expansion requires a non-empty current belief."
            )
        source = {
            (state, 0.0): probability / total
            for state, probability in state_weights.items()
        }
    else:
        source = {
            (state, float(elapsed)): float(probability)
            for (state, elapsed), probability in progress.augmented_belief.items()
            if float(probability) > 0.0
        }

    unnormalized: dict[tuple[StateKey, float], float] = {}
    for (state, elapsed), source_probability in source.items():
        duration = model.duration_for_state(state, action_label)
        next_elapsed = elapsed + duration
        state_mapping = kernel.state_from_key(state)
        transition_distribution = kernel.transition_distribution(
            state_mapping, action_assignment
        )
        transition_weights = {
            next_state: float(probability)
            for next_state, probability in transition_distribution.items()
            if float(probability) > 0.0
        }
        transition_total = sum(transition_weights.values())
        if transition_total <= 0:
            continue
        for next_state, transition_weight in transition_weights.items():
            observation_probability = kernel.observation_probability(
                observation, next_state, action_assignment
            )
            probability = (
                source_probability
                * transition_weight
                / transition_total
                * observation_probability
            )
            if probability > 0:
                key = (next_state, next_elapsed)
                unnormalized[key] = unnormalized.get(key, 0.0) + probability

    normalizer = sum(unnormalized.values())
    if normalizer <= 0:
        raise ValueError(
            "Chance-duration augmented-state update has zero probability for "
            f"observation {observation!r}."
        )
    augmented_belief = {
        state_duration: probability / normalizer
        for state_duration, probability in unnormalized.items()
    }
    mean = sum(
        elapsed * probability
        for (_, elapsed), probability in augmented_belief.items()
    )
    variance = sum(
        probability * (elapsed - mean) ** 2
        for (_, elapsed), probability in augmented_belief.items()
    )
    return DurationProgress(
        mean=mean,
        variance=variance,
        augmented_belief=augmented_belief,
    )


def _child_frontier(
    *,
    observation_node: ANDORNode,
    interface: ANDORSearchInterface,
    should_expand: bool,
    belief: Mapping[Any, float],
    action_belief: Mapping[Any, Any],
    ordinary_mass: Mapping[StateKey, float],
    constraint_mass: Mapping[StateKey, float],
    ordinary_mass_trace: tuple[Mapping[StateKey, float], ...],
    observation_keys: tuple[ObservationKey, ...],
    duration_progress: DurationProgress,
) -> tuple[FrontierItem, ...]:
    """Create action children under one observation node. / 在 observation 节点下创建 action 子节点。"""
    if not should_expand:
        return ()
    action_nodes = interface.action_nodes(observation_node, belief=action_belief)
    return tuple(
        FrontierItem(
            node=child,
            belief=belief,
            ordinary_mass=ordinary_mass,
            constraint_mass=constraint_mass,
            ordinary_mass_trace=ordinary_mass_trace,
            observation_keys=observation_keys,
            duration_progress=duration_progress,
        )
        for child in action_nodes
    )


def _action_assignments_for_history(
    interface: ANDORSearchInterface,
    action_labels: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    """Return action assignments aligned with history labels. / 返回与 history action 标签对齐的 action assignment。"""
    by_label = {choice.label: dict(choice.assignment) for choice in interface.actions}
    assignments: list[Mapping[str, Any]] = []
    for label in action_labels:
        if label not in by_label:
            raise ValueError(f"History references unknown action label: {label}")
        assignments.append(by_label[label])
    return tuple(assignments)
