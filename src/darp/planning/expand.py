"""Paper Algorithm 2 over grounded finite transition kernels. / 基于有限 grounded 转移核实现论文 Algorithm 2。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from darp.adapter.kernel import ObservationKey, RDDLKernel, StateKey
from darp.model.and_or_tree import ANDORNode, ANDORSearchInterface
from darp.model.duration import (
    ChanceConstrainedDurationModel,
    DurationProgress,
    FixedDurationModel,
    HistoryDurationEvaluator,
)
from darp.planning.heuristic import UtilityHeuristic, history_heuristic_coefficient
from darp.planning.preprocess import FrontierItem


@dataclass(frozen=True)
class ExpansionMetrics:
    r"""Store utility/risk coefficients: actual values or frontier estimates.

    ``chance_risk`` is Lemma 3.3's safe-flow first-entry coefficient.
    ExpandedAction keeps actual values; HistoryRecord may hold F estimates.
    / 保存效用与风险系数；chance_risk 是 Lemma 3.3 的首次失败系数。
    ExpandedAction 保存实际值，HistoryRecord 中可以是 F 的估值。
    """

    utility: float  # u_q in E, h_u_q in F / 已展开效用或 frontier 估值。
    chance_risk: float  # r_q in E, h_r_q in F / 首次失败风险或 frontier 下界。


@dataclass(frozen=True)
class ExpandedAction:
    """Store one expanded action node and child frontiers. / 保存展开后的 action 节点和子 frontier。"""

    # Next action layer, reached through observations. / 经观测分支到达的下一层动作。
    child_frontier: tuple[FrontierItem, ...]
    metrics: ExpansionMetrics
    observation_frontiers: tuple[ObservationFrontier, ...] = ()


@dataclass(frozen=True)
class ObservationFrontier:
    """Store one qao observation branch and its child actions. / 保存一个 qao observation 分支及其子 action。"""

    observation: ObservationKey
    # Actions after this observation, not direct children of qa. / 此观测之后的动作，并非 qa 的直接子节点。
    child_frontier: tuple[FrontierItem, ...]
    should_expand: bool
    duration_stopped: bool


def apply_terminal_heuristic(
    item: FrontierItem,
    expanded: ExpandedAction,
    interface: ANDORSearchInterface,
    heuristic: UtilityHeuristic,
) -> ExpandedAction:
    """Replace exact utility only on branches stopped by the duration bound. / 仅在时长截止分支上替换为终端效用。"""
    terminal = tuple(
        branch for branch in expanded.observation_frontiers if branch.duration_stopped
    )
    if not terminal:
        return expanded
    action = item.node.assignment
    kernel = interface.kernel
    if action is None or kernel is None:
        raise ValueError("Terminal heuristic evaluation requires an action and kernel.")

    if (
        len(terminal) == len(expanded.observation_frontiers)
        and not getattr(kernel.grounded_model, "terminations", ())
    ):
        utility = history_heuristic_coefficient(
            heuristic,
            state_mass=item.ordinary_mass,
            action_label=item.action_label,
            action=action,
            non_fluents=kernel.non_fluents,
        )
    else:
        utility = expanded.metrics.utility
        for branch in terminal:
            branch_mass, branch_utility = (
                kernel.action_start_mass_and_utility_for_observation(
                    item.ordinary_mass,
                    action,
                    branch.observation,
                )
            )
            utility += history_heuristic_coefficient(
                heuristic,
                state_mass=branch_mass,
                action_label=item.action_label,
                action=action,
                non_fluents=kernel.non_fluents,
            )
            utility -= branch_utility
    return replace(
        expanded,
        metrics=replace(expanded.metrics, utility=utility),
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

    - Lines 1-4: compute u_qa from ordinary_mass and r_qa from safe_mass.
      Equations 9-11's ordinary product $$\rho^*(q)\tilde b^*_q(s)$$ is
      stored directly, with continuing-event restriction. For state-action
      reward, ``u_qa = sum_s ordinary_mass[s] * U(s,a)``; the kernel also
      handles successor-dependent RDDL rewards. Lemma 3.3's first-failure
      coefficient is ``r_qa = sum_s safe_mass[s] * sum_risky_s' T(s,a,s')``;
      safe_mass requires safety through q's current state.
    - Lines 5-9: propagate ordinary/safe mass through each observation.
      Retain continuing states without rescaling either mass; normalize
      ordinary mass to obtain b_qao. Safe propagation also removes failures.
    - Lines 10-20: update duration, using backward smoothing where needed,
      and decide whether the qao branch continues.
    - Line 21: return the ILP constants and the next action layer through qao.

    / 显式实现论文 Algorithm 2：从 grounded CPF 枚举 transition 与
    observation；普通质量保存历史概率乘后验，安全质量还要求直到当前
    状态均未失败；两者均保留继续执行权重，计算 $$u_{qa}$$、$$r_{qa}$$ 与 $$\tau(qao)$$。
    """
    kernel = interface.kernel
    if kernel is None:
        raise ValueError("Paper Expand requires a finite kernel.")

    # item.node is qa; its belief and masses still describe the preceding q.
    # / item.node 是 qa，belief 和概率质量仍属于此前观测历史 q。
    ordinary_mass_q = item.ordinary_mass
    safe_mass_q = item.safe_mass
    action = item.node.assignment
    if action is None:
        raise ValueError("AND-OR action node has no action assignment.")
    u_qa = kernel.utility_coefficient_for_mass(ordinary_mass_q, action)
    ordinary_qa = kernel.expand_ordinary_mass(ordinary_mass_q, action)
    safe_qa = kernel.expand_safe_mass(safe_mass_q, action)
    r_qa = safe_qa.coefficient

    safe_outcomes = {
        outcome.observation: outcome for outcome in safe_qa.observations
    }

    # Lines 5-20: enumerate every qao branch and attach the next action frontier.
    # 第 5-20 行：枚举每个 $$qao$$ 分支，分别传播普通/安全概率、计算 smoothed belief 和 $$\tau(qao)$$。
    branches: list[ObservationFrontier] = []
    next_frontier: list[FrontierItem] = []
    for ordinary_outcome in ordinary_qa.observations:
        observation = ordinary_outcome.observation
        qao_node = interface.observation_node(item.node, ordinary_outcome.label)
        ordinary_mass_qao = kernel.continuing_mass(ordinary_outcome.state_mass)
        if not ordinary_mass_qao:
            # The last transition's reward/risk was already counted above.
            # / 环境已结束：保留叶边和最后一步 reward/risk，不再要求累计 duration 达到 h。
            branches.append(ObservationFrontier(observation, (), False, False))
            continue
        b_qao = kernel.normalize_mass(ordinary_mass_qao)
        safe_outcome = safe_outcomes.get(observation)
        safe_mass_qao = kernel.continuing_mass(
            safe_outcome.state_mass if safe_outcome is not None else {}
        )
        observation_keys_qao = item.observation_keys + (
            observation,
        )  # Complete observation sequence / 完整观测序列 o_1..o_k。
        ordinary_mass_trace_qao = item.ordinary_mass_trace + (ordinary_mass_qao,)

        # Lines 10-20: duration and continuation for this qao branch.
        # / 第 10-20 行：计算此 qao 分支的时长及是否继续。
        duration_qao, expand_qao = _duration_continuation(
            item=item,
            interface=interface,
            kernel=kernel,
            duration_evaluator=duration_evaluator,
            action=action,
            observation=observation,
            ordinary_mass_qao=ordinary_mass_qao,
            ordinary_mass_trace_qao=ordinary_mass_trace_qao,
            observation_keys_qao=observation_keys_qao,
        )
        # Only live trajectories reach the next decision; their belief includes done=False.
        # / 下一次决策已知环境未结束；只归一化 belief，不归一化 history mass。
        child_actions = _child_frontier(
            observation_node=qao_node,
            interface=interface,
            should_expand=expand_qao,
            belief=b_qao,
            ordinary_mass=ordinary_mass_qao,
            safe_mass=safe_mass_qao,
            ordinary_mass_trace=ordinary_mass_trace_qao,
            observation_keys=observation_keys_qao,
            duration_progress=duration_qao,
        )
        branches.append(
            ObservationFrontier(
                observation=observation,
                child_frontier=child_actions,
                should_expand=expand_qao,
                # Model-terminal-only outcomes were retained as leaves above.
                # 仅含模型终止状态的观测结果已在上面保留为叶节点。
                duration_stopped=not expand_qao,
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


def _duration_continuation(
    *,
    item: FrontierItem,
    interface: ANDORSearchInterface,
    kernel: RDDLKernel,
    duration_evaluator: HistoryDurationEvaluator,
    action: Mapping[str, Any],
    observation: ObservationKey,
    ordinary_mass_qao: Mapping[StateKey, float],
    ordinary_mass_trace_qao: tuple[Mapping[StateKey, float], ...],
    observation_keys_qao: tuple[ObservationKey, ...],
) -> tuple[DurationProgress, bool]:
    """Compute duration and continuation after observing qao.

    Fixed duration carries an O(1) sum; chance duration carries the joint
    state/elapsed-time posterior; other models use smoothed action-start
    beliefs Pr(S_i | qao).

    / 固定时长按 O(1) 累加；机会约束时长传播状态与累计时长的联合后验；
    其他模型用吸收后续观测的动作起始平滑信念 Pr(S_i | qao)。
    """
    if isinstance(duration_evaluator.model, FixedDurationModel):
        duration_qao = item.duration_progress.add(
            duration_evaluator.model.estimate(
                item.belief,
                item.action_label,
            )
        )
    elif isinstance(duration_evaluator.model, ChanceConstrainedDurationModel):
        # Preserve state/time correlation for Pr(G_q < h | q).
        # / 保留 Pr(G_q < h | q) 所需的状态与累计时长相关性。
        duration_qao = _advance_augmented_duration_belief(
            kernel=kernel,
            model=duration_evaluator.model,
            progress=item.duration_progress,
            current_state_mass=item.ordinary_mass,
            action_label=item.action_label,
            action_assignment=action,
            observation=observation,
            next_state_support=ordinary_mass_qao,
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
    return duration_qao, expand_qao


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
    *complete* action-start state. Converting that joint distribution into
    separate fluent marginals would lose probability mass for all-false states
    and double-count states with multiple true fluents.

    / 用完整状态的 smoothed belief 计算 expected/Gaussian duration，不能把
    joint state 错当成互相独立的 fluent 边缘概率。
    """

    progress = DurationProgress()
    for index, action_label in enumerate(actions):
        # Action a_i duration: fixed sum_s b_i(s)c_ai; stochastic mean
        # sum_s b_i(s)mu_s,ai, with variance computed by the duration model.
        # 动作 a_i 的时长用平滑动作起始 belief 加权；方差由 duration 模型计算。
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
    next_state_support: Mapping[StateKey, float],
) -> DurationProgress:
    r"""Apply the paper's deterministic chance-duration state augmentation.

    For each source augmented state :math:`(s,g)`, this computes

    .. math::

       T'((s,g),a,(s',g')) = T(s,a,s')
       \quad\text{when }g'=g+D(s,a),

    multiplies by :math:`O(o\mid s',a)`, and normalizes on the observed
    history and ``done=False``, using ``next_state_support`` from the ordinary
    live posterior. This keeps the duration distribution consistent with the
    episode's continuation, without renormalizing each transition row.

    / 将状态增广为 (s,g)，按 g'=g+D(s,a) 传播，再按观测和未终止条件归一化；
    保留累计时长分布，不对每条转移行单独归一化。
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
            if next_state not in next_state_support:
                continue
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
    ordinary_mass: Mapping[StateKey, float],
    safe_mass: Mapping[StateKey, float],
    ordinary_mass_trace: tuple[Mapping[StateKey, float], ...],
    observation_keys: tuple[ObservationKey, ...],
    duration_progress: DurationProgress,
) -> tuple[FrontierItem, ...]:
    """Create action children under one observation node. / 在 observation 节点下创建 action 子节点。"""
    if not should_expand:
        return ()
    action_nodes = interface.action_nodes(observation_node, belief=belief)
    return tuple(
        FrontierItem(
            node=child,
            belief=belief,
            ordinary_mass=ordinary_mass,
            safe_mass=safe_mass,
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
    """Reuse read-only action assignments in history order. / 按 history 标签顺序复用只读 action assignment。"""
    by_label = {choice.label: choice.assignment for choice in interface.actions}
    assignments: list[Mapping[str, Any]] = []
    for label in action_labels:
        if label not in by_label:
            raise ValueError(f"History references unknown action label: {label}")
        assignments.append(by_label[label])
    return tuple(assignments)
