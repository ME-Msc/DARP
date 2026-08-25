"""Paper-style exact Expand operation over grounded finite kernels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from darp.adapter.exact import (
    ExactRDDLKernel,
    ObservationKey,
    RiskConstraintType,
    StateKey,
    risk_constraint_type_for_kernel,
)
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
    r"""Store Algorithm 2 constants with unambiguous constraint semantics.

    ``penalty`` is Lemma 3.2's ordinary-flow coefficient
    :math:`\rho(q)E[P(S,a)]`; ``chance_risk`` is Lemma 3.3's safe-flow
    first-entry coefficient. ``constraint_value`` selects the coefficient
    used by the active constraint.

    / ``penalty`` 和 ``chance_risk`` 分别对应 Lemma 3.2/3.3；
    ``constraint_value`` 返回当前约束实际使用的系数。
    """

    utility: float
    penalty: float
    chance_risk: float
    constraint_type: RiskConstraintType
    objective_support_preserved: bool
    utility_exact: Fraction | None
    penalty_exact: Fraction | None
    chance_risk_exact: Fraction | None

    @property
    def constraint_value(self) -> float:
        """Return the ILP coefficient selected by the model. / 返回当前模型的 ILP 系数。"""
        return self.chance_risk if self.constraint_type == "chance" else self.penalty


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
      DARP evaluates these values from pyRDDLGym grounded CPFs through
      `ExactRDDLKernel`; ordinary beliefs support smoothing, while safe
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

    / 显式实现论文 Algorithm 2：从 grounded CPF 精确枚举 transition 与
    observation；函数会计算 full-ILP 所需的 $$u_q$$、$$r_q$$、$$\rho(qao)$$、$$\tilde\rho(qao)$$ 与 $$\tau(qao)$$
    """
    exact_kernel = interface.exact_kernel
    if exact_kernel is None:
        raise ValueError("Paper Expand requires an exact kernel.")

    b_q = item.belief
    ordinary_mass_q = item.ordinary_mass
    constraint_mass_q = item.constraint_mass
    a_q = item.node.assignment
    if a_q is None:
        raise ValueError("AND-OR action node has no action assignment.")
    constraint_type = risk_constraint_type_for_kernel(exact_kernel)
    u_qa, objective_support_preserved, utility_exact = (
        exact_kernel.utility_coefficient_for_mass(ordinary_mass_q, a_q)
    )

    # Expected-cost mass is exactly the ordinary mass, so its expansion is
    # also the ordinary T/O expansion.  This avoids evaluating identical
    # transition and observation rows twice.  Chance constraints necessarily
    # retain distinct ordinary and survival-conditioned flows.
    if constraint_type == "expected":
        if dict(constraint_mass_q) != dict(ordinary_mass_q):
            raise ValueError("Expected-cost constraint mass must equal ordinary mass.")
        exact_constraint_qa = exact_kernel.expand_expected_constraint_mass(
            constraint_mass_q, a_q
        )
        ordinary_mass_qa = exact_constraint_qa
        p_qa = exact_constraint_qa.coefficient
        penalty_exact = exact_constraint_qa.coefficient_exact
        r_qa = 0.0
        chance_risk_exact = None
    else:
        ordinary_mass_qa = exact_kernel.expand_ordinary_mass(ordinary_mass_q, a_q)
        exact_constraint_qa = exact_kernel.expand_safe_constraint_mass(
            constraint_mass_q, a_q
        )
        p_qa = 0.0
        penalty_exact = None
        r_qa = exact_constraint_qa.coefficient
        chance_risk_exact = exact_constraint_qa.coefficient_exact

    constraint_outcomes = {
        outcome.observation: outcome for outcome in exact_constraint_qa.observations
    }

    # Lines 5-20: enumerate every qao branch and attach the next action frontier.
    # 第 5-20 行：枚举每个 $$qao$$ 分支，分别传播普通/安全概率、计算 smoothed belief 和 $$\tau(qao)$$。
    branches: list[ObservationFrontier] = []
    next_frontier: list[FrontierItem] = []
    for ordinary_outcome in ordinary_mass_qa.observations:
        observation = ordinary_outcome.observation
        ordinary_mass_qao = ordinary_outcome.state_mass
        b_qao = exact_kernel.constraint_mass_belief(ordinary_mass_qao)
        constraint_outcome = constraint_outcomes.get(observation)
        if constraint_type == "chance":
            constraint_mass_qao = (
                constraint_outcome.state_mass if constraint_outcome is not None else {}
            )
        else:
            if constraint_outcome is None:
                raise ValueError(
                    "Expected-cost expansion omitted an ordinary observation branch."
                )
            constraint_mass_qao = constraint_outcome.state_mass
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
            # Paper Sec. 3 chance-constrained duration: propagate the exact
            # posterior over augmented states (s, g), g being elapsed duration.
            # State marginals or a scalar expected duration cannot preserve the
            # correlation needed by Pr(G_q < h | q).
            duration_qao = _advance_augmented_duration_belief(
                exact_kernel=exact_kernel,
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
            exact_smoothed_beliefs_qao = _algorithm2_backward_and_smoothed_beliefs(
                exact_kernel=exact_kernel,
                actions=actions_qa,
                action_assignments=action_assignments_qa,
                observations=observation_keys_qao,
                filtered_masses=ordinary_mass_trace_qao,
            )
            duration_qao = _algorithm2_duration_from_smoothed_beliefs(
                actions=actions_qa,
                exact_smoothed_beliefs=exact_smoothed_beliefs_qao,
                duration_evaluator=duration_evaluator,
            )
        expand_qao = duration_evaluator.model.should_continue(
            duration_qao,
            duration_evaluator.horizon,
            duration_evaluator.zeta,
        )
        callback_belief_qao: Mapping[Any, Any] = _normalized_fraction_belief(
            ordinary_mass_qao
        )
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
        penalty=p_qa,
        chance_risk=r_qa,
        constraint_type=constraint_type,
        objective_support_preserved=objective_support_preserved,
        utility_exact=utility_exact,
        penalty_exact=penalty_exact,
        chance_risk_exact=chance_risk_exact,
    )
    return ExpandedAction(
        child_frontier=tuple(next_frontier),
        metrics=metrics,
        observation_frontiers=tuple(branches),
    )


def _algorithm2_backward_and_smoothed_beliefs(
    *,
    exact_kernel: ExactRDDLKernel,
    actions: Sequence[str],
    action_assignments: Sequence[Mapping[str, Any]],
    observations: Sequence[ObservationKey],
    filtered_masses: Sequence[Mapping[StateKey, Fraction]],
) -> tuple[Mapping[StateKey, Fraction], ...]:
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
            "Exact mass trace must contain b0 plus one mass per observation."
        )

    exact_messages: list[dict[StateKey, Fraction]] = [{} for _ in filtered_masses]
    exact_messages[-1] = {state: Fraction(1) for state in filtered_masses[-1]}
    for index in range(len(actions) - 1, -1, -1):
        exact_messages[index] = dict(
            exact_kernel.backward_fraction_message(
                filtered_masses[index],
                exact_messages[index + 1],
                action_assignments[index],
                observations[index],
            )
        )

    exact_smoothed: list[Mapping[StateKey, Fraction]] = []
    for index, filtered_mass in enumerate(filtered_masses):
        unnormalized = {
            state: Fraction(probability) * exact_messages[index].get(state, Fraction(0))
            for state, probability in filtered_mass.items()
            if Fraction(probability) > 0
        }
        total = sum(unnormalized.values(), start=Fraction(0))
        if total <= 0:
            raise ValueError(
                "Algorithm 2 exact smoothing produced zero probability for a qao branch."
            )
        exact_smoothed.append(
            {
                state: probability / total
                for state, probability in unnormalized.items()
                if probability > 0
            }
        )

    return tuple(exact_smoothed)


def _algorithm2_duration_from_smoothed_beliefs(
    *,
    actions: Sequence[str],
    duration_evaluator: HistoryDurationEvaluator,
    exact_smoothed_beliefs: Sequence[Mapping[StateKey, Fraction]],
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
            exact_smoothed_beliefs[index],
            action_label,
        )
        progress = progress.add(estimate_i)
    return progress


def _advance_augmented_duration_belief(
    *,
    exact_kernel: ExactRDDLKernel,
    model: ChanceConstrainedDurationModel,
    progress: DurationProgress,
    current_state_mass: Mapping[StateKey, Fraction],
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
            state: Fraction(probability)
            for state, probability in current_state_mass.items()
            if Fraction(probability) > 0
        }
        total = sum(state_weights.values(), start=Fraction(0))
        if total <= 0:
            raise ValueError(
                "Chance-duration expansion requires a non-empty current belief."
            )
        source = {
            (state, Fraction(0)): probability / total
            for state, probability in state_weights.items()
        }
    else:
        source = {
            (state, elapsed): Fraction(probability)
            for (state, elapsed), probability in progress.augmented_belief.items()
            if Fraction(probability) > 0
        }

    unnormalized: dict[tuple[StateKey, Fraction], Fraction] = {}
    for (state, elapsed), source_probability in source.items():
        duration = Fraction.from_float(model.duration_for_state(state, action_label))
        next_elapsed = elapsed + duration
        state_mapping = exact_kernel.state_from_key(state)
        transition_distribution = exact_kernel.transition_fraction_distribution(
            state_mapping, action_assignment
        )
        transition_weights = {
            next_state: (
                Fraction(probability)
                if isinstance(probability, Fraction)
                else Fraction.from_float(float(probability))
            )
            for next_state, probability in transition_distribution.items()
            if (
                Fraction(probability)
                if isinstance(probability, Fraction)
                else Fraction.from_float(float(probability))
            )
            > 0
        }
        transition_total = sum(transition_weights.values(), start=Fraction(0))
        if transition_total <= 0:
            continue
        for next_state, transition_weight in transition_weights.items():
            observation_probability = exact_kernel.observation_fraction_probability(
                observation, next_state, action_assignment
            )
            probability = (
                source_probability
                * transition_weight
                / transition_total
                * Fraction(observation_probability)
            )
            if probability > 0:
                key = (next_state, next_elapsed)
                unnormalized[key] = unnormalized.get(key, Fraction(0)) + probability

    normalizer = sum(unnormalized.values(), start=Fraction(0))
    if normalizer <= 0:
        raise ValueError(
            "Chance-duration augmented-state update has zero probability for "
            f"observation {observation!r}."
        )
    augmented_belief = {
        state_duration: probability / normalizer
        for state_duration, probability in unnormalized.items()
    }
    mean_exact = sum(
        (
            elapsed * probability
            for (_, elapsed), probability in augmented_belief.items()
        ),
        start=Fraction(0),
    )
    variance_exact = sum(
        (
            probability * (elapsed - mean_exact) ** 2
            for (_, elapsed), probability in augmented_belief.items()
        ),
        start=Fraction(0),
    )
    return DurationProgress(
        mean=mean_exact,
        variance=variance_exact,
        augmented_belief=augmented_belief,
    )


def _child_frontier(
    *,
    observation_node: ANDORNode,
    interface: ANDORSearchInterface,
    should_expand: bool,
    belief: Mapping[Any, float],
    action_belief: Mapping[Any, Any],
    ordinary_mass: Mapping[StateKey, Fraction],
    constraint_mass: Mapping[StateKey, Fraction],
    ordinary_mass_trace: tuple[Mapping[StateKey, Fraction], ...],
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


def _normalized_fraction_belief(
    mass: Mapping[StateKey, Fraction],
) -> dict[StateKey, Fraction]:
    """Normalize authoritative mass without losing subnormal support."""
    positive = {
        state: Fraction(probability)
        for state, probability in mass.items()
        if Fraction(probability) > 0
    }
    total = sum(positive.values(), start=Fraction(0))
    return (
        {state: probability / total for state, probability in positive.items()}
        if total > 0
        else {}
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
