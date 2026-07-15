"""Paper-style exact Expand operation over grounded finite kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from darp.adapter.exact import ExactRDDLKernel, ObservationKey, StateKey
from darp.model.and_or_tree import ANDORNode, ANDORSearchInterface
from darp.model.duration import Belief, DurationProgress, HistoryDurationEvaluator
from darp.planning.preprocess import FrontierItem


@dataclass(frozen=True)
class ExpansionMetrics:
    """Store paper metrics for one expanded history. / 保存一次 history 展开的论文指标。"""

    reward: float
    utility: float
    risk: float
    rho: float
    observation_probability: float
    tau: float
    zeta: float
    duration: DurationProgress
    observation_label: str
    state_label: str
    terminated: bool
    truncated: bool

    @property
    def done(self) -> bool:
        """Return whether the simulator stopped after this expansion. / 返回此次展开后 simulator 是否结束。"""
        return self.terminated or self.truncated

    @property
    def should_expand(self) -> bool:
        """Return whether children may still be expanded. / 返回是否还能继续展开子节点。"""
        return not self.done and self.tau > self.zeta


@dataclass(frozen=True)
class ExpandedAction:
    """Store one expanded action node and child frontiers. / 保存展开后的 action 节点和子 frontier。"""

    action_node: ANDORNode
    observation_node: ANDORNode
    child_frontier: tuple[FrontierItem, ...]
    metrics: ExpansionMetrics
    observation_frontiers: tuple["ObservationFrontier", ...] = ()


@dataclass(frozen=True)
class ObservationFrontier:
    """Store one qao observation branch and its child actions. / 保存一个 qao observation 分支及其子 action。"""

    observation_node: ANDORNode
    child_frontier: tuple[FrontierItem, ...]
    probability: float
    rho: float
    tau: float
    duration: DurationProgress
    should_expand: bool
    smoothing: "Algorithm2Smoothing"


@dataclass(frozen=True)
class Algorithm2Smoothing:
    r"""Store Algorithm 2 backward messages and smoothed beliefs.

    `filtered_beliefs[i]` is the forward belief
    $$\tilde b^i_{qao}(s_i)=Pr(s_i\mid q_{\le i})$$.
    `backward_messages[i]` is
    $$f_i(s_i)=Pr(q_{>i}\mid s_i)$$.
    `smoothed_beliefs[i]` is the paper's smoothed belief
    $$\bar b^i_{qao}(s_i)=Pr(s_i\mid qao)$$.

    / 保存 Algorithm 2 的 filtered belief、backward message 和 smoothed belief。
    """

    filtered_beliefs: tuple[Mapping[StateKey, float], ...]
    backward_messages: tuple[Mapping[StateKey, float], ...]
    smoothed_beliefs: tuple[Mapping[StateKey, float], ...]


# Paper Algorithm 2: Expand.
# 论文 Algorithm 2：Expand。
def expand_frontier_item(
    item: FrontierItem,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
) -> ExpandedAction:
    r"""Implement paper Algorithm 2 `Expand[qa, bq, ..., rho*(q), M']`.

    Line correspondence:

    - Lines 1-4 compute ordinary observation support and CC-POMDP constants:
        $$
            b_{qa}(s)=\sum_{s'}T(s',a,s)\tilde b_q(s'),\quad
            u_{qa}=\rho^*(q)\sum_s b^*_q(s)U(s,a),\quad
            r_{qa}=\rho^*(q)r(b_{qa}).
        $$
      DARP evaluates these values from pyRDDLGym grounded CPFs through
      `ExactRDDLKernel`; ordinary beliefs support smoothing, while safe
      beliefs support the chance constraint.

    - Lines 5-9 compute observation branches and their occurrence probability:

      $$\rho^*(qao)=\rho^*(q)(1-r(b_{qa}))Pr(o\mid q,a,\mathrm{safe})$$

      DARP enumerates all finite observation outcomes and posterior beliefs.

    - Lines 10-20 compute backward messages, smoothed beliefs, and the
      durative stopping value $$\tau(qao)$$.

    - Line 21 returns the ILP constants and child histories.

    / 显式实现论文 Algorithm 2：从 grounded CPF 精确枚举 transition 与
    observation；函数会计算 full-ILP 所需的 $$u_q$$、$$r_q$$、$$\rho^*(qao)$$、$$\tau(qao)$$
    """
    exact_kernel = interface.exact_kernel
    if exact_kernel is None or item.belief is None:
        raise ValueError("Paper Expand requires interface.exact_kernel and item.belief.")

    # Input symbols of $$Expand[qa, b_q, ..., rho^*(q), M']$$.
    # Expand 输入符号：observation history $$q$$ 的普通 belief、safe occurrence $$\rho^*(q)$$，以及 action history $$qa$$ 的动作 $$a_q$$。
    b_q = item.belief  # 以状态为键的普通 belief dict，表示历史 $$q$$ 中每个状态的概率；论文 C-POMDP 符号是 $$b_q$$。
    b_star_q = item.safe_belief or b_q  # safe belief，论文 CC-POMDP 符号是 $$b^*_q$$。
    rho_star_q = item.rho  # $$\rho^*(q)$$：安全前缀到达 action history q 的概率。
    a_q = _action_assignment(item.node.metadata)
    actions_qa = item.node.history.actions
    action_assignments_qa = _action_assignments_for_history(interface, actions_qa)
    filtered_beliefs_q = item.belief_trace or (b_q,)

    # Lines 1-4 for ordinary belief: enumerate T/O support for later smoothing.
    # 普通 belief 路径：枚举完整 transition/observation support，供 backward message 和 duration smoothing 使用。
    qa = exact_kernel.expand_action(b_q, a_q)
    b_qa = qa.prior_belief  # $$b_{qa}(s') = \sum_s T(s,a_q,s') b_q(s)$$，动作后、观测前的 prior belief。
    outcomes_qa = qa.observations  # 包含：所有可能 observation $$o$$，以及 $$Pr(o|qa)$$ 和 posterior $$b_{qao}$$。

    # Lemma 3.3 CC-POMDP constants:
    # $$u_q = \rho^*(q) \sum_s b^*_{q-1}(s) U(s,a_q)$$
    # $$r_q = \rho^*(q) r(b_q)$$, where $$b_q$$ is the safe prior after $$a_q$$.
    # CC-POMDP 常量：目标和风险都基于 safe occurrence $$\rho^*(q)$$ 与 safe belief $$b^*$$。
    safe_qa = exact_kernel.expand_safe_action(b_star_q, a_q)
    safe_outcomes_by_observation = {
        outcome.observation: outcome
        for outcome in safe_qa.observations
    }
    u_qa = rho_star_q * safe_qa.utility
    r_qa = rho_star_q * safe_qa.risk

    # Lines 5-20: enumerate every qao branch and attach the next action frontier.
    # 第 5-20 行：枚举每个 $$qao$$ 分支，计算 $$\rho^*(qao)$$、smoothed belief、$$\tau(qao)$$，并挂接下一层 action。
    branches: list[ObservationFrontier] = []
    next_frontier: list[FrontierItem] = []
    for outcome in outcomes_qa:
        p_o = outcome.probability  # $$Pr(o|qa)$$，在 $$b_{qa}$$ 下观测到 $$o$$ 的概率。
        safe_outcome = safe_outcomes_by_observation.get(outcome.observation)
        p_star_o = safe_outcome.probability if safe_outcome is not None else 0.0  # $$Pr(o|qa,safe)$$。
        b_star_qao = safe_outcome.belief if safe_outcome is not None else {}  # $$b^*_{qao}$$，safe posterior belief。
        rho_star_qao = (
            rho_star_q
            * safe_qa.survival_probability
            * p_star_o
        )  # $$\rho^*(qao)=rho^*(q)(1-r(b_{qa}))Pr(o|qa,safe)$$。
        b_qao = outcome.belief  # $$b_{qao}$$，观测 $$o$$ 后的 posterior belief。
        observation_keys_qao = item.observation_keys + (outcome.observation,)  # 完整观测序列 o_1..o_k。
        filtered_beliefs_qao = filtered_beliefs_q + (b_qao,)  # forward beliefs \tilde b^0..\tilde b^k。
        smoothing_qao = _algorithm2_backward_and_smoothed_beliefs(
            exact_kernel=exact_kernel,
            actions=actions_qa,
            action_assignments=action_assignments_qa,
            observations=observation_keys_qao,
            filtered_beliefs=filtered_beliefs_qao,
        )
        qao_node = interface.observation_node(item.node, outcome.label)
        item.node.add_child(qao_node)

        # Lines 10-20 after the backward messages: compute duration from
        # smoothed action-start beliefs.  For action a_i, D(S_i,a_i) uses
        # $$Pr(S_i | qao)$$, not just the forward belief before observing $$q_{>i}$$.
        # 第 10-20 行后半段：用 smoothed action-start belief 计算 duration；
        # 对动作 $$a_i$$，应使用 $$Pr(S_i | qao)$$，即已吸收未来观测信息后的 belief。
        duration_qao, duration_beliefs_qao = _algorithm2_duration_from_smoothed_beliefs(
            exact_kernel=exact_kernel,
            actions=actions_qa,
            smoothed_beliefs=smoothing_qao.smoothed_beliefs,
            duration_evaluator=duration_evaluator,
        )
        tau_qao = duration_evaluator.model.tau(duration_qao, duration_evaluator.horizon)
        expand_qao = duration_evaluator.model.should_continue(
            duration_qao,
            duration_evaluator.horizon,
            duration_evaluator.zeta,
        )
        child_actions = _child_frontier(
            item=item,
            observation_node=qao_node,
            interface=interface,
            rho=rho_star_qao,
            should_expand=expand_qao and rho_star_qao > 0.0 and bool(b_star_qao),
            belief=b_qao,
            safe_belief=b_star_qao,
            duration_beliefs=duration_beliefs_qao,
            belief_trace=filtered_beliefs_qao,
            observation_keys=observation_keys_qao,
        )
        branches.append(
            ObservationFrontier(
                observation_node=qao_node,
                child_frontier=child_actions,
                probability=p_o,
                rho=rho_star_qao,
                tau=tau_qao,
                duration=duration_qao,
                should_expand=expand_qao and rho_star_qao > 0.0 and bool(b_star_qao),
                smoothing=smoothing_qao,
            )
        )
        next_frontier.extend(child_actions)

    # ExpandedAction still has single-value diagnostic fields; use the first
    # branch only as a summary, while branches contains the actual qao set.
    # ExpandedAction 仍保留单值诊断字段；summary_branch 只用于展示，真正的 qao 集合在 branches 中。
    summary_branch = branches[0] if branches else None
    summary_node = (
        summary_branch.observation_node
        if summary_branch
        else interface.observation_node(item.node, "(none)")
    )
    if summary_branch:
        summary_tau = summary_branch.tau
        summary_duration = summary_branch.duration
    else:
        summary_duration = DurationProgress()
        summary_tau = duration_evaluator.model.tau(summary_duration, duration_evaluator.horizon)
    summary_label = str(summary_node.metadata.get("observation", "(none)"))
    metrics = ExpansionMetrics(
        reward=qa.utility,
        utility=u_qa,
        risk=r_qa,
        rho=rho_star_q,
        observation_probability=sum(outcome.probability for outcome in outcomes_qa),
        tau=summary_tau,
        zeta=duration_evaluator.zeta,
        duration=summary_duration,
        observation_label=summary_label,
        state_label="belief:" + repr(
            {exact_kernel.state_label(state): prob for state, prob in b_qa.items()}
        ),
        terminated=False,
        truncated=False,
    )
    return ExpandedAction(
        action_node=item.node,
        observation_node=summary_node,
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
    filtered_beliefs: Sequence[Mapping[StateKey, float]],
) -> Algorithm2Smoothing:
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
        raise ValueError("Action labels and action assignments must have the same length.")
    if len(actions) != len(observations):
        raise ValueError("A complete qao branch must have one observation per action.")
    if len(filtered_beliefs) != len(actions) + 1:
        raise ValueError("Filtered belief trace must contain b0 plus one belief per observation.")

    # Algorithm 2 line 10 starts at $$i=|qao|$$ with $$f_i(s)=1$$.
    # 论文第 10 行从末端开始：最后一步之后没有未来观测，所以 $$f_k(s)=1$$。
    messages: list[dict[StateKey, float]] = [{} for _ in filtered_beliefs]
    messages[-1] = {state: 1.0 for state in filtered_beliefs[-1]}

    # Algorithm 2 line 10: for $$i = |qao|-1$$ downto $$0$$.
    # 论文第 10 行：从后往前递推未来 action-observation 对当前 state 的 likelihood。
    for index in range(len(actions) - 1, -1, -1):
        action_i = action_assignments[index]
        observation_next = observations[index]
        next_message = messages[index + 1]
        if hasattr(exact_kernel, "backward_message"):
            messages[index] = dict(
                exact_kernel.backward_message(
                    filtered_beliefs[index],
                    next_message,
                    action_i,
                    observation_next,
                )
            )
        else:
            # Compatibility path for small test kernels. / 小型测试 kernel 的兼容路径。
            message_i: dict[StateKey, float] = {}
            for state_i in filtered_beliefs[index]:
                probability_of_future = 0.0
                for state_next, transition_prob in exact_kernel.transition_distribution(
                    exact_kernel.state_from_key(state_i),
                    action_i,
                ).items():
                    observation_prob = exact_kernel.observation_probability(
                        observation_next,
                        state_next,
                        action_i,
                    )
                    probability_of_future += (
                        next_message.get(state_next, 0.0)
                        * observation_prob
                        * transition_prob
                    )
                message_i[state_i] = probability_of_future
            messages[index] = message_i

    smoothed: list[Mapping[StateKey, float]] = []
    for index, filtered_belief_i in enumerate(filtered_beliefs):
        # Paper Bayes rule after line 10:
        # $$\bar b_i(s) = \alpha \tilde b_i(s) f_i(s)$$.
        # 第 10 行后由 Bayes 公式得到 smoothed belief：
        # 当前 filtered belief 乘以后向消息，再归一化。
        unnormalized = {
            state: probability * messages[index].get(state, 0.0)
            for state, probability in filtered_belief_i.items()
        }
        smoothed_i = _normalize_state_distribution(unnormalized)
        if not smoothed_i:
            raise ValueError(
                "Algorithm 2 smoothing produced zero probability for a qao branch; "
                "check observation likelihood support."
            )
        smoothed.append(smoothed_i)

    return Algorithm2Smoothing(
        filtered_beliefs=tuple(dict(belief) for belief in filtered_beliefs),
        backward_messages=tuple(messages),
        smoothed_beliefs=tuple(smoothed),
    )


def _algorithm2_duration_from_smoothed_beliefs(
    *,
    exact_kernel: ExactRDDLKernel,
    actions: Sequence[str],
    smoothed_beliefs: Sequence[Mapping[StateKey, float]],
    duration_evaluator: HistoryDurationEvaluator,
) -> tuple[DurationProgress, tuple[Belief, ...]]:
    r"""Compute fixed/stochastic duration formulas from smoothed beliefs.

    The paper's duration formulas use $$\bar b^i_{qao}(s)$$ for each
    action-start state. DARP sidecars define durations over grounded fluent
    names, so each exact state distribution is converted into fluent marginals
    before calling `DurationModel.estimate`.

    / 用 smoothed belief 计算 fixed/expected/Gaussian duration；sidecar 以
    grounded fluent 为键，因此先把 exact state belief 转为 fluent marginals。
    """

    progress = DurationProgress()
    duration_beliefs: list[Belief] = []
    for index, action_label in enumerate(actions):
        # Duration contribution for action $$a_i$$: 

        # fixed: $$\sum_s \bar b_i(s) c_{a_i}$$

        # stochastic: $$\sum_s \bar b_i(s) \mu_{s,a_i}$$, variance analogously.

        # 动作 $$a_i$$ 的持续时间贡献由 smoothed action-start belief $$\bar b_i$$ 加权得到

        belief_i = exact_kernel.fluent_belief(smoothed_beliefs[index])
        estimate_i = duration_evaluator.model.estimate(belief_i, action_label)
        progress = progress.add(estimate_i)
        duration_beliefs.append(belief_i)
    return progress, tuple(duration_beliefs)


def _child_frontier(
    *,
    item: FrontierItem,
    observation_node: ANDORNode,
    interface: ANDORSearchInterface,
    rho: float,
    should_expand: bool,
    belief: Mapping[Any, float] | None,
    safe_belief: Mapping[Any, float] | None,
    duration_beliefs: tuple[Mapping[Any, float], ...],
    belief_trace: tuple[Mapping[StateKey, float], ...],
    observation_keys: tuple[ObservationKey, ...],
) -> tuple[FrontierItem, ...]:
    """Create action children under one observation node. / 在 observation 节点下创建 action 子节点。"""
    if not should_expand:
        return ()
    action_nodes = interface.action_nodes(observation_node)
    for child in action_nodes:
        observation_node.add_child(child)
    return tuple(
        FrontierItem(
            node=child,
            rho=rho,
            root_action_label=item.root_label,
            belief=belief,
            safe_belief=safe_belief,
            duration_beliefs=duration_beliefs,
            belief_trace=belief_trace,
            observation_keys=observation_keys,
        )
        for child in action_nodes
    )


def _action_assignment(metadata: Mapping[str, object]) -> Mapping[str, Any]:
    """Return the pyRDDLGym action assignment stored on an action node. / 返回 action 节点上保存的 pyRDDLGym action 赋值。"""
    assignment = metadata.get("assignment")
    if not isinstance(assignment, Mapping):
        raise ValueError("AND-OR action node metadata must contain an action assignment.")
    return assignment


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


def _normalize_state_distribution(
    distribution: Mapping[StateKey, float],
) -> dict[StateKey, float]:
    """Normalize a state distribution. / 归一化 state 分布。"""
    cleaned = {
        state: float(probability)
        for state, probability in distribution.items()
        if abs(float(probability)) > 1e-15
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        return {}
    return {state: probability / total for state, probability in cleaned.items()}
