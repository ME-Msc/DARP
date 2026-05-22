"""Paper-style Expand operation over pyRDDLGym runtime copies."""

# TODO(phase-9.1): Replace deterministic rho propagation with explicit
# observation-probability calculation when the pyRDDLGym sampler exposes it.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from darp.adapter.runtime import _json_ready, state_label
from darp.model.and_or_tree import ANDORNode, ANDORSearchInterface
from darp.model.duration import DurationProgress, HistoryDurationEvaluator
from darp.planning.preprocess import FrontierItem


@dataclass(frozen=True)
class ExpansionMetrics:
    """Store paper metrics for one expanded history. / 保存一次 history 展开的论文指标。"""

    reward: float
    utility: float
    risk: float
    rho: float
    tau: float
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
        return not self.done and self.tau > 0.0


@dataclass(frozen=True)
class ExpandedAction:
    """Store one expanded action node and generated children. / 保存展开后的 action 节点和生成的子节点。"""

    action_node: ANDORNode
    observation_node: ANDORNode
    child_frontier: tuple[FrontierItem, ...]
    metrics: ExpansionMetrics


def expand_frontier_item(
    item: FrontierItem,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
) -> ExpandedAction:
    r"""Expand one action-history node with a pyRDDLGym generative step.

    Paper correspondence:

    - This is the DARP runtime analogue of the paper's `Expand` routine.
    - For a history :math:`q`, we apply action :math:`a_q`, observe
      :math:`o_q`, update the child history, and compute:

      .. math::

         \rho(q) \leftarrow \rho(parent) P(o_q \mid q, a_q)
         \qquad
         u_q \leftarrow \rho(q) R(q, a_q)
         \qquad
         r_q \leftarrow \rho(q) C(q, a_q)
         \qquad
         \tau(q) \leftarrow D.\tau(D(q), H)

    - The current pyRDDLGym adapter is generative, not enumerative, so the
      sampled observation branch keeps :math:`P(o_q \mid q,a_q)=1`.
    - DARP has no grounded cost/risk fluent yet, so :math:`r_q=0`; Phase 9
      benchmark work should add explicit constrained-cost extraction.

    / 展开一个 action-history：执行 action、得到 observation/reward，并计算
    :math:`\rho(q)`、:math:`u_q`、:math:`r_q`、:math:`\tau(q)` 的当前原型值。
    """

    action = _action_assignment(item.node.metadata)
    runtime = item.parent_runtime.clone()
    observation, reward, terminated, truncated, _ = runtime.step(action)
    obs_label = observation_label(observation, runtime.state, interface)
    obs_node = interface.observation_node(item.node, obs_label)
    _add_child_once(item.node, obs_node)

    progress = duration_evaluator.progress_for_history(item.node.history)
    tau = duration_evaluator.tau_for_history(item.node.history)
    metrics = ExpansionMetrics(
        reward=reward,
        utility=item.rho * reward,
        risk=0.0,
        rho=item.rho,
        tau=tau,
        duration=progress,
        observation_label=obs_label,
        state_label=state_label(runtime.state),
        terminated=terminated,
        truncated=truncated,
    )
    child_frontier = _child_frontier(
        item=item,
        runtime=runtime,
        observation_node=obs_node,
        interface=interface,
        should_expand=metrics.should_expand,
    )
    return ExpandedAction(
        action_node=item.node,
        observation_node=obs_node,
        child_frontier=child_frontier,
        metrics=metrics,
    )


def observation_label(
    observation: Mapping[str, Any],
    state: Mapping[str, Any],
    interface: ANDORSearchInterface,
) -> str:
    """Return the label used for an observation branch. / 返回 observation 分支使用的标签。"""
    if interface.observation_scope.mode == "mdp-state":
        return state_label(state)
    return repr(_json_ready(dict(observation)))


def _child_frontier(
    *,
    item: FrontierItem,
    runtime: Any,
    observation_node: ANDORNode,
    interface: ANDORSearchInterface,
    should_expand: bool,
) -> tuple[FrontierItem, ...]:
    """Create action children under one observation node. / 在 observation 节点下创建 action 子节点。"""
    if not should_expand:
        return ()
    action_nodes = interface.action_nodes(observation_node)
    for child in action_nodes:
        _add_child_once(observation_node, child)
    return tuple(
        FrontierItem(
            node=child,
            parent_runtime=runtime.clone(),
            rho=item.rho,
            root_action_label=item.root_label,
        )
        for child in action_nodes
    )


def _action_assignment(metadata: Mapping[str, object]) -> Mapping[str, Any]:
    """Return the pyRDDLGym action assignment stored on an action node. / 返回 action 节点上保存的 pyRDDLGym action 赋值。"""
    assignment = metadata.get("assignment")
    if not isinstance(assignment, Mapping):
        raise ValueError("AND-OR action node metadata must contain an action assignment.")
    return assignment


def _add_child_once(parent: ANDORNode, child: ANDORNode) -> None:
    """Attach a child only when its node id is not already present. / 仅在 node id 尚不存在时挂接子节点。"""
    if all(existing.node_id != child.node_id for existing in parent.children):
        parent.add_child(child)
