"""Root-frontier initialization helpers for paper Algorithm 1."""

# TODO(phase-9.1): Add benchmark trace hooks for frontier scores and ILP
# variable ids.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from darp.adapter.runtime import PyRDDLGymRuntime
from darp.model.and_or_tree import ANDORNode, ANDORSearchInterface
from darp.adapter.exact import ObservationKey, StateKey


@dataclass(frozen=True, eq=False)
class FrontierItem:
    """Track one expandable action history and its numeric beliefs. / 跟踪一个可展开动作历史及其数值 belief。"""

    node: ANDORNode
    rho: float = 1.0
    root_action_label: str | None = None
    belief: Mapping[StateKey, float] | None = None
    safe_belief: Mapping[StateKey, float] | None = None
    duration_beliefs: tuple[Mapping[StateKey, float], ...] = ()
    belief_trace: tuple[Mapping[StateKey, float], ...] = ()
    observation_keys: tuple[ObservationKey, ...] = ()

    @property
    def action_label(self) -> str:
        """Return the node action label. / 返回该节点的 action 标签。"""
        return str(self.node.metadata.get("action", "noop"))

    @property
    def root_label(self) -> str:
        """Return the first action on this branch. / 返回该分支上的第一个 action。"""
        return self.root_action_label or self.action_label


@dataclass(frozen=True)
class RootFrontier:
    r"""Store the initialized root frontier for Algorithm 1. / 保存 Algorithm 1 的 root frontier 初始化结果。

    Paper symbols:

    - ``root`` is the empty observation history `0`.
    - ``open_histories`` starts as $$N=\{0\}$$ .
    - ``frontier`` contains the action histories $$qa$$ .

    / 这里仅完成 Algorithm 1 的 root 初始化；完整 preprocessing 主循环在
    `planning.ilp_tree.paper_preprocess` 中执行。
    """

    root: ANDORNode
    frontier: tuple[FrontierItem, ...]
    open_histories: tuple[ANDORNode, ...] = field(default_factory=tuple)


def initialize_root_frontier(
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
    *,
    root_belief: Mapping[StateKey, float] | None = None,
) -> RootFrontier:
    r"""Implement paper Algorithm 1 line-1 initialization and root expansion.

    Line correspondence:

    - Line 1 initializes $$G$$, $$N=\{0\}$$, $$F=\emptyset$$, and $$\rho(0)=1$$.
    - Lines 3-6 pick the root observation history $$0$$ and create one
      action history $$0a$$ for each $$a\in A$$.

    The constants $$u_{qa}$$, $$r_{qa}$$, $$\tau(qao)$$, and $$\rho (qao)$$ are
    intentionally not computed here; the complete Algorithm 1 loop calls
    Algorithm 2 (`expand_frontier_item`) from `planning.ilp_tree.paper_preprocess`.

    / 显式实现论文 Algorithm 1 的初始化部分：root 是历史 ``0``，
    frontier 是 root 下所有待调用 Algorithm 2 的 action history。
    """

    # Algorithm 1 line 1: initialize $$G$$, $$N=\{0\}$$, $$F=\emptyset$$, and $$\rho(0)=1$$.
    # 论文第 1 行：初始化树、open observation history 集合 $$N$$、已展开集合 $$F$$，以及 $$\rho(0)$$。
    root = interface.root
    open_histories = (root,)
    root_belief = _root_belief_from_runtime_or_override(runtime, interface, root_belief)
    if interface.exact_kernel is not None and root_belief is not None:
        # Lemma 3.3 root safe belief b0*: real ExactRDDLKernel conditions on
        # the initial safe event; no-risk test kernels may omit the method.
        # Lemma 3.3 的 b0*：真实 kernel 会扣除初始 unsafe；无风险测试替身可直接复用 b0。
        safe_root = getattr(interface.exact_kernel, "safe_belief_from_belief", None)
        root_safe_belief = safe_root(root_belief) if safe_root is not None else root_belief
    else:
        root_safe_belief = None

    # Algorithm 1 lines 3-6: pop q=root from N and create qa for every action.
    # 论文第 3-6 行：从 N 取出 root observation history，并为每个 action 创建 qa。
    action_nodes = interface.action_nodes(root)
    for node in action_nodes:
        root.add_child(node)
    frontier = tuple(
        FrontierItem(
            node=node,
            rho=1.0,
            root_action_label=str(node.metadata.get("action", "noop")),
            belief=root_belief,
            safe_belief=root_safe_belief,
            duration_beliefs=(),
            belief_trace=(root_belief,) if root_belief is not None else (),
            observation_keys=(),
        )
        for node in action_nodes
    )
    return RootFrontier(
        root=root,
        frontier=frontier,
        open_histories=open_histories,
    )
def _root_belief_from_runtime_or_override(
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
    root_belief: Mapping[StateKey, float] | None,
) -> Mapping[StateKey, float] | None:
    """Return explicit online belief or singleton runtime-state belief. / 返回显式在线 belief 或 runtime state 单点 belief。"""
    if interface.exact_kernel is None:
        return None
    if root_belief is not None:
        return _normalize_root_belief(root_belief)
    return interface.exact_kernel.initial_belief_from_state(runtime.state)


def _normalize_root_belief(belief: Mapping[StateKey, float]) -> Mapping[StateKey, float]:
    """Normalize root belief probabilities. / 归一化 root belief 概率。"""
    cleaned = {
        state: float(probability)
        for state, probability in belief.items()
        if abs(float(probability)) > 1e-15
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        raise ValueError("Root belief must contain positive probability mass.")
    return {state: probability / total for state, probability in cleaned.items()}
