"""Root-frontier initialization helpers for paper Algorithm 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import isfinite
from typing import Mapping

from darp.adapter.runtime import PyRDDLGymRuntime
from darp.model.and_or_tree import ANDORNode, ANDORSearchInterface
from darp.model.duration import DurationProgress
from darp.adapter.exact import (
    ObservationKey,
    StateKey,
    risk_constraint_type_for_kernel,
)


@dataclass(frozen=True, eq=False)
class FrontierItem:
    """Track one action history with authoritative exact probability masses."""

    node: ANDORNode
    belief: Mapping[StateKey, float]
    ordinary_mass: Mapping[StateKey, Fraction]
    constraint_mass: Mapping[StateKey, Fraction]
    ordinary_mass_trace: tuple[Mapping[StateKey, Fraction], ...] = ()
    observation_keys: tuple[ObservationKey, ...] = ()
    duration_progress: DurationProgress = field(default_factory=DurationProgress)

    @property
    def action_label(self) -> str:
        """Return the node action label. / 返回该节点的 action 标签。"""
        return self.node.action_label or "noop"


def initialize_root_frontier(
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
    *,
    root_belief: Mapping[StateKey, float] | None = None,
) -> tuple[FrontierItem, ...]:
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
    kernel = interface.exact_kernel
    if kernel is None:
        raise ValueError("Paper preprocessing requires interface.exact_kernel.")
    root_belief = resolve_root_belief(runtime, interface, root_belief)
    if root_belief is None:
        raise ValueError("Paper preprocessing requires a root belief.")

    # Exact unnormalized mass is the sole probability-flow representation.
    root_ordinary_mass = kernel.initial_constraint_mass(root_belief)
    constraint_type = risk_constraint_type_for_kernel(kernel)
    if constraint_type == "chance":
        root_constraint_mass = kernel.initial_safe_mass(root_belief)
    else:
        root_constraint_mass = root_ordinary_mass

    # Algorithm 1 lines 3-6: pop q=root from N and create qa for every action.
    # 论文第 3-6 行：从 N 取出 root observation history，并为每个 action 创建 qa。
    action_nodes = interface.action_nodes(root, belief=root_belief)
    frontier = tuple(
        FrontierItem(
            node=node,
            belief=root_belief,
            ordinary_mass=root_ordinary_mass,
            constraint_mass=root_constraint_mass,
            ordinary_mass_trace=(root_ordinary_mass,),
            observation_keys=(),
            duration_progress=DurationProgress(),
        )
        for node in action_nodes
    )
    return frontier
def resolve_root_belief(
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
    root_belief: Mapping[StateKey, float] | None,
) -> Mapping[StateKey, float] | None:
    """Resolve one root belief consistently for tree and risk encoding.

    An explicit online belief always wins. A POMDP without one uses the
    model-declared b0 so no simulator hidden state leaks into planning. An MDP
    uses the runtime's current state, which matters when the public planner API
    is called after the environment has already advanced.
    """
    if interface.exact_kernel is None:
        return None
    if root_belief is not None:
        return _normalize_root_belief(root_belief)
    if interface.observation_scope.mode == "pomdp-observation":
        return _normalize_root_belief(
            interface.exact_kernel.initial_belief_from_model()
        )
    return interface.exact_kernel.initial_belief_from_state(runtime.state)


def _normalize_root_belief(belief: Mapping[StateKey, float]) -> Mapping[StateKey, float]:
    """Normalize root belief probabilities. / 归一化 root belief 概率。"""
    numeric: dict[StateKey, Fraction] = {}
    non_finite: dict[StateKey, object] = {}
    for state, raw_probability in belief.items():
        if isinstance(raw_probability, Fraction):
            numeric[state] = raw_probability
            continue
        probability = float(raw_probability)
        if not isfinite(probability):
            non_finite[state] = probability
        else:
            numeric[state] = Fraction.from_float(probability)
    if non_finite:
        raise ValueError(f"Root belief contains non-finite probability mass: {non_finite!r}")
    negative = {
        state: probability
        for state, probability in numeric.items()
        if probability < 0
    }
    if negative:
        raise ValueError(f"Root belief contains negative probability mass: {negative!r}")
    cleaned = {
        state: probability
        for state, probability in numeric.items()
        if probability > 0
    }
    total = sum(cleaned.values(), start=Fraction(0))
    if total <= 0:
        raise ValueError("Root belief must contain positive probability mass.")
    return {state: probability / total for state, probability in cleaned.items()}
