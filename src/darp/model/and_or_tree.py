"""AND-OR history tree data structures for DARP search. / DARP 搜索使用的 AND-OR 历史树数据结构。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import quote

# Histories receive POMDP observations or fully observed MDP states.
# 历史接收 POMDP 观测，或完全可观测的 MDP 状态。
ObservationMode = Literal["pomdp-observation", "mdp-state"]


@dataclass(frozen=True, slots=True)
class History:
    """Store alternating action and observation labels. / 保存交替的 action 与 observation 标签。"""

    actions: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()

    @property
    def depth(self) -> int:
        """Return the number of action decisions in the history. / 返回 history 中 action decision 的数量。"""
        return len(self.actions)

    def append_action(self, action: str) -> History:
        """Return a history extended by one action. / 返回追加一个 action 后的 history。"""
        return History(actions=self.actions + (action,), observations=self.observations)

    def append_observation(self, observation: str) -> History:
        """Return a history extended by one observation. / 返回追加一个 observation 后的 history。"""
        return History(actions=self.actions, observations=self.observations + (observation,))

    def label(self) -> str:
        """Return a compact action-observation path label. / 返回紧凑的 action-observation 路径标签。"""
        parts: list[str] = []
        for index, action in enumerate(self.actions):
            parts.append(f"a{index}={action}")
            if index < len(self.observations):
                parts.append(f"o{index + 1}={self.observations[index]}")
        return " / ".join(parts) if parts else "root"


@dataclass(slots=True)
class ANDORNode:
    """Represent one node in an AND-OR history tree. / 表示 AND-OR history tree 中的一个节点。"""

    node_id: str
    node_index: int = -1  # Compact arena id for hot-path lookup. / 热路径查询使用的紧凑节点编号。
    history: History = field(default_factory=History)
    action_label: str | None = None
    assignment: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ActionChoice:
    """Describe one concrete action branch for search. / 描述搜索中的一个具体 action 分支。"""

    label: str
    assignment: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ANDORSearchInterface:
    """Bundle grounded action and observation inputs for AND-OR search. / 打包 AND-OR 搜索所需的 action 与 observation 输入。"""

    root: ANDORNode
    actions: tuple[ActionChoice, ...]
    observation_mode: ObservationMode
    kernel: Any | None = None
    _nodes_by_id: dict[str, ANDORNode] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Register the root in the compact node arena. / 将 root 登记到紧凑节点池。"""
        self.root.node_index = 0
        self._nodes_by_id[self.root.node_id] = self.root

    @classmethod
    def from_actions_and_observations(
        cls,
        actions: tuple[ActionChoice, ...],
        observation_mode: ObservationMode,
        kernel: Any | None = None,
    ) -> ANDORSearchInterface:
        """Create a root interface from action choices and observation mode. / 从动作选项和观测模式创建根接口。"""
        return cls(
            root=ANDORNode(node_id="root"),
            actions=actions,
            observation_mode=observation_mode,
            kernel=kernel,
        )

    def action_choices(
        self,
        belief: Mapping[Any, Any] | None = None,
    ) -> tuple[ActionChoice, ...]:
        """Return actions available at ``belief``, preserving grounded order.

        Kernels with state-dependent action sets may expose
        ``available_action_labels(belief)``.  Static RDDL kernels need no extra
        method and retain the original global action set.

        / 保持实例化动作的顺序；支持状态依赖动作集的内核可提供上述方法，
        静态 RDDL 内核直接沿用全局动作集。
        """
        available = getattr(self.kernel, "available_action_labels", None)
        if belief is None or not callable(available):
            return self.actions
        labels = tuple(available(belief))
        if any(not isinstance(label, str) for label in labels):
            raise TypeError("available_action_labels() must return string labels")
        selected = set(labels)
        if len(labels) != len(selected):
            raise ValueError("available_action_labels() returned duplicate labels")
        known = {action.label for action in self.actions}
        unknown = selected - known
        if unknown:
            raise ValueError(
                "available_action_labels() returned unknown actions: "
                + ", ".join(sorted(unknown))
            )
        return tuple(action for action in self.actions if action.label in selected)

    def action_nodes(
        self,
        parent: ANDORNode | None = None,
        *,
        belief: Mapping[Any, Any] | None = None,
    ) -> tuple[ANDORNode, ...]:
        """Return available AND children. / 返回当前 belief 下可用的 AND 子节点。"""
        source = parent or self.root
        return tuple(
            self._intern_node(
                ANDORNode(
                    node_id=f"{source.node_id}/a:{_node_token(action.label)}",
                    history=source.history.append_action(action.label),
                    action_label=action.label,
                    assignment=dict(action.assignment),
                )
            )
            for action in self.action_choices(belief)
        )

    def observation_node(self, parent: ANDORNode, observation_label: str) -> ANDORNode:
        """Return an OR child for one observation outcome. / 为一个 observation outcome 返回 OR 子节点。"""
        return self._intern_node(
            ANDORNode(
                node_id=f"{parent.node_id}/o:{_node_token(observation_label)}",
                history=parent.history.append_observation(observation_label),
            )
        )

    def _intern_node(self, candidate: ANDORNode) -> ANDORNode:
        """Reuse one history node or append it to the integer arena. / 复用 history 节点或将其加入整数节点池。"""
        existing = self._nodes_by_id.get(candidate.node_id)
        if existing is not None:
            return existing
        candidate.node_index = len(self._nodes_by_id)
        self._nodes_by_id[candidate.node_id] = candidate
        return candidate


def _node_token(value: str) -> str:
    """Return an injective path-safe token for an action/observation label. / 将动作或观测标签编码为无冲突且路径安全的标记。"""
    return f"{len(value)}:{quote(value, safe='')}"
