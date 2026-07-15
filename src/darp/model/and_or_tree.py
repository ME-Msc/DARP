"""AND-OR history tree data structures for DARP search."""

# TODO(phase-9.1): Add optional trace links from tree nodes to ILP variables if
# benchmark reports need them.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class History:
    """Store alternating action and observation labels. / 保存交替的 action 与 observation 标签。"""

    actions: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()

    @property
    def depth(self) -> int:
        """Return the number of action decisions in the history. / 返回 history 中 action decision 的数量。"""
        return len(self.actions)

    def append_action(self, action: str) -> "History":
        """Return a history extended by one action. / 返回追加一个 action 后的 history。"""
        return History(actions=self.actions + (action,), observations=self.observations)

    def append_observation(self, observation: str) -> "History":
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


class ANDORNodeKind(str, Enum):
    """Distinguish action-choice and observation-outcome nodes. / 区分 action choice 与 observation outcome 节点。"""

    AND = "and"
    OR = "or"


@dataclass(slots=True)
class ANDORNode:
    """Represent one node in an AND-OR history tree. / 表示 AND-OR history tree 中的一个节点。"""

    node_id: str
    kind: ANDORNodeKind
    node_index: int = -1  # Compact arena id for hot-path lookup. / 热路径查询使用的紧凑节点编号。
    parent_index: int | None = None  # Parent arena id; root uses None. / 父节点编号，root 为 None。
    history: History = field(default_factory=History)
    children: list["ANDORNode"] = field(default_factory=list)
    metadata: Mapping[str, object] = field(default_factory=dict)
    _child_ids: set[str] = field(default_factory=set, init=False, repr=False)

    def add_child(self, child: "ANDORNode") -> None:
        """Attach a child once with O(1) id lookup. / 使用 O(1) 编号查询仅挂接一次子节点。"""
        if child.node_id in self._child_ids:
            return
        self._child_ids.add(child.node_id)
        self.children.append(child)

    @property
    def is_leaf(self) -> bool:
        """Return whether the node has no children. / 返回该节点是否没有子节点。"""
        return not self.children


@dataclass(frozen=True, slots=True)
class ActionChoice:
    """Describe one concrete action branch for search. / 描述搜索中的一个具体 action 分支。"""

    label: str
    assignment: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ObservationScope:
    """Describe which variables define observations. / 描述哪些变量定义 observation。"""

    mode: str
    variables: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ANDORSearchInterface:
    """Bundle grounded action and observation inputs for AND-OR search. / 打包 AND-OR 搜索所需的 action 与 observation 输入。"""

    root: ANDORNode
    actions: tuple[ActionChoice, ...]
    observation_scope: ObservationScope
    exact_kernel: Any | None = None
    _nodes_by_id: dict[str, ANDORNode] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Register the root in the compact node arena. / 将 root 登记到紧凑节点池。"""
        self.root.node_index = 0
        self._nodes_by_id[self.root.node_id] = self.root

    @property
    def node_count(self) -> int:
        """Return the number of unique materialized histories. / 返回已实体化的唯一 history 数量。"""
        return len(self._nodes_by_id)

    @classmethod
    def from_actions_and_observations(
        cls,
        actions: tuple[ActionChoice, ...],
        observation_scope: ObservationScope,
        exact_kernel: Any | None = None,
    ) -> "ANDORSearchInterface":
        """Create a root interface from action choices and observation scope. / 从 action choice 和 observation scope 创建根接口。"""
        return cls(
            root=ANDORNode(node_id="root", kind=ANDORNodeKind.OR),
            actions=actions,
            observation_scope=observation_scope,
            exact_kernel=exact_kernel,
        )

    def action_nodes(self, parent: ANDORNode | None = None) -> tuple[ANDORNode, ...]:
        """Return AND children for each action choice. / 为每个 action choice 返回 AND 子节点。"""
        source = parent or self.root
        return tuple(
            self._intern_node(
                ANDORNode(
                    node_id=f"{source.node_id}/a:{_node_token(action.label)}",
                    kind=ANDORNodeKind.AND,
                    parent_index=source.node_index,
                    history=source.history.append_action(action.label),
                    metadata={"action": action.label, "assignment": dict(action.assignment)},
                )
            )
            for action in self.actions
        )

    def observation_node(self, parent: ANDORNode, observation_label: str) -> ANDORNode:
        """Return an OR child for one observation outcome. / 为一个 observation outcome 返回 OR 子节点。"""
        return self._intern_node(
            ANDORNode(
                node_id=f"{parent.node_id}/o:{_node_token(observation_label)}",
                kind=ANDORNodeKind.OR,
                parent_index=parent.node_index,
                history=parent.history.append_observation(observation_label),
                metadata={
                    "observation": observation_label,
                    "observation_mode": self.observation_scope.mode,
                    "observation_variables": self.observation_scope.variables,
                },
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
    """Return a path-safe node token. / 返回适合 node_id 路径使用的 token。"""
    return value.replace("/", "_").replace(" ", "_") or "empty"
