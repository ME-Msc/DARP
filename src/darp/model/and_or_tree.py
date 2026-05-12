"""AND-OR history tree data structures for DARP search."""

# TODO(phase-7.1): Add expansion bookkeeping from the paper once the grounded
# model view exposes transition and observation support.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


@dataclass(frozen=True)
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


@dataclass
class ANDORNode:
    """Represent one node in an AND-OR history tree. / 表示 AND-OR history tree 中的一个节点。"""

    node_id: str
    kind: ANDORNodeKind
    history: History = field(default_factory=History)
    children: list["ANDORNode"] = field(default_factory=list)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def add_child(self, child: "ANDORNode") -> None:
        """Attach a child node. / 挂接一个子节点。"""
        self.children.append(child)

    @property
    def is_leaf(self) -> bool:
        """Return whether the node has no children. / 返回该节点是否没有子节点。"""
        return not self.children


@dataclass(frozen=True)
class ActionChoice:
    """Describe one concrete action branch for search. / 描述搜索中的一个具体 action 分支。"""

    label: str
    assignment: Mapping[str, Any]


@dataclass(frozen=True)
class ObservationScope:
    """Describe which variables define observations. / 描述哪些变量定义 observation。"""

    mode: str
    variables: tuple[str, ...]


@dataclass(frozen=True)
class ANDORSearchInterface:
    """Bundle grounded action and observation inputs for AND-OR search. / 打包 AND-OR 搜索所需的 action 与 observation 输入。"""

    root: ANDORNode
    actions: tuple[ActionChoice, ...]
    observation_scope: ObservationScope

    @classmethod
    def from_actions_and_observations(
        cls,
        actions: tuple[ActionChoice, ...],
        observation_scope: ObservationScope,
    ) -> "ANDORSearchInterface":
        """Create a root interface from action choices and observation scope. / 从 action choice 和 observation scope 创建根接口。"""
        return cls(
            root=ANDORNode(node_id="root", kind=ANDORNodeKind.OR),
            actions=actions,
            observation_scope=observation_scope,
        )

    def action_nodes(self, parent: ANDORNode | None = None) -> tuple[ANDORNode, ...]:
        """Return AND children for each action choice. / 为每个 action choice 返回 AND 子节点。"""
        source = parent or self.root
        return tuple(
            ANDORNode(
                node_id=f"{source.node_id}/a:{_node_token(action.label)}",
                kind=ANDORNodeKind.AND,
                history=source.history.append_action(action.label),
                metadata={"action": action.label, "assignment": dict(action.assignment)},
            )
            for action in self.actions
        )

    def observation_node(self, parent: ANDORNode, observation_label: str) -> ANDORNode:
        """Return an OR child for one observation outcome. / 为一个 observation outcome 返回 OR 子节点。"""
        return ANDORNode(
            node_id=f"{parent.node_id}/o:{_node_token(observation_label)}",
            kind=ANDORNodeKind.OR,
            history=parent.history.append_observation(observation_label),
            metadata={
                "observation": observation_label,
                "observation_mode": self.observation_scope.mode,
                "observation_variables": self.observation_scope.variables,
            },
        )


def _node_token(value: str) -> str:
    """Return a path-safe node token. / 返回适合 node_id 路径使用的 token。"""
    return value.replace("/", "_").replace(" ", "_") or "empty"
