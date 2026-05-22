"""Preprocessing helpers for paper-aligned AND-OR search."""

# TODO(phase-9.1): Add benchmark trace hooks for frontier scores and generated
# ILP variable ids.

from __future__ import annotations

from dataclasses import dataclass

from darp.adapter.runtime import PyRDDLGymRuntime
from darp.model.and_or_tree import ANDORNode, ANDORSearchInterface


@dataclass(frozen=True, eq=False)
class FrontierItem:
    """Track one expandable action node and its parent runtime. / 跟踪一个可展开 action 节点及其父 runtime。"""

    node: ANDORNode
    parent_runtime: PyRDDLGymRuntime
    rho: float = 1.0
    root_action_label: str | None = None

    @property
    def action_label(self) -> str:
        """Return the node action label. / 返回该节点的 action 标签。"""
        return str(self.node.metadata.get("action", "noop"))

    @property
    def root_label(self) -> str:
        """Return the first action on this branch. / 返回该分支上的第一个 action。"""
        return self.root_action_label or self.action_label


@dataclass(frozen=True)
class PreprocessedSearchTree:
    """Store the root and initial frontier for paper search. / 保存论文搜索的根节点和初始 frontier。"""

    root: ANDORNode
    frontier: tuple[FrontierItem, ...]


def preprocess_search_tree(
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
) -> PreprocessedSearchTree:
    r"""Initialize the paper search tree before repeated expansion.

    Paper correspondence:

    - Preprocessing starts from the root observation history with
      :math:`\rho(root)=1`.
    - The first frontier :math:`F` contains one action-history child for each
      available action under the root OR node.

    This helper only builds deterministic bookkeeping; Phase 8 ILP encoders
    consume the resulting frontier to create full-ILP and p-ILP variables.

    / 初始化论文搜索树：root 的 :math:`\rho=1`，frontier 为根节点下的所有 action-history。
    """

    action_nodes = interface.action_nodes(interface.root)
    for node in action_nodes:
        _add_child_once(interface.root, node)
    frontier = tuple(
        FrontierItem(
            node=node,
            parent_runtime=runtime.clone(),
            rho=1.0,
            root_action_label=str(node.metadata.get("action", "noop")),
        )
        for node in action_nodes
    )
    return PreprocessedSearchTree(root=interface.root, frontier=frontier)


def _add_child_once(parent: ANDORNode, child: ANDORNode) -> None:
    """Attach a child only once by node id. / 按 node id 仅挂接一次子节点。"""
    if all(existing.node_id != child.node_id for existing in parent.children):
        parent.add_child(child)
