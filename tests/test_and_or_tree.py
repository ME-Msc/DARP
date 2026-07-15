"""Tests for DARP-native AND-OR tree data structures."""

from darp.model.and_or_tree import (
    ANDORNode,
    ANDORNodeKind,
    ANDORSearchInterface,
    ActionChoice,
    History,
    ObservationScope,
)


def test_history_tracks_actions_observations_and_depth():
    """Check histories append action/observation labels immutably. / 检查 history 能不可变地追加 action/observation 标签。"""
    history = History().append_action("move-east").append_observation("at-c12")

    assert history.depth == 1
    assert history.actions == ("move-east",)
    assert history.observations == ("at-c12",)
    assert history.label() == "a0=move-east / o1=at-c12"


def test_and_or_node_attaches_children():
    """Check AND-OR nodes can form a tree. / 检查 AND-OR 节点可以组成树。"""
    root = ANDORNode(node_id="root", kind=ANDORNodeKind.OR)
    child = ANDORNode(
        node_id="root/move-east",
        kind=ANDORNodeKind.AND,
        history=History().append_action("move-east"),
    )

    assert root.is_leaf is True
    root.add_child(child)

    assert root.is_leaf is False
    assert root.children == [child]
    assert root.children[0].history.depth == 1


def test_and_or_node_deduplicates_children_by_id():
    """Check child attachment uses a constant-time id index. / 检查子节点挂接使用常数时间编号索引去重。"""
    root = ANDORNode(node_id="root", kind=ANDORNodeKind.OR)
    first = ANDORNode(node_id="root/a:move", kind=ANDORNodeKind.AND)
    duplicate = ANDORNode(node_id="root/a:move", kind=ANDORNodeKind.AND)

    root.add_child(first)
    root.add_child(duplicate)

    assert root.children == [first]


def test_search_interface_reuses_integer_indexed_history_nodes():
    """Check the custom node arena reuses histories without NetworkX overhead. / 检查专用节点池复用 history 且无需 NetworkX。"""
    interface = ANDORSearchInterface.from_actions_and_observations(
        actions=(ActionChoice(label="move", assignment={"move": True}),),
        observation_scope=ObservationScope(mode="mdp-state", variables=("at",)),
    )

    first = interface.action_nodes()[0]
    second = interface.action_nodes()[0]
    observation = interface.observation_node(first, "at-goal")

    assert first is second
    assert interface.root.node_index == 0
    assert first.node_index == 1
    assert first.parent_index == 0
    assert observation.parent_index == first.node_index
    assert interface.node_count == 3
