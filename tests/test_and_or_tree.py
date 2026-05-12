"""Tests for DARP-native AND-OR tree data structures."""

from darp.model.and_or_tree import ANDORNode, ANDORNodeKind, History


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
