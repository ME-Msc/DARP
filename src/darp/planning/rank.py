"""Restrict p-ILP action choices without changing its objective or risk semantics.

/ 筛选 p-ILP 的候选动作历史，不改变效用目标、全局风险预算或观测分支语义。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from math import ceil, isfinite

from darp.ilp.model import ILPLinearConstraint, ILPModelSpec


def validate_rank(alpha: float, lambda_: float) -> None:
    """Validate the nonnegative risk weight and retention fraction.

    / alpha 是非负风险评分权重；lambda 是候选保留比例，必须在 (0, 1] 内。
    """
    if isinstance(alpha, bool) or not isfinite(alpha) or alpha < 0.0:
        raise ValueError("rank alpha must be finite and nonnegative.")
    if isinstance(lambda_, bool) or not isfinite(lambda_) or not 0.0 < lambda_ <= 1.0:
        raise ValueError("rank lambda must be finite and in (0, 1].")


def restrict_rank_candidates(
    spec: ILPModelSpec,
    *,
    alpha: float,
    lambda_: float,
    frontier_variable_ids: AbstractSet[str] = frozenset(),
    warm_start: Mapping[str, float] | None = None,
) -> ILPModelSpec:
    """Protect F prefixes, then prune E bottom-up before trimming optional F.

    lambda is the retained fraction in (0, 1]; alpha weights first-entry risk.
    Protect ceil(lambda*|F|) high-score histories and usable previous selections.
    Promote candidate subtree roots until their unprotected E capacity meets
    the remaining deletion target. Score only then, keeping protected paths
    and at least one action per observation. Terminal E leaves also seed the
    traversal, so completed branches do not disappear from consideration.
    Quotas are soft; restriction preserves coefficients, not global optimality.

    / lambda 为保留比例，alpha 为首次风险评分权重。保护前 ceil(lambda*|F|)
    个 F、上一轮可用选择及其祖先。自底向上提升候选子树根，容量足够后才评分；
    删除时保留受保护路径与每个观测的至少一个动作。已完成 E 叶节点也作为
    遍历起点。比例是软目标；不修改目标和风险系数，但不保证全局最优性。
    """
    validate_rank(alpha, lambda_)
    if lambda_ == 1.0:
        return spec
    variable_ids = spec.variable_ids()
    declared_ids = set(variable_ids)
    if len(declared_ids) != len(variable_ids):
        raise ValueError("Rank requires distinct action-history variables.")
    if set(spec.objective) - declared_ids:
        raise ValueError("Rank objective references undeclared variables.")
    frontier_ids = set(frontier_variable_ids)
    if frontier_ids - declared_ids:
        raise ValueError("Rank frontier references undeclared variables.")
    if any(not isfinite(value) for value in spec.objective.values()):
        raise ValueError("Rank requires finite objective coefficients.")

    # OR groups choose one action; different observation groups are AND branches.
    # OR 组中选一个动作；同一父动作的不同观测组是必须全部覆盖的 AND 分支。
    groups: dict[str, list[tuple[str, ...]]] = {q: [] for q in variable_ids}
    parent: dict[str, str] = {}
    roots: tuple[str, ...] | None = None
    deadends: set[str] = set()
    risk: Mapping[str, float] = {}
    budget: float | None = None
    row_names: set[str] = set()
    for row in spec.constraints:
        if set(row.coefficients) - declared_ids:
            raise ValueError(f"Rank constraint references undeclared variables: {row.name}")
        if row.name in row_names:
            raise ValueError(f"Duplicate p-ILP constraint: {row.name}")
        row_names.add(row.name)
        terms = {q: value for q, value in row.coefficients.items() if value != 0.0}
        if not isfinite(row.rhs) or any(not isfinite(value) for value in terms.values()):
            raise ValueError(f"Nonfinite p-ILP constraint: {row.name}")
        if row.name == "root_action":
            if row.sense != "==" or row.rhs != 1.0 or any(value != 1.0 for value in terms.values()):
                raise ValueError("Rank requires sum(root actions) == 1.")
            roots = tuple(terms)
        elif row.name == "risk_budget":
            if row.sense != "<=" or any(value < 0.0 for value in terms.values()):
                raise ValueError("Rank requires a nonnegative first-entry risk row.")
            risk, budget = terms, row.rhs
        elif row.name.startswith("deadend_"):
            if row.sense != "==" or row.rhs != 0.0 or len(terms) != 1 or next(iter(terms.values())) != 1.0:
                raise ValueError(f"Unsupported dead-end row: {row.name}")
            deadends.update(terms)
        elif row.name.startswith("flow_"):
            parents = [q for q, value in terms.items() if value == -1.0]
            if row.sense != "==" or row.rhs != 0.0 or len(parents) != 1 or any(value not in (-1.0, 1.0) for value in terms.values()):
                raise ValueError(f"Unsupported observation-flow row: {row.name}")
            q = parents[0]
            children = tuple(child for child, value in terms.items() if value == 1.0)
            for child in children:
                if child in parent:
                    raise ValueError("Rank requires a history tree, not shared child variables.")
                parent[child] = q
            groups[q].append(children)
        else:
            raise ValueError(f"Unsupported p-ILP constraint for Rank: {row.name}")
    if roots is None or any(q in parent for q in roots):
        raise ValueError("Rank requires a root group with no parent histories.")
    if any(groups[q] for q in frontier_ids):
        raise ValueError("Rank frontier histories must have no child-flow rows.")

    # Use tree topology, not variable numbering; this also detects orphan/cyclic rows.
    # 根据树拓扑而非变量编号遍历，同时拒绝孤立节点或环形依赖。
    order: list[str] = []
    queue = deque(roots)
    prefix_risk: dict[str, float] = {}
    depth: dict[str, int] = {}
    while queue:
        q = queue.popleft()
        order.append(q)
        depth[q] = depth[parent[q]] + 1 if q in parent else 1
        prefix_risk[q] = risk.get(q, 0.0) + (prefix_risk[parent[q]] if q in parent else 0.0)
        for children in groups[q]:
            queue.extend(children)
    if len(order) != len(variable_ids):
        raise ValueError("Rank requires every variable to belong to the rooted history tree.")

    # R already equals Delta - r(b0). Prefix safety is necessary, not sufficient:
    # other observations also contribute risk. Keep the original global row.
    # R 已扣除初始风险；前缀未超预算仅是必要条件，其他观测仍贡献风险，须保留全局行。
    eligible: set[str] = set()
    for q in reversed(order):
        if q in deadends or (budget is not None and prefix_risk[q] > budget + 1e-6):
            continue
        if all(any(child in eligible for child in children) for children in groups[q]):
            eligible.add(q)

    # F scores are local; defer E continuation scores until capacity suffices.
    # F 直接评分；E 的后续评分延迟到候选容量足够之后。
    score = {q: spec.objective.get(q, 0.0) - alpha * risk.get(q, 0.0)
             for q in frontier_ids}
    if any(not isfinite(value) for value in score.values()):
        raise ValueError("Rank score overflow; rescale utility or alpha.")

    # A locally usable child can still lie below an impossible ancestor.
    # Remove such branches before ranking, so protected histories are reachable.
    # 局部可用子节点可能位于不可行祖先之下；先排除这些分支，保证保护对象可达。
    for q in order:
        if q in parent and parent[q] not in eligible:
            eligible.discard(q)

    ranked_frontier = sorted(
        (q for q in variable_ids if q in frontier_ids and q in eligible),
        key=score.__getitem__, reverse=True,
    )
    protected = set(ranked_frontier[:ceil(len(frontier_ids) * lambda_)])
    if warm_start:
        protected.update(
            q for q in variable_ids
            if q in eligible and warm_start.get(q, 0.0) > 0.5
        )
    # Propagate protection once; keep warm starts feasible under zero-fixing.
    # 一次向上传播保护标记，使仍可用的上一轮解不会因本轮筛选丢失。
    for q in reversed(order):
        if q in protected and q in parent:
            protected.add(parent[q])

    action_groups = [roots] + [children for q in order for children in groups[q]]
    kept = set(eligible)
    _prune_expanded(
        kept, frontier_ids, protected, parent, groups, action_groups, order,
        spec.objective, risk, alpha,
        ceil((len(variable_ids) - len(frontier_ids)) * lambda_),
    )

    # Remove unprotected F only where an E action or protected F remains;
    # otherwise keep the best F. All observations of a kept E remain covered.
    # 若组内已有 E 或受保护 F，则移除其余 F；否则保留最佳 F，保证所有观测有后续动作。
    for children in action_groups:
        available = [q for q in children if q in kept]
        optional = [q for q in available if q in frontier_ids and q not in protected]
        if optional and len(optional) == len(available):
            optional.remove(max(optional, key=score.__getitem__))
        kept.difference_update(optional)

    if kept == declared_ids:
        return spec

    # Substitute omitted x_q=0. Only empty 0=0 structural rows disappear;
    # root/risk rows stay, including an infeasible 0=1 or 0<=negative budget.
    # 将未保留变量代入 0；仅删除恒等的结构行，空根约束或负预算行仍保留以表达不可行。
    constraints: list[ILPLinearConstraint] = []
    for row in spec.constraints:
        terms = {q: value for q, value in row.coefficients.items() if q in kept and value != 0.0}
        if not terms and row.name not in ("root_action", "risk_budget") and row.sense == "==" and row.rhs == 0.0:
            continue
        constraints.append(ILPLinearConstraint(row.name, terms, row.sense, row.rhs))
    return ILPModelSpec(
        name=spec.name,
        variables=tuple(variable for variable in spec.variables if variable.var_id in kept),
        objective={q: value for q, value in spec.objective.items() if q in kept},
        constraints=tuple(constraints),
    )


def _select_subtree_roots(
    action_groups: list[tuple[str, ...]],
    frontier_ids: AbstractSet[str],
    eligible: AbstractSet[str],
    protected: AbstractSet[str],
    score: Mapping[str, float],
    depth: Mapping[str, int],
    expanded_count: Mapping[str, int],
    target: int,
) -> tuple[str, ...]:
    """Select disjoint E roots at the deepest depth able to meet the soft target.

    Reserve every protected choice and at least one usable action per group.
    If a whole group consists of deletable E roots, reserve its best-score
    root. Its remaining E count defines that group's deletion capacity under
    this rule (not the maximum possible capacity). Candidate loss is the gap
    to the best sibling, NOT a certified bound. Capacities are summed by depth;
    no subtree is modified during layer selection. If no depth meets target,
    use the largest positive capacity, breaking ties towards deeper layers.
    Sort only the chosen layer by decreasing sibling gap, then delete whole
    subtrees until target is met or capacity is exhausted. Overshoot is allowed.

    / 选择能达到软目标的最深层 E 子树根。每组保护历史选择并至少保留一个动作；
    若全组均可删除，预留最高分 E。其余子树的 E 数量构成该规则下的容量，并非
    数学上的最大可删量。评分差仅用于排序，不是损失上界。先按深度统计容量，
    不实际试删；都不达标时取容量最大的层，同容量优先深层。仅排序最终选定层，
    按相对最佳兄弟的评分差从大到小删除整棵子树，允许最后一棵导致超额。
    """
    if target <= 0:
        return ()
    candidates: dict[int, list[tuple[str, float]]] = {}
    capacity: dict[int, int] = {}
    for children in action_groups:
        available = [q for q in children if q in eligible]
        removable = [q for q in available if q not in frontier_ids and q not in protected]
        if not removable:
            continue
        best = max(available, key=score.__getitem__)
        if len(removable) == len(available):
            removable.remove(best)
        for q in removable:
            d = depth[q]
            candidates.setdefault(d, []).append((q, score[best] - score[q]))
            capacity[d] = capacity.get(d, 0) + expanded_count[q]
    if not capacity:
        return ()
    sufficient = [d for d, count in capacity.items() if count >= target]
    selected_depth = max(sufficient) if sufficient else max(capacity, key=lambda d: (capacity[d], d))
    removed: list[str] = []
    count = 0
    for q, _ in sorted(candidates[selected_depth], key=lambda item: item[1], reverse=True):
        removed.append(q)
        count += expanded_count[q]
        if count >= target:
            break
    return tuple(removed)

def _prune_expanded(
    kept: set[str],
    frontier: AbstractSet[str],
    protected: AbstractSet[str],
    parent: Mapping[str, str],
    groups: Mapping[str, list[tuple[str, ...]]],
    action_groups: list[tuple[str, ...]],
    order: list[str],
    objective: Mapping[str, float],
    risk: Mapping[str, float],
    alpha: float,
    limit: int,
) -> None:
    """Prune unprotected side branches in-place, promoting roots bottom-up.

    Counts include E only, even completed E leaves. Nested candidate roots are
    deduplicated before capacity summation. Deleting a branch updates counts
    only along its ancestors. Each scoring wave uses one bottom-up DP; scores
    order deletions, they are not loss bounds. No tree mutation or recursion.

    / 原位筛选 kept，不修改原树。E 计数包含已完成叶节点；嵌套候选根先去重。
    删枝仅更新祖先计数，每个评分轮次用一次自底向上 DP。评分仅用于排序，
    不是损失上界；受保护节点和最后一个观测动作始终保留，无递归调用。
    """
    expanded = len(kept - frontier)
    if expanded <= limit:
        return
    group_of = {q: i for i, children in enumerate(action_groups) for q in children}
    available = [sum(q in kept for q in children) for children in action_groups]
    count: dict[str, int] = {}
    for q in reversed(order):
        count[q] = int(q in kept and q not in frontier and q not in protected) + sum(
            count[child] for children in groups[q] for child in children
        )
    # Completed E branches must participate even when there is no frontier.
    # 无 F 的已完成分支也参与筛选，不能仅从 F 的父节点寻找 E。
    candidates = {parent[q] for q in kept & frontier - protected if q in parent}
    candidates.update(q for q in kept - frontier if not groups[q])
    position = {q: i for i, q in enumerate(order)}
    while expanded > limit and candidates:
        # Keep only outer roots: capacities then sum without counting twice.
        # 只留最外层候选根；不等深子树重叠时容量不会重复累计。
        disjoint = set()
        for q in sorted(candidates, key=position.__getitem__):
            ancestor = parent.get(q)
            while ancestor is not None and ancestor not in disjoint:
                ancestor = parent.get(ancestor)
            if ancestor is None and q in kept:
                disjoint.add(q)
        candidates = disjoint
        parents = {parent[q] for q in candidates if q in parent and parent[q] in kept}
        capacity = sum(count[q] for q in candidates)
        if capacity and (capacity >= expanded - limit or not parents):
            score: dict[str, float] = {}
            for q in reversed(order):
                if q in kept:
                    score[q] = objective.get(q, 0.0) - alpha * risk.get(q, 0.0) + sum(
                        max(score[child] for child in children if child in kept)
                        for children in groups[q]
                    )
                    if not isfinite(score[q]):
                        raise ValueError("Rank score overflow; rescale utility or alpha.")
            for root in sorted(candidates, key=lambda q: (score[q], position[q])):
                stack = [root]
                while stack and expanded > limit:
                    q = stack.pop()
                    if q not in kept or not count[q]:
                        continue
                    if q not in protected and available[group_of[q]] > 1:
                        # This cut is a complete unprotected action subtree.
                        # 仅此处整棵删枝；受保护根则继续向下寻找可删侧枝。
                        removed_count = count[q]
                        descendants = [q]
                        while descendants:
                            child = descendants.pop()
                            if child not in kept:
                                continue
                            kept.remove(child)
                            available[group_of[child]] -= 1
                            count[child] = 0
                            descendants.extend(c for obs in groups[child] for c in obs)
                        expanded -= removed_count
                        ancestor = parent.get(q)
                        while ancestor is not None:
                            count[ancestor] -= removed_count
                            ancestor = parent.get(ancestor)
                    else:
                        stack.extend(reversed([c for obs in groups[q] for c in obs]))
                if expanded <= limit:
                    break
        candidates = parents & kept
