# DARP 实现审查与实验说明

## 结论

当前代码已从“能运行的原型”整理成与论文三段伪代码相对应的实现。最关键的修复是：恢复 HILP 的 incumbent-guided frontier expansion；分离 utility 的普通 belief flow 与 chance constraint 的 safe-conditioned flow；fixed duration 不再执行不必要的 smoothing；数值状态/转移只在可达 history 被触达时生成。full-ILP 保留为小规模 oracle，不再作为默认基准。

## 论文与代码映射

| 论文概念 | 当前实现 | 审查要点 |
|---|---|---|
| Algorithm 1：AND/OR history tree 初始化 | `planning/preprocess.py` | 根节点同时保存普通流、安全流和 duration progress；POMDP 初始 belief 来自模型分布，不读取仿真器隐藏真状态。 |
| Algorithm 2：belief、utility、risk、duration 传播 | `planning/expand.py` | 普通分支用于期望收益；safe-conditioned 分支用于 chance risk。fixed duration 走 O(1) 进度更新，stochastic duration 才 backward smoothing。 |
| policy-variable tree 与 ILP 系数 | `planning/ilp_tree.py` | 节点 ID 来自整数 arena，避免字符串碰撞；BFS 用 `deque`；完整树有显式节点上限。 |
| Algorithm 3：HILP dual-space search | `planning/hilp.py` | 每轮 solve p-ILP 后，只展开 incumbent 中 `x_q > 0.5` 的 frontier；复用上轮 MIP solution 做 warm start；只有 frontier 已穷尽或全剩余深度的 admissible bound 能排除未展开备选时才标记 `complete`。 |
| exact RDDL 数值核 | `adapter/exact.py` | 不枚举全状态空间；首次访问 `(state, action)` 时生成非零转移、reward、observation 并缓存。 |
| RDDL/sidecar 校验 | `adapter/grounded.py`、`model/duration*.py` | 对当前不支持且会破坏正确性的语义 fail closed；拒绝负概率、非法时长/方差/budget。 |

论文中的双概率递推是实现正确性的核心：普通 `rho/belief` 不能被安全事件过滤，否则会改变原问题的无条件期望 utility；safe `rho/belief` 则用于累计首次进入危险状态的执行概率。论文 Lemma 3.3 中 utility 的上标 `*` 指所求策略下的普通 Eq. (9)/(10) 概率与 belief，不是 safe 标记；安全概率另记为 `tilde rho`。两者现在在 `FrontierItem` 和 expansion 中显式分开。

## Git 历史审查

- `3a88e77` 已经实现了论文 Algorithm 3 的关键条件：只展开 ILP 解中被选择的 frontier variable。
- `c80b760` 引入回归：忽略 incumbent，改成全局 greedy frontier expansion；这会让搜索方向脱离当前可行策略，根节点也可能绕过 ILP 选择。本次已恢复 incumbent-guided expansion，并用回归测试锁定。
- `94c7566` 重构实验目录时删除了若干脚本，但 README 继续引用旧路径。本次移除所有失效命令，改成单一可执行 runner。
- `6b918ce` 引入按需可达状态、稀疏 kernel 和 cache；本次保留并补上 syntax-grounded model cache、truth-row cache 及 cache 指标。

## 性能与结构优化

1. **树展开**：HILP 只沿 incumbent policy refinement；`frontier_width` 只限制其中的候选，而不是从整棵树重新贪心挑选。
2. **belief/risk**：普通与 safe 流分离；safe mass 为零时仍保留影响 utility 的普通 observation branch。
3. **duration**：fixed action 只更新离散进度；避免每个 history 运行后向平滑。非正 duration 或非法 Gaussian 参数提前报错；full/HILP 均同时遵守 durative stopping 与 RDDL `remaining_depth` action-step 上限。
4. **ILP**：跨 refinement 轮次 warm start；Gurobi 默认单线程以提高实验可重复性；full planner 加节点上限，full/HILP 均将 runner 的剩余 wall budget 传给 Gurobi `TimeLimit`。
5. **热路径**：BFS 改为 `deque.popleft()`；节点 path token 使用可逆编码；state truth row、transition、reward、observation 均缓存。
6. **正确性边界**：当前 exact adapter 不支持的并发动作、非布尔动作、precondition/invariant/termination、discount 语义会明确报错，不再静默忽略；belief 概率中的负数、NaN 和 Inf 也会在进入 kernel/ILP 前被拒绝。

## grounding 的实际边界

目前实现不是完全 lifted 的逐步符号 grounding。pyRDDLGym 在解析阶段仍会把参数化 fluent/CPF grounding；本次把该结果缓存为一次性成本。真正昂贵的状态赋值笛卡尔积没有预先枚举：状态、非零 transition row 和 observation row 随 HILP 的可达 history 展开而按需生成。

因此当前优化准确地说是 **lazy reachable numerical grounding**。若要继续实现 lifted incremental grounding，需要从 parser AST 构造 action/CPF dependency slice，仅实例化当前 belief 支持集和候选 action 依赖的对象组合；这会绕开 pyRDDLGym 的整体 grounder，属于下一阶段的独立架构工作，不能用缓存包装冒充已完成。

## 对比算法选择

RAO* 与本项目的问题定义最匹配：都是有限时域、确定性策略、chance-constrained POMDP 的 AND/OR belief search。为避免外部仓库版本、Python 依赖和模型转换掩盖语义差异，本仓库实现了一个独立的小规模 RAO*-style baseline：完整展开 finite belief hypergraph，并在节点上维护 `(risk, value)` Pareto labels。它是 exhaustive semantics comparator，不是官方 RAO* 的 heuristic forward expansion，因此这里只比较相同语义下的本机实现表现，不声称超越官方 RAO*。

较新的方法并不构成同一实验协议下的直接替代：ConstrainedZero 使用神经网络与 MCTS，需要离线训练；RC-POMDP 改变了约束语义；recursive dual-ascent 方法则是另一类 Lagrangian/online 求解框架。因此本实验把 RAO*-style semantics 作为最接近的受控 comparator，而不是宣称 RAO* 在所有 constrained POMDP 定义下都是当前唯一 SOTA。

一手资料：

- RAO*（AAAI 2016）：<https://ojs.aaai.org/index.php/AAAI/article/view/10423>
- RAO* 官方公开代码：<https://github.com/JarvisIsFriday/RAOStar>
- ConstrainedZero（IJCAI 2024）：<https://www.ijcai.org/proceedings/2024/746>
- Recursively-Constrained POMDPs（UAI 2024）：<https://proceedings.mlr.press/v244/ho24a.html>
- Recursive dual ascent for constrained POMDPs（ICAPS 2024）：<https://ojs.aaai.org/index.php/ICAPS/article/view/31518>

## 实验场景与假设

实验采用论文中的 5×5 grid：start `(5,1)`、goal `(1,5)`；危险格为 `(1,1) (2,4) (2,5) (4,1) (4,2)`；移动成功/左右偏移概率为 `0.85/0.075/0.075`；wall-count observation 的正确/其他概率为 `0.85/0.075`。本次 unit-duration 比较令 goal absorbing，并使用论文风格的负 Manhattan terminal tail；muddy cell 不增加默认 cost。

默认矩阵固定 horizon 3、risk budget `{0.1, 0.2, 0.3}`、每格 3 次。每次 fresh kernel，记录 median 和 IQR。terminal tail 让小 horizon 仍能区分动作，但也意味着结果是 admissible-tail solution，不是 full-horizon exact certificate。`objective_cost=-objective` 含有 `(1-Δ)×Manhattan` surrogate tail，chance risk 只覆盖显式 h-step prefix；它不是独立 rollout 得到的期望到达成本，不应跨 budget 直接比较。三次重复只反映本机确定性求解时间波动，不等价于论文的 25 个试验实例。

CSV 中 `nodes` 的单位随算法而异，并由 `node_metric_kind` 明示：HILP 是实际展开的 action histories，RAO*-style 是 belief OR nodes，full-ILP 是 action-history variables。比较时应同时查看 HILP 的 `policy_variables/frontier_nodes`，不能把不同类型的节点数当作完全同构的工作量。`expansion_rounds_cap` 是配置上限，`expansion_rounds_used` 才是实际 refinement 次数。

运行：

```bash
tools/run_repro.sh
```

full-ILP 仅在 horizon 2 且 `--disable-terminal-tail` 时作为 exact-prefix oracle。这样 HILP、RAO*-style 与 full-ILP 的 objective contract 一致；带 tail 与不带 tail 的结果不会计算 cost gap。它仍留在代码中，但默认实验排除它，因为完整 history tree 的成本与 partial search 不在一个量级。

## 已知限制与后续优先级

1. 实现 lifted dependency-sliced grounding，并把 syntax grounding 时间单独纳入 RDDL 端到端 benchmark。
2. 若要声称更大规模性能，需要接入官方 RAO* Science Agent/PSR 或做经过验证的等价模型转换，并报告相同硬件和停止准则。
3. 对 stochastic duration 构造论文规模的专门场景，分别测量 smoothing、tree growth 和 chance-risk 误差。
4. 若需要多步在线执行，应维护 episode-level remaining risk budget 和跨 replanning 的 duration progress；当前 exact online session 对不安全的配置显式拒绝。
