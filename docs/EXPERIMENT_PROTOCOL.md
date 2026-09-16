# Grid 实验协议

本文记录 Table 1 duration 和 Table 2 DARP vs RAO* 的配置、指标及比较边界。使用方法统一见 README 的[实验命令](../README.md#实验)、[RDDL 扩展](../README.md#rddl-扩展)及[策略保存与回放](../README.md#策略保存与回放)。

## 模型与实验矩阵

两个实验各自使用目录内的 `rddl/domain.rddl` 和 instance，输入互不依赖。共同配置如下：

| 项目 | 配置 |
|:--|:--|
| 起点 / 目标 | `(size-1,0)` / `(0,size-1)` |
| 动作与转移 | L、U、R、D；意图方向概率 0.85，两侧滑移各 0.075 |
| 观测 | 相邻边界墙数 0/1/2；正确概率 0.85 |
| Cost / heuristic | 非目标状态 cost 1，目标状态 cost 0；Manhattan distance |
| Horizon / risk budget | `h ∈ {3,4,5,6}`，`Δ ∈ {0.1,0.2,0.3}` |
| 风险位置 | 5×5 模板 `(0,0),(3,0),(3,1),(1,3),(1,4)`；100×100 按 `(row mod 5,col mod 5)` 平铺 |

Grid 每步只采样一次 `move_outcome` intermediate fluent，行列和模 5 坐标共享该结果，保持联合滑移分布。确定性初态直接来自 RDDL。Risk 表示整条策略至少进入一次危险状态的概率，包含初态，重复进入不重复计数。

Table 1 使用 5×5 Grid，比较 Full-ILP 与 HILP。F 的时长恒为 1；E 的泥地动作均值为 2、其余为 1，使用方差为 0 的代表性均值模型；S 使用相同均值、Normal 方差 0.1 和 percentile 概率阈值 0.3。泥地位置为 `(0,3),(1,1),(2,2),(3,4),(4,2)`，“接触泥地”指源状态或意图终点位于泥地。`h` 是 duration 阈值，E/S 使用规划中的平滑 belief 停止量。每轮完整矩阵有 27 条 Full-ILP 和 36 条 HILP 记录，`h=6` 只运行 HILP；Full-ILP 的预处理上限为 800,000 条 action record，触发时记录失败。

作者 E/S 原始实验 artifact 和 E 的物理方差尚不明确，当前 E/S 数值与原表存在差异；这些结果用于报告上述明确配置下的实验，不宣称完整复现 Table 1。

Table 2 使用 5×5、100×100 Grid，共 24 个配置，分别运行 DARP-HILP 与 RAO*。Duration 恒为 1，DARP 的 duration horizon 与 RAO* 的 action-depth horizon 对齐。两端共享 T/O/risk/duration，但 DARP 在 terminal action node 使用 Manhattan replacement，RAO* 保留 step-cost 加 depth-`h` child Manhattan backup；native objective 的边界定义不同，不能直接作为共同的策略质量指标。

## 外部 baseline

Table 2 的场景和 adapter 固定为 `ME-Msc/Constrained-POMDP@d84d099493b973a63d879255d2221c1930d649aa`，RAO* 固定为 `ME-Msc/RAOStar@543f782d80ceb9555130e911c1fcf7074153d267`。RAO* 是第三方 reimplementation，不是原作者 artifact；固定提交均无顶层 LICENSE，使用其代码发表或分发前需核查授权。

`darp_runner.py` 配置 RDDL、Manhattan heuristic 并调用 DARP；`raostar_runner.py` 校验固定提交、clean worktree 和必要文件，通过外部 `raostar_adapter.run_raostar()` 执行 baseline；`run.py` 配对运行并汇总。自动缓存仅首次 detached checkout，已有目录不会 pull/reset，也不修改 baseline 算法。离线 checkout 和缓存参数见 README。

## 指标与计时

| 指标 | 定义 |
|:--|:--|
| `objective` | DARP 为 `-achieved_utility`，转成 cost 符号；RAO* 为其 native objective |
| `risk` | 求解器报告的首次风险概率，CSV 保留原始浮点值，不裁剪超预算值 |
| `time_s` | DARP 为完整 `choose_action()`，包括树构建、Gurobi model 创建、更新和求解，不含 RDDL 加载；RAO* 为固定 adapter 的 `search()` |
| Table 1 节点数 | Full-ILP `n` 为 `tree_nodes`；`Act.n` 和 HILP `Exp.n` 为各自 `ilp_variables`；`Exp.% = 100 × Exp.n / Act.n` |
| Table 2 `n` | DARP 为 `expanded_nodes + frontier_nodes` action histories；RAO* 为 belief hypergraph nodes |
| `iterations` | HILP 为 p-ILP solves，Full-ILP 为 1，RAO* 为 expansions；仅表示各实现的搜索规模 |
| `policy_execution_time_s` | `rollout_time_s / episodes`，每条 episode 的环境 reset、策略查询、`env.step` 与统计墙钟开销，不含规划时间 |

正式 completion-time 实验不设 timeout；调试 Table 2 时，同一个 `--timeout` 传给两端。DARP 使用 `MIPGap=1e-6`、Gurobi 默认 `1e-6` 线性约束可行性容差及默认线程数。`complete` 表示求解器报告搜索完成，不表示 zero-gap、有理数复核或实验脚本独立证明可行性。

每个 DARP trial 保存完整 `DARPResult` JSON，重新载入策略后，在相同 RDDL 的 pyRDDLGym 环境执行，默认 1000 条 episode。`risk_rate` 是至少失败一次的 episode 比例，风险后继续执行策略；有限采样略超预算不等于模型约束被违反。风险和 duration 统计读取真实状态，策略仍只收到 observation。

`physical_duration_mean` 是真实轨迹累计动作时长的均值：F/E 使用配置的确定性时长，S 使用独立随机数流采样 Normal；它与 E/S 规划停止量及执行墙钟时间含义不同。策略按叶节点或环境终止条件停止。BaseAgent 的 `statistics["mean"]` 等回报字段是原始 RDDL discounted reward，不应用 Manhattan terminal replacement，不能与规划 objective 混用。RAO* 只报告搜索指标，没有 DARP executor rollout，两端不比较执行墙钟时间。

## 输出与续跑

当前结果见 [Table 1](../experiments/DARP-table1-grid/output/table1.md) 和 [Table 2](../experiments/DARP-vs-RAOstar-grid/output/table2.md)；对应 long-form CSV 与策略 JSON 保存在各自 `output/` 下。Markdown 从 CSV 汇总，供手动对照原文，执行附表保留逐 trial 指标。

Runner 默认 25 个规划 trial，当前两张结果表使用每配置 1 个 trial；单次时间是一次观测，不是论文的 25 次均值，不能据此断言稳定速度优势。每策略 1000 条 episode 是执行采样次数，不是独立规划次数。多 trial 时，Table 1 对成功记录求均值并报告失败数；Table 2 要求完整配对 trial。

`--resume` 仅用于继续同一代码版本、RDDL、配置、seed、trial 和 episode 设置的中断实验。CSV schema、episode 数和策略文件检查不能替代源码或 RDDL 哈希校验；旧版输出不可混入当前运行。Table 1 已记录的失败 trial 也会被跳过，续跑不会自动重试。策略 JSON 不绑定模型路径或哈希，执行方需保留匹配的 RDDL。
