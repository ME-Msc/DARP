# Grid 实验协议

本文记录 Table 1 duration 和 Table 2 DARP vs RAO* 的配置、指标及比较边界。使用方法统一见 README 的[实验命令](../README.md#实验)、[RDDL 扩展](../README.md#rddl-扩展)及[策略保存与回放](../README.md#策略保存与回放)。

## 模型与实验矩阵

输入按 duration 类型统一放在 `benchmarks/grid/`，算法从公共入口运行。共同配置如下：

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

`experiments/run.py` 调用公共 solve_rddl；`experiments/grid/raostar.py` 校验固定源码版本并调用外部算法；`experiments/compare.py` 读取独立运行批次，生成 LaTeX/PDF。使用方法见[实验入口](../experiments/README.md)。

## 指标与计时

| 指标 | 定义 |
|:--|:--|
| `objective` | DARP 为 `-achieved_utility`，转成 cost 符号；RAO* 为其 native objective |
| `risk` | 求解器报告的首次风险概率，CSV 保留原始浮点值，不裁剪超预算值 |
| `time_s` | DARP 为完整 `choose_action()`，包括树构建、Gurobi model 创建、更新和求解，不含 RDDL 加载；RAO* 为固定 adapter 的 `search()` |
| Table 1 节点数 | Full-ILP `n` 为 `tree_nodes`；`Act.n` 和 HILP `Exp.n` 为各自 `ilp_variables`；`Exp.% = 100 × Exp.n / Act.n` |
| Table 2 `n` | DARP 为 `expanded_nodes + frontier_nodes` action histories；RAO* 为 belief hypergraph nodes |
| `iterations` | HILP 为 p-ILP solves，Full-ILP 为 1，RAO* 为 expansions；仅表示各实现的搜索规模 |
| `rank_alpha / rank_lambda` | HILP 风险评分权重／保留比例（0<λ≤1）；先保护高分 F，再自底向上删未保护 E 子树；默认 1/1，Full-ILP 不筛选；RAO* 不适用 |
| `ilp_variables / full_partial_ilp_variables` | 最后一次实际提交的变量数／完整当前 p-ILP 变量数；不是全 horizon 已枚举树 |
| `rank_filter_ms / rank_fallbacks` | 搜索期间累计筛选耗时／受限不可行后恢复完整 p-ILP 的次数；Full-ILP 为 0，RAO* 不适用，留空 |
| `policy_execution_time_s` | `rollout_time_s / episodes`，每条 episode 的环境 reset、策略查询、`env.step` 与统计墙钟开销，不含规划时间 |

新运行入口默认时限 120 秒，`--timeout` 同时作用于两端；历史无时限数据不视为同配置计时。DARP 使用 `MIPGap=1e-6`、Gurobi 默认 `1e-6` 线性约束可行性容差及默认线程数。`complete` 表示求解器报告搜索完成，不表示 zero-gap、有理数复核或实验脚本独立证明可行性。

启用 Rank-based 后，两套实验仍要求 DARP 策略完整可执行且风险可行，但允许 `decision.complete=False` 并原样记录，不能把受限最优解当作全局最优解。策略 JSON 的 `decision.policy.complete` 对应 Python 的 `duration_complete`，与 CSV 及外层 `decision.complete` 的全局搜索认证分别记录。`time_s` 包括筛选与受限 Gurobi 模型重建；Table 2 另列实际／完整当前 ILP 变量数，原有 `n` 仍统计完整搜索历史。当前 Rank 实验固定 `alpha=1`，比较 `lambda=0.3,0.5,0.7,0.9`；不能只凭变量下降或单次计时声称优化成功。

新入口为每个 DARP trial 保存完整 `DARPResult` JSON，重新载入后回放；命令行默认 10 episodes，正式矩阵配置指定 1000 episodes。`risk_rate` 是至少失败一次的 episode 比例，风险后继续执行策略；有限采样略超预算不等于模型约束被违反。风险和 duration 统计读取真实状态，策略仍只收到 observation。

Rank 矩阵为 5×5/100×100、h=3/4/5/6、Delta=0.1/0.2/0.3、lambda=0.3/0.5/0.7/0.9、alpha=1，每配置 3 trials，合计 648 条。历史 HILP、E+F、F-only 结果分别迁入 `experiments/grid/runs/pruning-HILP`、`pruning-pruningEF`、`pruning-pruningF`。原 metadata 保留首次与补跑两批来源；未伪造新的计时。旧报告作为 historical 保存，新报告通过 compare.py 生成。

新 LaTeX 表按实际输入内容匹配，分别列出算法、alpha/lambda 和运行批次；统计中位数，失败/缺失显示 --，不丢弃失败 trial。Cost increase 为 100*(C-C_HILP)/abs(C_HILP)，零或非唯一基准不计算；RAO* 的 native objective 不用于该增量。加粗表示 cost 变化；下划线只用于完整设置和 trial/seed 一致的更低耗时。历史缺失环境参数不补造、不用于自动声称加速。

`physical_duration_mean` 是真实轨迹累计动作时长的均值：F/E 使用配置的确定性时长，S 使用独立随机数流采样 Normal；它与 E/S 规划停止量及执行墙钟时间含义不同。策略按叶节点或环境终止条件停止。BaseAgent 的 `statistics["mean"]` 等回报字段是原始 RDDL discounted reward，不应用 Manhattan terminal replacement，不能与规划 objective 混用。RAO* 只报告搜索指标，没有 DARP executor rollout，两端不比较执行墙钟时间。

## 输出与续跑

每个 `experiments/<场景>/runs/<批次>/` 保存 config.json、results.csv 和 policies/。迁移了 2013 条历史记录、303 个策略文件，保留原始数值列和来源；其中 legacy 数据与当前版本不混为新实验。迁移清单见 `experiments/grid/migration.json`。

展示文件生成到 `experiments/output/<场景>-<算法名>/`，剪枝对比使用 `grid-HILP-pruningF-pruningEF`。每个表格保存 .tex、.pdf、.selection.json；selection 记录来源批次和筛选条件。辅助编译文件进入临时目录。

默认每配置 1 trial，正式剪枝配置为 3 trials；回放次数与规划重复次数不同。--resume 仅补尚未记录的 trial，输入内容、配置、seed 和运行环境必须匹配；失败 trial 不自动重试。策略 JSON 不绑定 RDDL，执行方负责提供匹配环境。

旧参数扫描命令由 experiments/run.py 的 --alpha、--lambdas 和 --algorithms 统一替代。tools/run_repro.sh 使用 RUN_NAME 选择批次、TRIALS 设置重复次数，RESUME=1 续跑。
