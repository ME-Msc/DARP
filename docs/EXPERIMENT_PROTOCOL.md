# DARP vs RAO* 实验协议

## 1. 目的与外部实现

实验比较 DARP-HILP 与外部 RAO* 在同一部分可观测 Grid CC-POMDP 上的 native objective、first-entry chance risk、搜索时间、节点数和迭代数。

外部场景和 adapter 来自 `ME-Msc/Constrained-POMDP@d84d099493b973a63d879255d2221c1930d649aa`，RAO* 来自 `ME-Msc/RAOStar@543f782d80ceb9555130e911c1fcf7074153d267`。后者是 reimplementation，不是 RAO* 原作者 artifact。两个提交目前均无顶层 LICENSE；发表时必须披露来源并单独核查授权。

## 2. 固定模型

```text
size                  5×5, 100×100
horizon               3, 4, 5, 6
risk budget delta     0.1, 0.2, 0.3
start / goal          (size-1,0) / (0,size-1)
actions               L, U, R, D
transition            intended .85, slips .075/.075
observation           boundary-wall count 0/1/2; correct .85
cost                  1 at non-goal, 0 at goal
duration              deterministic 1
heuristic             Manhattan distance to goal
```

5×5 风险模板为 `(0,0),(3,0),(3,1),(1,3),(1,4)`，100×100 按 `(row mod 5,col mod 5)` 平铺。risk 是执行中首次进入危险状态的概率。

DARP 的 domain duration 表达式在 Table 2 默认 non-fluent 下化简为 `D(s,a)=1.0`，`rddl/risk.json` 的 `budget + risky_states` 给出 CC-POMDP 风险约束，instance RDDL 的 horizon 是 duration 阈值。RAO* 使用相同数值的 action-depth horizon。

DARP 的 terminal action node 使用论文 HILP 的 Manhattan replacement，RAO* 保持其原生的 step-cost 加 depth-`h` child Manhattan backup。两端执行相同动作数并共享 T/O/risk/duration，但 native objective 的边界定义不同，因此表中 objective 不能直接解释为共同 policy-quality 指标。

## 3. 三个实验文件

```text
darp_runner.py     DARP 输入路径、Manhattan heuristic 与一次求解调用
raostar_runner.py  固定仓库下载与校验、外部 Grid 和 RAO* adapter 调用
run.py             参数矩阵、配对执行、CSV 和 Markdown 汇总
```

Grid domain 使用每个动作只采样一次的 `move_outcome` intermediate fluent。`grid_row'`、`grid_col'`、`row_mod5'` 和 `col_mod5'` 共享该结果，因此保持论文中的 `.85/.075/.075` 联合滑移分布，同时不把随机结果放入 belief state。RDDL 声明的确定性初态由 DARP 直接读取，不需要外部 root-belief provider。

`darp_runner.py` 只配置输入并调用 DARP。此前的模型等价性与策略重算检查属于独立历史审计，不在当前实验运行中重复执行。

外部仓库必须位于固定 commit、worktree clean 且包含必要文件。自动缓存只做首次 detached checkout，已有目录不会被 pull 或 reset。DARP 不复制或修改 baseline 算法；RAO* 始终由外部 `raostar_adapter.run_raostar()` 执行。

## 4. 执行与输出

单配置：

```bash
.venv/bin/python -m experiments.DARP-vs-RAOstar-grid.run \
  --instance experiments/DARP-vs-RAOstar-grid/rddl/instance_5_h3.rddl \
  --trials 1 --episodes 1000 \
  --output experiments/DARP-vs-RAOstar-grid/output/smoke.csv
```

完整 24-cell 矩阵，每个 cell 的 DARP-HILP 和 RAO* 各求解 1 次，每个 DARP 策略执行 1000 条 episode：

```bash
TRIALS=1 bash tools/run_repro.sh
# 仅继续同一版本、同一配置的中断实验：
TRIALS=1 RESUME=1 bash tools/run_repro.sh
```

`run.py` 同时生成 long-form CSV 和按原表结构排版的 Markdown。`--trials 1` 时规划时间就是一次求解的观测值，不是论文 25 次 trial 的均值，也不能据此断言稳定的速度优势。默认结果由 Git 记录在 `experiments/DARP-vs-RAOstar-grid/output/`。离线运行可指定 `--constrained-pomdp-repo`、`--raostar-checkout` 和 `--baseline-cache`。

每个 DARP trial 保存完整 `DARPResult` JSON，再从 JSON 读取策略，在同一 RDDL 输入构建的 pyRDDLGym 环境中调用 `agent.evaluate(env, episodes=1000, seed=seed, risk_path=RISK)`。策略格式与统计字段见 [策略格式](POLICY_FORMAT.md)。实验代码负责生成表格，原论文数值由读者手动对照。

主表 objective 和 risk 来自求解器，附表仅记录 episode 数、首次风险频率、平均物理时长和执行墙钟时间。`statistics["mean"]` 等 BaseAgent 字段仍返回原始 RDDL discounted reward，执行阶段不应用 Manhattan terminal replacement，因此这些回报不与规划 objective 混用。

CSV 的 `policy_execution_time_s` 为 `rollout_time_s / episodes`，表示平均每条 episode 的环境 reset、策略查询、`env.step` 及统计开销；它不包含规划时间，也不是动作的物理时长。外部 RAO* 只运行搜索并报告原生规划指标，未通过 DARP executor 做策略 rollout；两端不比较策略执行墙钟时间。

Risk 每条 episode 只记录“是否至少失败过一次”：包括初始状态属于风险集的情况，多次进入或停留在风险状态仍只计一次，发生风险后继续执行原策略。`risk_rate` 是采样频率，有限采样的频率略超预算不等于策略违反模型约束。风险统计使用 simulator 的真实状态，但策略仍只收到其定义的 observation。

正式 completion-time 实验不设置 timeout；调试时的 `--timeout` 同时传给 DARP 和 RAO*。DARP 的计时覆盖完整 `choose_action()`，包括 Gurobi model 创建、增量更新和求解；RDDL 与 risk JSON 加载不计时。RAO* 的计时覆盖其 `search()` 调用，与固定 baseline adapter 的定义一致。

`complete` 表示对应求解器报告搜索完成，不代表实验脚本重新证明了策略可行性。DARP 使用 `MIPGap=1e-6` 和 Gurobi 默认的 `1e-6` 线性约束可行性容差；这不表示 zero-gap 或有理数复核。两端 CSV 都保留原始浮点 risk，不裁剪超预算值。较严格的 gap 避免约 200 的 Grid objective 因默认相对容差提前停止而影响两位小数。Gurobi 线程数使用默认设置。DARP 的 `n` 是 `expanded+frontier` action histories，RAO* 的 `n` 是 belief hypergraph nodes；`iterations` 分别表示 p-ILP solves 和 RAO* expansions，只作为各自实现的搜索规模指标。

## 5. Table 1 duration 实验

`experiments/DARP-table1-grid/` 共用 Grid domain 和 risk 文件，用 instance 的 non-fluents 指定 F、E、S。F 为所有动作时长 1；E 按论文文字设置接触泥地均值 2、其他均值 1，当前使用方差为 0 的代表性均值模型，作者原实验的 E 物理方差不明确；S 使用相同均值和 Normal 方差 0.1，percentile 阈值为 0.3。对 $h\in\{3,4,5,6\}$、$\Delta\in\{0.1,0.2,0.3\}$ 运行 Full-ILP/HILP，其中 $h=6$ 按原表只运行 HILP：

```bash
.venv/bin/python -m experiments.DARP-table1-grid.run --trials 1 --episodes 1000 \
  --summary experiments/DARP-table1-grid/output/table1.md
```

该命令产生 27 条 Full-ILP 和 36 条 HILP 规划记录；1000 条 episode 是每次所得策略的执行次数，不是独立规划 trial。Full-ILP 保留 800,000 action-record 预处理上限，触发资源上限时记录失败。

执行器按每步真实状态读取原 RDDL duration：F/E 累计当前配置的确定性时长，S 采样 Normal 时长，汇总为 `physical_duration_mean`。独立随机数流保持环境 T/O 抽样不变。物理时长与论文使用平滑 belief 的 E/S 停止量含义不同；策略按规划结果的叶节点停止。

结果保存在 [Table 1](../experiments/DARP-table1-grid/output/table1.md) 和 [Table 2](../experiments/DARP-vs-RAOstar-grid/output/table2.md)，供手动对照。当前缺少作者 E/S 实验的原始 artifact，已有部分数值不一致，不能宣称完整复现 Table 1。例如 E/h=4/Δ=0.1 为 9.62（论文 9.66），S/h=5/Δ=0.1 为 10.63（论文 10.50）。泥地 duration 当前按“源状态或意图终点位于泥地”定义，核对这一解释仍需作者原始配置。

旧版输出可能包含不同的执行字段；恢复实验仅使用同一版本、配置和 trial 设置的结果，避免混用计时口径。

2026-09-14 的精简只删除了当前 `table1-raw.csv`、`table2-raw.csv` 的诊断列并重新排版；63 条和 48 条原记录的保留值全部未变，计时仍来自此前运行，不是精简后重新测量。原 CSV 备份在项目内 `.cache/metric-cleanup-20260914/`；保存的策略 JSON 未修改。
