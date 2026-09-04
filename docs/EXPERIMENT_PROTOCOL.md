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

DARP 从 `rddl/duration.json` 读取固定单位 duration，从 `rddl/risk.json` 的 `budget + risky_states` 读取 CC-POMDP 风险约束，并以 instance RDDL 的 horizon 作为 duration 阈值。RAO* 使用相同数值的 action-depth horizon。

DARP 的 terminal action node 使用论文 HILP 的 Manhattan replacement，RAO* 保持其原生的 step-cost 加 depth-`h` child Manhattan backup。两端执行相同动作数并共享 T/O/risk/duration，但 native objective 的边界定义不同，因此表中 objective 不能直接解释为共同 policy-quality 指标。

## 3. 三个实验文件

```text
darp_runner.py     DARP 输入路径、Manhattan heuristic 与一次求解调用
raostar_runner.py  固定仓库下载与校验、外部 Grid 和 RAO* adapter 调用
run.py             参数矩阵、配对执行、CSV 和 Markdown 汇总
```

Grid domain 使用每个动作只采样一次的 `move_outcome` intermediate fluent。`grid_row'`、`grid_col'`、`row_mod5'` 和 `col_mod5'` 共享该结果，因此保持论文中的 `.85/.075/.075` 联合滑移分布，同时不把随机结果放入 belief state。RDDL 声明的确定性初态由 DARP 直接读取，不需要外部 root-belief provider。

`darp_runner.py` 不重复实现模型或执行逐状态等价性检查；正式 domain 已针对外部 `GridInstance` 的全部 5×5 状态和动作验证 transition、observation、reward 与 risk 一致。

外部仓库必须位于固定 commit、worktree clean 且包含必要文件。自动缓存只做首次 detached checkout，已有目录不会被 pull 或 reset。DARP 不复制或修改 baseline 算法；RAO* 始终由外部 `raostar_adapter.run_raostar()` 执行。

## 4. 执行与输出

单配置：

```bash
.venv/bin/python -m experiments.DARP-vs-RAOstar-grid.run \
  --instance experiments/DARP-vs-RAOstar-grid/rddl/instance_5_h3.rddl \
  --trials 1 \
  --output output/DARP-vs-RAOstar-grid/smoke.csv
```

完整 24-cell × 25-trial 矩阵：

```bash
bash tools/run_repro.sh
# 仅继续同一版本、同一配置的中断实验：
RESUME=1 bash tools/run_repro.sh
```

`run.py` 同时生成 long-form CSV 和 Markdown 均值表。默认结果由 Git 记录在 `output/DARP-vs-RAOstar-grid/`。离线运行可指定 `--constrained-pomdp-repo`、`--raostar-checkout` 和 `--baseline-cache`。

正式 completion-time 实验不设置 timeout；调试时的 `--timeout` 同时传给 DARP 和 RAO*。DARP 的计时覆盖完整 `choose_action()`，包括 Gurobi model 创建、增量更新和求解；RDDL 与 sidecar 加载不计时。RAO* 的计时覆盖其 `search()` 调用，与固定 baseline adapter 的定义一致。

`complete` 表示算法自然收敛、Gurobi 在 `MIPGap=1e-6` 下返回 `OPTIMAL`，且风险不超过预算加 Gurobi 默认的 `1e-6` 线性约束可行性容差；它不表示 zero-gap 或有理数复核。CSV 保留原始浮点 risk，不会截断容差内的轻微超限。较严格的 gap 避免约 200 的 Grid objective 因新版 Gurobi 在默认相对容差内提前停止而影响论文表格的两位小数复现。Gurobi 线程数使用默认设置。DARP 的 `n` 是 `expanded+frontier` action histories，RAO* 的 `n` 是 belief hypergraph nodes；`iterations` 也分别表示 p-ILP solves 和 RAO* expansions，二者只能作为各自实现的搜索规模指标。
