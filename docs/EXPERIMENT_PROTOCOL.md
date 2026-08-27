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

DARP 从 `rddl/duration.json` 读取 duration，并以 instance RDDL 的 horizon 作为 duration 阈值。RAO* 只有 action-depth horizon，因此比较前强制检查 `FixedDurationModel`、所有动作时长为 1、`zeta=0`、chance constraint，以及两端 horizon 相等；其他 duration 配置直接拒绝比较。

DARP 的 terminal action node 使用论文 HILP 的 Manhattan replacement，RAO* 保持其原生的 step-cost 加 depth-`h` child Manhattan backup。两端执行相同动作数并共享 T/O/risk/duration，但 native objective 的边界定义不同，因此表中 objective 不能直接解释为共同 policy-quality 指标。

## 3. 三个实验文件

```text
darp_runner.py     DARP 调用、Manhattan、root belief、RDDL/外部模型等价性检查
raostar_runner.py  固定仓库下载与校验、外部 Grid 和 RAO* adapter 调用
run.py             参数矩阵、配对执行、CSV 和 Markdown 汇总
```

标准 RDDL 的随机 CPF 独立采样。Grid domain 使用隐藏 `noise∈{0,1,2}` 保持 row/column 滑移相关；`darp_runner.py` 将 instance 的 `noise=0` 占位初态转换为 `.85/.075/.075` root belief。CLI 和实验调用同一个 provider。

每个 instance 在计时外检查：

- 初始 position/noise belief；
- action 顺序与 codec；
- 所有有限时域可达状态的 transition、observation、reward、risk 和 Manhattan；
- goal 的 terminal、自环、零 reward 与零 heuristic；
- duration、constraint type 和 horizon。

外部仓库必须位于固定 commit、worktree clean 且包含必要文件。自动缓存只做首次 detached checkout，已有目录不会被 pull 或 reset。DARP 不复制或修改 baseline 算法；RAO* 始终由外部 `raostar_adapter.run_raostar()` 执行。

## 4. 执行与输出

单配置：

```bash
.venv/bin/python -m experiments.DARP-vs-RAOstar.run \
  --instance experiments/DARP-vs-RAOstar/rddl/instance_5_h3.rddl \
  --trials 1 \
  --output output/DARP-vs-RAOstar/smoke.csv
```

完整 24-cell × 25-trial 矩阵：

```bash
bash tools/run_repro.sh
```

`run.py` 同时生成 long-form CSV 和 Markdown 均值表。默认路径位于被 Git 忽略的 `output/DARP-vs-RAOstar/`。离线运行可指定 `--constrained-pomdp-repo`、`--raostar-checkout` 和 `--baseline-cache`。

正式 completion-time 实验不设置 timeout；调试时的 `--timeout` 同时传给 DARP 和 RAO*。模型加载、parity、外部 import 和 Gurobi warm-up 都不计入 planner time。

`complete` 表示算法自然结束并返回可行完整策略；`certified` 是 DARP 更严格的系数精确性证书，RAO* 没有同义字段。DARP 的 `n` 是 `expanded+frontier` action histories，RAO* 的 `n` 是 belief hypergraph nodes；`iterations` 也分别表示 p-ILP solves 和 RAO* expansions，二者只能作为各自实现的搜索规模指标。
