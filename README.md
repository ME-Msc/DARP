# DARP

DARP 是论文 *Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions* 的研究实现。它从 RDDL 按需构建有限可达模型，采用与论文参考实现一致的稀疏浮点概率质量传播 belief/risk，并使用 Gurobi 求解 full-ILP 或增量 HILP。

仓库目前只保留核心求解器和 `DARP vs RAO*` 验证实验。

## 安装

```bash
bash tools/install.sh
```

需要 CPython 3.12.3、`requirements-lock.txt` 中的依赖以及有效的 Gurobi 许可证。查看通用求解入口：

```bash
.venv/bin/python -m darp --help
```

## Heuristic、duration 与 risk

HILP 可通过 `--heuristic package.module:OBJECT` 加载外部 `UtilityHeuristic`。回调返回 utility-to-go；若计算的是 cost-to-go，应返回其负值。history probability 由核心统一加权：

$$
h_q=\sum_s \rho(q)b_q(s)h(s,a_q).
$$

Duration 和 risk 不写入 RDDL，分别通过 `--duration` 与 `--risk` 指定 JSON sidecar。Duration 支持 fixed、state-dependent、chance 和 Gaussian；fixed 使用以下格式：

```json
{
  "kind": "fixed",
  "default": 1.0,
  "actions": {
    "slow_action": 2.0
  }
}
```

`default` 是所有动作的有限正时长；可选的 `actions` 按 grounded action label 覆盖个别动作，统一时长时可写成空对象。horizon 由 instance RDDL 提供。论文规定 fixed duration 的 $\varsigma=0$，因此该格式不接受 `zeta`。

Risk 使用独立 JSON sidecar，直接定义 CC-POMDP 的预算 $\Delta$ 和风险状态集合 $R$。例如：

```json
{
  "budget": 0.1,
  "risky_states": [
    {"unsafe": true},
    {"phase": 2, "blocked": true}
  ]
}
```

`budget` 是 `[0,1]` 内的概率；`risky_states` 中每个对象是一个部分状态 selector：对象内的 fluent 等式同时满足（AND）即匹配，任一对象匹配（OR）即属于 $R$。selector 支持当前有限 kernel 使用的 Boolean 和 integer fluent，因此既可表达 `{"unsafe": true}`，也可表达 Grid 位置、阶段或模式组合。初始状态属于 $R$ 的质量会先从预算扣除，之后只统计首次进入 $R$ 的概率。若危险由动作或转移触发，应由 RDDL CPF 更新 `unsafe` 等状态 fluent，再由 `risky_states` 引用；risk sidecar 不重复定义转移逻辑。`--risk-budget` 可覆盖文件预算，但不会改变风险集合。

单独运行或调试 DARP 时，必须同时指定 domain、instance、duration 和 risk：

```bash
.venv/bin/python -m darp \
  --domain experiments/DARP-vs-RAOstar-grid/rddl/domain.rddl \
  --instance experiments/DARP-vs-RAOstar-grid/rddl/instance_5_h3.rddl \
  --duration experiments/DARP-vs-RAOstar-grid/rddl/duration.json \
  --risk experiments/DARP-vs-RAOstar-grid/rddl/risk.json \
  --heuristic experiments.DARP-vs-RAOstar-grid.darp_runner:MANHATTAN \
  --root-belief experiments.DARP-vs-RAOstar-grid.darp_runner:initial_belief \
  --terminal-heuristic \
  --output output/DARP-vs-RAOstar-grid/darp.json
```

`.vscode/launch.json` 使用同一入口和显式文件路径。

## DARP vs RAO* 实验

仓库包含三个互相独立的成对实验；每个目录都只有 DARP runner、RAO* runner、配对汇总和 RDDL sidecar，不修改 DARP 核心：

| 目录 | 场景与来源 | 结果 |
|:---|:---|:---|
| `DARP-vs-RAOstar-grid` | HILP 论文 Grid | `output/DARP-vs-RAOstar-grid/` |
| `DARP-vs-RAOstar-science-agent` | 公开 RAO* Science Agent source test | `output/DARP-vs-RAOstar-science-agent/` |
| `DARP-vs-RAOstar-power-supply` | 基于 Thiébaux/Cordier PSR 的自建三线简化实例 | `output/DARP-vs-RAOstar-power-supply/` |

Science Agent 和 Power Supply 都调用 compatibility fork `ME-Msc/rao-star@f51bfdc1ff8f` 中未改写的 `RAOStar` 算法类；该 fork 为现代 RMPyL 增加兼容层，并为 Science model 增加等价的 duration 名称 alias。首次运行会自动缓存该仓库，也可用 `--raostar-checkout` 指定 clean checkout。两边计时前会检查完整有限可达模型的 transition、observation、reward、risk、heuristic 和 terminal 语义。

```bash
.venv/bin/python -m experiments.DARP-vs-RAOstar-science-agent.run
.venv/bin/python -m experiments.DARP-vs-RAOstar-power-supply.run
```

Science Agent 严格匹配公开代码中的 `perform_scheduling=False` 模式；[Benazera et al. 2005](https://aiweb.cs.washington.edu/ai/planning/papers/mausam-ijcai05.pdf) 是其连续资源 rover 原型，并未给出 RAO* Table 1 的具体参数，因此当前实验不是 PARIS 时间窗复现。

RAO* 论文的 PSR Table 2 使用 semi-rural 网络，但公开 RAO* 仓库缺少该 model、fault instances、sensor placement 和 horizon。Power Supply 实验因此采用 [Bonet & Thiébaux 2003](https://users.cecs.anu.edu.au/~thiebaux/papers/icaps03.pdf) 明确给出的三线例子、动作效果和 finish cost；均匀单故障先验、horizon、sensor subset 与 chance constraint 是本实验公开补充的选择。它只验证基于 [Thiébaux & Cordier 2001](https://users.cecs.anu.edu.au/~thiebaux/papers/ecp01.pdf) 的共享语义，不是原 Table 2 实例或数值复现。原始网络数据与工具索引见[官方 PSR benchmark 页面](https://users.cecs.anu.edu.au/~thiebaux/benchmarks/pds/)。

两个新实验都通过现有外部 utility-heuristic 接口配置，不修改 HILP：Science Agent 使用“所有尚未访问且存在的 discovery utility 之和”；三线 PSR 中论文式理想恢复 penalty 为 0，因此其 admissible utility upper bound 是 0，但搜索引导较弱。风险由 RDDL state 与 `risk.json` 定义，并由 HILP 的 frontier risk coefficient 统一处理，不需要场景专用 risk heuristic。

### Grid 实验

实验位于 `experiments/DARP-vs-RAOstar-grid/`，使用原论文 Table 2 的 Grid 配置，对比：

- DARP-HILP；
- 固定提交的外部 RAO* reimplementation。

实验代码只有三个入口：`darp_runner.py` 调用 DARP，`raostar_runner.py` 通过 [Constrained-POMDP](https://github.com/ME-Msc/Constrained-POMDP) 的 adapter 调用固定提交的 [RAOStar](https://github.com/ME-Msc/RAOStar)，`run.py` 配对执行并生成 CSV/Markdown。外部 RAO* 只支持 action-depth horizon，因此仅在 `D(s,a)=1`、`zeta=0` 时允许比较。首次运行会自动下载并校验两个仓库到 `.cache/baselines/`，无需手工部署。

单配置检查：

```bash
.venv/bin/python -m experiments.DARP-vs-RAOstar-grid.run \
  --instance experiments/DARP-vs-RAOstar-grid/rddl/instance_5_h3.rddl \
  --trials 1 \
  --output output/DARP-vs-RAOstar-grid/smoke.csv
```

单实例模式从 RDDL 读取网格大小与 horizon，并从 `rddl/risk.json` 读取默认 risk budget；命令行只保留 trial、timeout、seed 和输出等执行选项。批量 Table 2 实验中的 `size/horizon/delta` 列表仍是实验矩阵筛选器，每个被选实例的实际模型参数都会再次从 RDDL 校验。

正式实验会运行完整参数矩阵、逐 trial 保存并支持断点续跑：

```bash
bash tools/run_repro.sh
```

结果写入并由 Git 记录在 `output/DARP-vs-RAOstar-grid/`。已有本地 checkout 或离线运行时，可选设置 `CONSTRAINED_POMDP_REPO`、`RAOSTAR_CHECKOUT` 和 `BASELINE_CACHE`；本地 checkout 必须处在固定 commit 且 worktree clean。外部实现的 provenance、指标定义和计时边界见 [实验协议](docs/EXPERIMENT_PROTOCOL.md)。算法公式与代码对应见 [算法映射](docs/ALGORITHM_MAPPING.md)。

新增 RDDL 对比场景时可以复用同一 RAO* 缓存，但仍需在新的实验目录中提供该场景到 RAO* model API 的薄适配和等价性检查；不需要修改 DARP 的解析器、HILP、ILP 或 Gurobi 实现。

## 核心代码

```text
RDDL + duration.json + risk.json
  -> adapter       # 按需构建的稀疏浮点有限模型
  -> preprocess    # Algorithm 1
  -> expand        # Algorithm 2
  -> ilp_tree      # policy/flow/risk 约束
  -> hilp          # Algorithm 3
  -> gurobi        # 增量 p-ILP
  -> policy        # 策略提取与浮点指标汇总
```

## 自定义 Heuristic

用户可以在任意可被 Python 导入的模块中定义自己的 heuristic。例如，在项目根目录创建 `my_heuristic.py`：

```python
from darp.planning.heuristic import HeuristicInput, UtilityHeuristic


def _estimate(value: HeuristicInput) -> int:
    """Estimate utility-to-go for one grounded state/action pair."""

    row = int(value.state["grid_row"])
    col = int(value.state["grid_col"])
    goal_row = int(value.non_fluents["goal_row"])
    goal_col = int(value.non_fluents["goal_col"])

    if (row, col) == (goal_row, goal_col):
        return 0

    cost_lower_bound = abs(row - goal_row) + abs(col - goal_col)
    return -cost_lower_bound  # DARP maximizes utility, so negate cost-to-go.


MY_HEURISTIC = UtilityHeuristic(
    name="my-grid-manhattan",
    evaluate=_estimate,
    # Set True only after proving this is an upper bound on optimal utility.
    upper_bound=False,
)
```

`HeuristicInput.state` 是单个 grounded state，`action_label` 是当前动作名称，`action` 是完整的 grounded action assignment，`non_fluents` 是 RDDL 常量。回调只返回该状态和动作的 utility-to-go；不要在回调中乘 belief 或 history probability，DARP 会统一计算 $h_q=\sum_s\rho(q)b_q(s)h(s,a_q)$。

通过命令行加载：

```bash
.venv/bin/python -m darp \
  --domain path/to/domain.rddl \
  --instance path/to/instance.rddl \
  --duration path/to/duration.json \
  --risk path/to/risk.json \
  --planner hilp \
  --heuristic my_heuristic:MY_HEURISTIC \
  --output output/result.json
```

也可以通过 Python API 传入同一个对象：

```python
from darp.solve import solve_rddl
from my_heuristic import MY_HEURISTIC

result = solve_rddl(
    "path/to/domain.rddl",
    "path/to/instance.rddl",
    "path/to/duration.json",
    risk_path="path/to/risk.json",
    planner="hilp",
    heuristic=MY_HEURISTIC,
)
```

只有能够证明 heuristic 是最大化 utility 的上界时，才能设置 `upper_bound=True`；HILP 用该标志判断 frontier 清空后是否可以正常结束。普通场景不要添加 `--terminal-heuristic`，因为它会在 duration 边界用 heuristic 替换 RDDL 叶节点 reward，仅适用于明确采用这种终端估值定义的实验。
