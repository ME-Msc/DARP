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

## Heuristic 与 duration

HILP 可通过 `--heuristic package.module:OBJECT` 加载外部 `UtilityHeuristic`。回调返回 utility-to-go；若计算的是 cost-to-go，应返回其负值。history probability 由核心统一加权：

$$
h_q=\sum_s \rho(q)b_q(s)h(s,a_q).
$$

Duration 不写入 RDDL，必须通过 `--duration` 指定 JSON sidecar。支持 fixed、state-dependent、chance 和 Gaussian duration。fixed duration 使用以下格式：

```json
{
  "kind": "fixed",
  "default": 1.0,
  "actions": {
    "slow_action": 2.0
  }
}
```

`default` 是所有动作的有限正时长；可选的 `actions` 按 grounded action label 覆盖个别动作，统一时长时直接省略。horizon 由 instance RDDL 提供。论文规定 fixed duration 的 $\varsigma=0$，因此该格式不接受 `zeta`。

单独运行或调试 DARP 时，必须同时指定 domain、instance 和 duration：

```bash
.venv/bin/python -m darp \
  --domain experiments/DARP-vs-RAOstar-grid/rddl/domain.rddl \
  --instance experiments/DARP-vs-RAOstar-grid/rddl/instance_5_h3.rddl \
  --duration experiments/DARP-vs-RAOstar-grid/rddl/duration.json \
  --heuristic experiments.DARP-vs-RAOstar-grid.darp_runner:MANHATTAN \
  --root-belief experiments.DARP-vs-RAOstar-grid.darp_runner:initial_belief \
  --terminal-heuristic \
  --output output/DARP-vs-RAOstar-grid/darp.json
```

`.vscode/launch.json` 使用同一入口和显式文件路径。

## DARP vs RAO* 实验

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

单实例模式从 RDDL 读取网格大小与 horizon，并从 `rddl/duration.json` 读取默认 risk budget；命令行只保留 trial、timeout、seed 和输出等执行选项。批量 Table 2 实验中的 `size/horizon/delta` 列表仍是实验矩阵筛选器，每个被选实例的实际模型参数都会再次从 RDDL 校验。

正式实验会运行完整参数矩阵、逐 trial 保存并支持断点续跑：

```bash
bash tools/run_repro.sh
```

结果写入被 Git 忽略的 `output/DARP-vs-RAOstar-grid/`。已有本地 checkout 或离线运行时，可选设置 `CONSTRAINED_POMDP_REPO`、`RAOSTAR_CHECKOUT` 和 `BASELINE_CACHE`；本地 checkout 必须处在固定 commit 且 worktree clean。外部实现的 provenance、指标定义和计时边界见 [实验协议](docs/EXPERIMENT_PROTOCOL.md)。算法公式与代码对应见 [算法映射](docs/ALGORITHM_MAPPING.md)。

新增 RDDL 对比场景时可以复用同一 RAO* 缓存，但仍需在新的实验目录中提供该场景到 RAO* model API 的薄适配和等价性检查；不需要修改 DARP 的解析器、HILP、ILP 或 Gurobi 实现。

## 核心代码

```text
RDDL + duration/risk sidecar
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
    planner="hilp",
    heuristic=MY_HEURISTIC,
)
```

只有能够证明 heuristic 是最大化 utility 的上界时，才能设置 `upper_bound=True`；HILP 用该标志判断 frontier 清空后是否可以正常结束。普通场景不要添加 `--terminal-heuristic`，因为它会在 duration 边界用 heuristic 替换 RDDL 叶节点 reward，仅适用于明确采用这种终端估值定义的实验。
