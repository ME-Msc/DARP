# DARP

DARP 是论文 *Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions* 的离线求解器：读取标准或扩展 RDDL，通过 Full-ILP/HILP 求解并保存条件策略，在 pyRDDLGym 中回放。

- [算法映射](docs/ALGORITHM_MAPPING.md)：公式、符号、实现及支持范围。
- [实验协议](docs/EXPERIMENT_PROTOCOL.md)：配置、计时口径和比较边界。
- 结果：[Table 1](experiments/DARP-table1-grid/output/table1.md)、[Table 2](experiments/DARP-vs-RAOstar-grid/output/table2.md)。

## 安装与求解

需要 CPython 3.12.3 和有效的 Gurobi 许可证；在仓库根目录执行：

```bash
bash tools/install.sh
.venv/bin/python -m darp --help
```

以下 `domain.rddl`、`instance.rddl` 替换为你的标准或扩展 RDDL 文件。Full-ILP 用于小规模验证，HILP 见[自定义 Heuristic](#自定义-heuristic)。

### 命令行

```bash
.venv/bin/python -m darp \
  --domain domain.rddl --instance instance.rddl \
  --planner full-ilp --output result.json
```

### Python

```python
from darp.solve import solve_rddl

result = solve_rddl("domain.rddl", "instance.rddl", planner="full-ilp")
result.save("result.json")
```

默认 `timeout_s=60`，不是整个进程的硬超时，Full-ILP 预处理不计入求解时限。Full-ILP 默认最多预处理 100,000 个动作历史，超限会报错；返回结果后，用 `result.decision.complete` 判断搜索是否完成。断点调试使用 [launch.json](.vscode/launch.json)。

### 可选按比例的效用–风险候选筛选（Rank-based prototype）

`rank_alpha` 是风险评分权重，`rank_lambda` 是候选保留比例（0 < λ ≤ 1）。默认 `rank_alpha=1, rank_lambda=1`，保持原 HILP 的增量求解。启用后先保护高分 F 及其祖先，再自底向上删除未保护的 E 侧枝，最后筛选其余 F；策略可以完整可行，但不保证全局最优：

```bash
.venv/bin/python -m darp \
  --domain experiments/DARP-vs-RAOstar-grid/rddl/domain.rddl \
  --instance experiments/DARP-vs-RAOstar-grid/rddl/instance_5_h3.rddl \
  --heuristic experiments.DARP-vs-RAOstar-grid.darp_runner:MANHATTAN \
  --terminal-heuristic --rank-alpha 1 --rank-lambda 0.5 \
  --output experiments/DARP-vs-RAOstar-grid/output/rank.json
```

```python
result = solve_rddl("domain.rddl", "instance.rddl", rank_alpha=1.0, rank_lambda=0.5)
print(result.decision.complete)  # Global certificate / 全局搜索认证
print(result.decision.policy.duration_complete, result.decision.policy.feasible)
```

评分为 `utility - alpha × risk`；`alpha≥0` 只影响排序，不改变 ILP 的效用目标和风险预算。保护前 `ceil(|F| * lambda)` 个可用 F 和上一轮仍可用的选择；E 的目标是减少到约 `ceil(|E| * lambda)`，候选容量不足时向父层提升，保护路径内只能删除未保护侧枝。祖先连通与全部观测覆盖优先于比例，不保证严格缩小到 `lambda`。符号、伪代码、代价和限制见[算法方案](docs/ALGORITHM_MAPPING.md#rank-subtree)，同批对照和消融见[Rank 实验](experiments/RankedDarp-vs-HILP-vs-RAOstar-grid/README.md)。

Rank-based 模式每轮重新构建真正缩小的 Gurobi 模型，完整 E/F 留在内存中；受限模型无解时恢复完整当前 p-ILP，仍受同一总时限限制。只求得受限最优解时 `decision.complete=False`，但 `decision.policy.complete`（JSON）或 `duration_complete`（Python）可为真并允许回放。不会自动执行最终完整模型的最优性验证，也不保证筛选与重建后一定更快。

## RDDL 扩展

标准 RDDL 可以不写扩展字段。缺少 `duration` 时默认固定时长 1；缺少 `risk` 时默认没有危险状态，预算未提供时默认 0，等价于：

```rddl
duration = 1.0;    // domain
risk = false;     // domain
risk-budget = 0.0; // instance
```

显式设置的值不会被覆盖；只要显式声明了 `risk`（包括 `risk = false;`），就必须在 instance 中提供 `risk-budget`。

### Duration

Domain 内任选一种定义；示例中的 fluent 需先在模型中声明：

```rddl
duration = 1.0;                                            // fixed
duration = if (move_up) then 1.0 else 2.0;                  // 不同动作
duration = if (mud_contact) then 2.0 else 1.0;               // state-dependent
duration = Normal(if (mud_contact) then 2.0 else 1.0, 0.1);  // stochastic，第二项为方差
```

Instance 内使用原有的 `horizon` 指定时间边界：

```rddl
horizon = 3;
```

以上基础配置按固定或期望时长判断是否达到 `horizon`；仅写 `Normal(...)` 不会自动启用论文的 S 百分位停止判据，回放时仍采样其实际时长。`horizon` 不额外限制动作步数，`termination` 可提前结束执行。

Duration 表达式支持当前 kernel 的常量、算术、布尔、比较、`if`，及 state/action/non-fluent、确定性 intermediate fluent；`Normal` 需直接作为 duration 或 `if` 分支的结果，不能任意嵌入算术表达式。

### Risk

```rddl
// domain：unsafe 是已声明的 Boolean state fluent
risk = unsafe;

// instance：全策略至少失败一次的概率预算，包含初态
risk-budget = 0.1;
```

`risk` 必须是仅依赖当前 state/non-fluent 的确定性 Boolean，可经过确定性 intermediate fluent；不可引用 action、next-state 或随机采样。动作导致的失败应由 CPF 更新 `unsafe`。预算不是单步上限，重复进入风险状态不重复计数。

```rddl
// 无风险模型
risk = false;       // domain
risk-budget = 0.0;  // instance
```

### 语法与支持范围

```ebnf
duration-section  ::= "duration" "=" expr ";" ;
risk-section      ::= "risk" "=" expr ";" ;
risk-budget       ::= "risk-budget" "=" number ";" ;
```

`duration`、`risk` 在 domain，`risk-budget` 在 instance；省略时按上述规则处理。`horizon` 沿用原有 instance 字段，`expr/number` 复用 RDDL 语法。包含扩展声明的文件须用 DARP 的 `load_rddl()` 解析；标准文件也可直接用官方 `pyRDDLGym.make()` 加载。

当前支持 bool/int 状态、有限转移/观测、Boolean 单动作、`discount=1`；不支持并行动作、action preconditions、state invariants 或全部 RDDL 表达式。

## 策略保存与回放

### DARP 环境：加载策略并评估

```python
from contextlib import closing
from darp.adapter.loader import load_rddl
from darp.executor import PolicyExecutor
from darp.solve import DARPResult

result = DARPResult.load("result.json")
agent = PolicyExecutor(result.decision.policy)
with closing(load_rddl("domain.rddl", "instance.rddl").env) as env:
    statistics = agent.evaluate(env, episodes=1000, seed=0)
print(statistics)
```

返回 reward 的 `mean/median/min/max/std`、`risk_rate`、`physical_duration_mean`、`rollout_time_s` 和 `episodes`。采样风险频率不等于规划风险的证明；物理时长不同于墙钟时间；reward 不包含终端启发式。

### 原生 pyRDDLGym 环境：回放与可视化

标准 RDDL 必须与策略模型匹配，且不包含 DARP 扩展声明：

```python
from contextlib import closing
import pyRDDLGym
from darp.executor import PolicyExecutor
from darp.solve import DARPResult

result = DARPResult.load("result.json")
agent = PolicyExecutor(result.decision.policy)
with closing(pyRDDLGym.make("standard_domain.rddl", "standard_instance.rddl", vectorized=False)) as env:
    env.set_visualizer("text")  # 可替换成场景专用 visualizer
    statistics = agent.evaluate(env, seed=0, render=True)
    env.render()  # 最后一步结果
print(statistics)
```

无显示窗口时，将上例 `with` 内的三行替换为：

```python
statistics = agent.evaluate(env, seed=0)
image = env.render(to_display=False)
```

`evaluate()` 支持标准和扩展环境，共用相同默认值；标准无风险环境的 `risk_rate=0`，物理时长等于执行步数。若求解时定义了非默认 duration/risk，评估也必须使用相同定义，不能用原生环境的默认统计验证原约束。仅需单条轨迹信息时可用 `agent.run_episode(env)`，它不统计 risk/duration。可视化由环境负责，见[官方说明](https://pyrddlgym.readthedocs.io/en/latest/start.html#visualizing-environments)。

回放要求相同模型、对象、初始条件和 grounded fluent 命名，且策略完整可行；它重新采样轨迹，不是重放固定轨迹。执行器临时将环境步数上限扩到至少策略深度，到叶节点或环境终止时停止，随后恢复上限。自行调用 `sample_action()` 时须在返回 `None` 时停止。

### 单独序列化策略

以下接前面的 `result`；完整结果文件仍优先使用 `result.save()/DARPResult.load()`：

```python
import json
from darp.planning.policy import ConditionalPolicy

text = json.dumps(result.decision.policy.to_dict())
policy = ConditionalPolicy.from_dict(json.loads(text))
```

完整 JSON 为 `format="darp-result", version=1`，策略位于 `decision.policy`；策略字段如下：

| 字段 | 含义 |
| --- | --- |
| `format / version` | `darp-policy-graph / 1`，DARP 自有格式。 |
| `input / timing` | `observation` 或全可观测的 `state`；`act-then-observe`。 |
| `root / nodes` | 根动作 ID、有限无环策略图的节点列表。 |
| 节点 `id / stage / action_label / action` | ID、从 0 开始的深度、说明性标签、完整 grounded 动作字典。 |
| 节点 `transitions` | `{"observation": {...}, "next": "node-id"}` 的列表；`next=null` 为叶节点。 |
| `complete / feasible` | 策略级字段，均为 `true` 才能执行；Python 中 `complete` 对应 `duration_complete`，不同于外层 `decision.complete` 的全局搜索认证。 |
| `solver_status / achieved_utility / active_constraint_value` | 求解状态、规划效用、全策略风险。 |

执行契约：执行 action → 精确匹配 observation/state 字典 → 转到 next。沿边 `stage` 增加 1，字典顺序无关，名称和值须匹配。JSON 不绑定 RDDL 路径或哈希，调用者负责提供匹配环境。

## 实验

每个实验使用独立的 `rddl/` 和 `output/`；CSV 是原始指标，Markdown 是汇总表，`results/` 是策略 JSON。下面每配置求解 1 次，每个 DARP 策略默认回放 1000 次。

### Table 2：DARP vs RAO*

```bash
# 单配置；输出到 smoke.csv，避免覆盖正式结果
.venv/bin/python -m experiments.DARP-vs-RAOstar-grid.run \
  --instance experiments/DARP-vs-RAOstar-grid/rddl/instance_5_h3.rddl \
  --trials 1 --output experiments/DARP-vs-RAOstar-grid/output/smoke.csv

# 完整矩阵
TRIALS=1 bash tools/run_repro.sh

# 继续同版本、同配置的中断实验
TRIALS=1 RESUME=1 bash tools/run_repro.sh
```

首次运行自动下载固定 baseline 到 `.cache/baselines/`。离线运行使用固定 commit、clean worktree 的本地仓库：

```bash
CONSTRAINED_POMDP_REPO=/path/to/Constrained-POMDP \
RAOSTAR_CHECKOUT=/path/to/RAOStar \
TRIALS=1 bash tools/run_repro.sh
```

### Table 1：F/E/S duration

```bash
# 冒烟检查
.venv/bin/python -m experiments.DARP-table1-grid.run --smoke

# 完整矩阵
.venv/bin/python -m experiments.DARP-table1-grid.run --trials 1 \
  --summary experiments/DARP-table1-grid/output/table1.md
```

Table 1 的 E/S 仍与论文有数值差异；单次计时不是论文的 25 次统计。配置与比较边界见[实验协议](docs/EXPERIMENT_PROTOCOL.md)。

### Rank-based 参数对照

两个实验都支持 `--rank-alpha`、`--rank-lambda`；Table 1 的 Full-ILP、Table 2 的 RAO* 保持原算法。不同配置使用不同输出名，CSV 保存参数及全局搜索认证，`--resume` 拒绝混入其他参数或旧格式记录。

```bash
for fraction in 0.3 0.5 0.7 0.9; do
    .venv/bin/python -m experiments.DARP-table1-grid.run \
      --models F --horizons 3 --deltas 0.1 --planners hilp \
      --trials 1 --episodes 100 --rank-alpha 1 --rank-lambda "$fraction"
done
```

```bash
RANK_ALPHA=1 RANK_LAMBDA=0.5 TRIALS=1 bash tools/run_repro.sh
```

`alpha` 控制风险在候选评分中的重要性；当前对比实验固定为 1。`lambda` 控制筛选强度，`lambda=1` 不筛选，值越小，E 和 F 的目标保留数量越少。

同批对比原 HILP、只筛 F、筛选 E+F；RAO* 引用已有 Table 2 的历史结果：

```bash
.venv/bin/python -m experiments.RankedDarp-vs-HILP-vs-RAOstar-grid.run \
  --trials 3 --lambdas 0.3 0.5 0.7 0.9 --timeout 120 --episodes 1000
```

## 自定义 Heuristic

以 Grid 为例，在项目根目录创建 `my_heuristic.py`；其他场景需替换状态字段和估值逻辑：

```python
from darp.planning.heuristic import HeuristicInput, UtilityHeuristic

def estimate(value: HeuristicInput) -> float:
    # state：单个状态；non_fluents：模型常量
    # action_label：动作名；action：完整 grounded 动作字典
    row, col = value.state["grid_row"], value.state["grid_col"]
    goal_row, goal_col = value.non_fluents["goal_row"], value.non_fluents["goal_col"]
    return -float(abs(row - goal_row) + abs(col - goal_col))  # cost-to-go 取负

MY_HEURISTIC = UtilityHeuristic(
    name="my-grid-manhattan",
    evaluate=estimate,
    upper_bound=False,  # 证明是最优 utility 的上界后才能设 True
)
```

命令行加载：

```bash
.venv/bin/python -m darp \
  --domain domain.rddl --instance instance.rddl \
  --planner hilp --heuristic my_heuristic:MY_HEURISTIC --output result.json
```

Python 调用：

```python
from darp.solve import solve_rddl
from my_heuristic import MY_HEURISTIC

result = solve_rddl("domain.rddl", "instance.rddl", heuristic=MY_HEURISTIC)
result.save("result.json")
```

回调不乘 belief/history probability，核心统一加权。未证明上界时保持 `upper_bound=False`，结果可能无法认证搜索完成。`--terminal-heuristic` 会改变边界目标，仅用于明确采用终端估值的实验，不作为通用求解选项。
