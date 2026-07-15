# DARP

DARP（Durative Action RDDL Planner）是一个面向 RDDL 的 Python 研究型规划器。当前实现使用 pyRDDLGym 负责标准 RDDL 的解析、grounding 和仿真；DARP 负责 AND-OR history tree、duration sidecar、chance-constrained risk 表达、full-ILP/HILP 编码，并通过 Gurobi 求解。

当前仓库的实验对比对象是 RAO*。RAO* 不属于 DARP planner API，而是作为外部 baseline 放在 `experiments/baselines/rao_star/`，由实验脚本调用。

## 安装

```bash
bash tools/install_linux_deps.sh
source .venv/bin/activate
```

如果需要在新的 Ubuntu/Debian 机器上安装系统依赖：

```bash
INSTALL_SYSTEM_DEPS=1 bash tools/install_linux_deps.sh
```

Gurobi 说明：脚本会安装 `gurobipy`，但真实 full-ILP/HILP 实验还需要本机有有效 Gurobi license。

## DARP 命令行

查看帮助：

```bash
darp -h
```

运行默认 online trace：

```bash
darp \
  --domain experiments/inputs/rddl/tiny_grid_domain.rddl \
  --instance experiments/inputs/rddl/tiny_grid_instance.rddl
```

运行 HILP：

```bash
darp \
  --domain experiments/inputs/rddl/tiny_grid_domain.rddl \
  --instance experiments/inputs/rddl/tiny_grid_instance.rddl \
  --duration experiments/inputs/durations/tiny_grid.yaml \
  --planner hilp \
  --hilp-heuristic reachable-bellman \
  --heuristic-lookahead-depth 4 \
  --expansion-rounds 4 \
  --frontier-width 1 \
  --output /tmp/darp_tiny_grid_hilp.json
```

运行 full-ILP：

```bash
darp \
  --domain experiments/inputs/rddl/tiny_grid_domain.rddl \
  --instance experiments/inputs/rddl/tiny_grid_instance.rddl \
  --duration experiments/inputs/durations/tiny_grid.yaml \
  --planner full-ilp
```

HILP frontier heuristic：

- `reachable-bellman`：在当前 frontier action 的可达后继状态集合上做有限 horizon fully observable Bellman backup。
- `one-step-greedy`：只使用当前 action 的一步期望 reward，速度更快但更贪心。

## RAO* 对比实验

实验入口统一放在 `experiments/scripts/`。Science Agent / PSR 场景适配完成后，`--domain`、`--instance` 和 `--duration` 应指向 `experiments/inputs/rao_star/` 下的对应文件。

```bash
python experiments/scripts/run_rao_star_suite.py \
  --name science_agent_small_fixed \
  --domain experiments/inputs/rao_star/science_agent/science_agent_domain.rddl \
  --instance experiments/inputs/rao_star/science_agent/science_agent_small.rddl \
  --duration experiments/inputs/durations/fixed_1.yaml \
  --seeds 0 \
  --planners rao-star,hilp \
  --hilp-heuristic reachable-bellman \
  --heuristic-lookahead-depth 2 \
  --frontier-width 1
```

结果默认写入：

```text
experiments/outputs/<experiment-name>/
```

生成回放页面：

```bash
python experiments/scripts/visualize_replay.py \
  experiments/outputs/<experiment-name>/runs.csv \
  --output experiments/outputs/<experiment-name>/replay.html
```

LaTeX 表格模板放在：

```text
experiments/reports/latex/
```

## 实验工作区

```text
experiments/
├── inputs/
│   ├── rddl/            # tiny_grid、factored_door 等小型 RDDL sanity checks。
│   ├── durations/       # fixed / Gaussian duration sidecar 示例。
│   ├── rao_star/        # RAO* 原文 Science Agent / PSR 复现实验输入。
│   └── benchmarks/      # RDDL/IPPC benchmark corpus，供后续扩展使用。
├── baselines/
│   └── rao_star/        # 外部 RAO* 对比 baseline。
├── scripts/             # 实验运行、baseline 调用、replay 可视化脚本。
├── outputs/             # 生成的 JSON/CSV/log/replay 输出，默认不提交。
└── reports/
    └── latex/           # 可提交的 LaTeX 表格模板与预览入口。
```

## Duration Sidecar

duration sidecar 只描述动作持续时间和可选 risk，不重复 RDDL instance 中的 `horizon`。

```yaml
kind: fixed
default: 1
actions:
  move-east: 1
  move-south: 1
risk:
  budget: 0.25
  next_state_fluents:
    at___c22: 1
```

## 架构

```text
darp CLI
  -> adapter.RDDLLoader
  -> pyRDDLGym env/model/native AST
  -> adapter.GroundedRDDLView
  -> adapter.ExactRDDLKernel（按需状态索引、稀疏 NumPy belief、CPF 结果缓存）
  -> model.ANDORNode（整数节点池）/ History / DurationModel
  -> planning.preprocess / planning.expand
  -> planning.FullILPPlanner or planning.HILPPlanner
  -> ilp.GurobiSolver
  -> planning.OnlineSession
```

数值与树结构的性能设计：

- pyRDDLGym 仍然预先 grounding 参数化 fluent 和 CPF；DARP 不预先枚举所有状态赋值，而是在 Algorithm 2/HILP 首次触达 `(state, action)` 时生成其非零后继。
- exact belief 在公开接口中保持论文易读的 `StateKey -> probability`，在计算内部转换为整数 `state_id` 和稀疏 NumPy 概率向量。
- transition、reward 和 observation 按 `(state_id, action_id)` 持久缓存，并在同一 online session 的后续 HILP 轮次和决策中复用。
- AND-OR history tree 使用 DARP 专用整数节点池和 O(1) child 去重；NetworkX 不进入求解热路径，仅适合作为后续调试/可视化导出格式。
- `ActionDecision.timing` 中的 `exact_discovered_states`、`exact_transition_rows` 和 `exact_*_hits` 可用于检查按需状态发现与缓存效果。

项目结构：

```text
DARP/
├── src/darp/            # DARP 核心 planner、adapter、model、ILP 与 visualization 代码。
├── experiments/         # 实验输入、外部 baseline、脚本、输出和报告。
├── docs/                # 论文符号表、benchmark strategy 等开发文档。
├── tools/               # 安装与维护脚本。
└── tests/               # 单元测试和轻量集成测试。
```

## Roadmap

- [x] 使用 pyRDDLGym 作为标准 RDDL parser/grounder/simulator。
- [x] 建立 DARP AND-OR history tree、duration sidecar 和 exact finite kernel。
- [x] 实现 full-tree ILP 与 HILP partial-tree search 的 Gurobi 求解路径。
- [x] 支持 fixed duration、Gaussian percentile duration 和 sidecar risk budget。
- [x] 将 RAO* 从 DARP planner API 移到外部 baseline wrapper。
- [x] 统一实验输入、baseline、脚本、输出和报告目录。
- [x] 实现按需可达状态索引、稀疏 NumPy exact belief 和 transition/reward/observation 缓存。
- [x] 去除 exact planner 的 pyRDDLGym 环境深拷贝，并使用整数 AND-OR 节点池。
- [ ] 复现 RAO* 原文 Science Agent / PSR benchmark adapter。
- [ ] 为 HILP 实现增量 Gurobi model、warm start、online subtree 数值复用和 benchmark-scale pruning。
- [ ] 支持 concurrent action combinations 和非 bool action。
- [ ] 如确有必要，基于 pyRDDLGym parser 扩展原生 durative-action RDDL 语法。

## 测试

```bash
python -m pytest
```

基础测试不要求本机有 Gurobi license；真实 full-ILP/HILP 实验需要可用 Gurobi。

## 当前限制

- 当前 exact kernel 主要面向有限、grounded、bool fluent/action 的 RDDL 问题。
- pyRDDLGym 的 fluent/CPF grounding 仍在规划开始前完成；当前按需机制优化的是可达状态、转移和 belief 数值层，而不是 lifted symbolic grounding。
- 当前 `experiments/baselines/rao_star/` 仍是小规模 exact deterministic-policy 对照 wrapper；正式实验应迁移到 RAO* 原文 Science Agent / PSR 场景。
- full-ILP 会展开完整剩余 horizon，规模随 action/observation history 指数增长。
- HILP 是 partial-tree refinement，不等价于全局最优证明，除非展开到与 full tree 等价或加入严格 bound/certificate。
