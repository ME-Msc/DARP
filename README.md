# DARP

DARP（Durative Action RDDL Planner）是一个面向 RDDL 的 Python 研究型规划器。当前主线使用 pyRDDLGym 负责标准 RDDL 解析、grounding 和仿真，DARP 自己维护 AND-OR history tree、duration sidecar、full-ILP/HILP 编码，并使用 Gurobi 求解论文《Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions》中的 fixed-horizon / chance-constrained POMDP 规划问题。

当前 main 分支只保留一条实现方案：

- RDDL parser / grounder / simulator：pyRDDLGym
- DARP 数据结构：AND-OR tree、belief、duration、risk sidecar
- 论文算法：full-tree ILP baseline 与 HILP partial-tree search
- ILP solver：Gurobi
- durative action：暂时通过 YAML/JSON sidecar 定义，未来再考虑继承 pyRDDLGym parser 扩展 RDDL 语法

## 安装

推荐在 Linux 上使用项目脚本创建虚拟环境并安装 DARP 依赖：

```bash
bash scripts/install_linux_deps.sh
source .venv/bin/activate
```

如果是一台新的 Ubuntu/Debian 机器，并且希望同时安装系统依赖、克隆并构建 PROST 与 rddlsim：

```bash
INSTALL_SYSTEM_DEPS=1 bash scripts/install_linux_deps.sh
```

默认会把 PROST 和 rddlsim 放在 DARP 仓库同级目录：

```text
../prost-planner
../rddlsim
```

也可以覆盖路径：

```bash
PROST_ROOT=/path/to/prost-planner \
RDDLSIM_ROOT=/path/to/rddlsim \
bash scripts/install_linux_deps.sh
```

Gurobi 说明：脚本会安装 `gurobipy`，但 full-ILP/HILP 真实求解还需要本机有有效 Gurobi license。

手动安装方式：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[gurobi]" -r requirements-dev.txt
```

## DARP 命令行

查看帮助：

```bash
darp -h
```

运行默认 online trace：

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl
```

运行 HILP：

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --duration examples/durations/tiny_grid.yaml \
  --planner hilp \
  --heuristic-lookahead-depth 4 \
  --expansion-rounds 4 \
  --frontier-width 1 \
  --hilp-heuristic reachable-bellman \
  --output /tmp/darp_tiny_grid_hilp.json
```

运行 full-ILP：

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --duration examples/durations/tiny_grid.yaml \
  --planner full-ilp
```

HILP 当前支持两个 frontier heuristic：

- `reachable-bellman`：在当前 frontier action 的可达后继状态集合上做有限 horizon fully observable Bellman backup。
- `one-step-greedy`：只使用当前 action 的一步期望 reward，速度更快但更贪心。

## DARP/PROST 对比实验

实验统一由 `scripts/run_benchmark_suite.py` 运行。推荐用一行命令指定 benchmark 名称、RDDL 文件、duration、seed 和 HILP 参数；结果都会写入 `experiments/<name>/`，并默认生成可视化 HTML：

```bash
.venv/bin/python scripts/run_benchmark_suite.py \
  --name navigation_2011_inst2_fixed_1 \
  --domain examples/benchmarks/navigation-2011/navigation_mdp.rddl \
  --instance examples/benchmarks/navigation-2011/navigation_inst_mdp__2.rddl \
  --duration examples/durations/fixed_1.yaml \
  --seed 0 \
  --timeout 300 \
  --heuristic-lookahead-depth 6 \
  --expansion-rounds 2 \
  --frontier-width 1
```

默认 planner 是 `hilp`，默认 heuristic 是 `reachable-bellman`。固定 duration=1 的对比实验建议显式传入 `examples/durations/fixed_1.yaml`，这样实验配置和 DARP trace 都能清楚记录 duration 假设：

```bash
.venv/bin/python scripts/run_benchmark_suite.py \
  --name tiny_grid_fixed_1 \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --duration examples/durations/fixed_1.yaml \
  --prost-state-parser tiny_grid \
  --seed 0
```

常用选项：

- `--duration`：为 DARP 指定 duration sidecar；PROST 不支持 duration，DARP/PROST 对比推荐使用 `examples/durations/fixed_1.yaml`。
- `horizon`：来自 RDDL instance，决定 online episode 的最大真实步数。
- `--heuristic-lookahead-depth`：`reachable-bellman` 为 frontier action 估值时向未来做几层 fully observable Bellman backup。
- `--expansion-rounds`：可选的 HILP 扩树预算；省略时按 RDDL horizon 和 duration stopping condition 一直扩到 frontier 耗尽，适合小问题，benchmark 通常显式设置预算。
- `--frontier-width`：每轮展开几个 frontier。
- `--hilp-heuristics`：默认 `reachable-bellman`，也可设为 `one-step-greedy`。
- `--no-visualize`：关闭默认 HTML replay 生成。
- `--seeds`：一次运行多个 seed，例如 `--seeds 0,1,2`。

PROST/rddlsim 路径用环境变量管理：

```bash
export PROST_ROOT=/path/to/prost-planner
export RDDLSIM_ROOT=/path/to/rddlsim
export PROST_PYTHON=.venv/bin/python
```

脚本会自动创建/更新 `experiments/<name>/config.json`，用于记录这条命令定义的 benchmark。也可以直接传入已有 experiment 文件夹复跑：

```bash
.venv/bin/python scripts/run_benchmark_suite.py experiments/navigation_2011_inst2_fixed_1 --seed 0
```

每个实验目录会生成：

- `config.json`：实验定义，可提交到 git。
- `summary.json`：完整原始结果。
- `runs.csv`：long-format 逐 run 结果。
- `summary.csv`：按 solver variant 聚合的均值/标准差。
- `replay.html`：可选的同屏回放页面。
- `seed_<n>/`：stdout、stderr、PROST/rddlsim logs、DARP trace 等原始产物。

DARP 会在每一步执行完成后增量写出 trace。如果某个大 instance 在 `--timeout` 前没有完整跑完，`replay.html` 和 CSV 仍会尽量使用已经完成的 action/state 前缀；如果超时发生在第一步决策还未结束前，则只能显示空前缀。

DARP 的在线执行由 pyRDDLGym 在同一个 Python 进程内完成：`adapter.runtime.PyRDDLGymRuntime` 封装 `env.reset(seed=...)` 和 `env.step(action)`，`planning.run_online_session` 在每步调用 planner 选 action 后立刻交给 pyRDDLGym 更新环境状态。PROST 对比实验则仍然启动 rddlsim server 和 PROST client，通过 IPC 协议实时交互。

总表和 LaTeX 预览只放在：

```text
experiments/latex/
```

打开下面文件即可用 LaTeX Workshop 预览总表：

```text
experiments/latex/preview_summary_table.tex
```

## Duration Sidecar

duration sidecar 只描述动作持续时间和可选 risk，不重复 RDDL instance 中的 `horizon`。通用 fixed duration=1 配置放在 `examples/durations/fixed_1.yaml`，适合 PROST 对比实验中明确声明 DARP 的单位动作时长。

固定 duration：

```yaml
kind: fixed
default: 1
actions:
  move-east: 1
  move-south: 1
```

带 chance constraint risk budget：

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

Gaussian percentile duration：

```yaml
kind: gaussian
zeta: 0.3
default_mean: 1
default_variance: 0.05
state_actions:
  at___c22:
    move-east:
      mean: 2
      variance: 0.25
```

## 架构

```text
darp CLI
  -> adapter.RDDLLoader
  -> pyRDDLGym env/model/native AST
  -> adapter.GroundedRDDLView
  -> adapter.ExactRDDLKernel
  -> model.ANDORNode / History / DurationModel
  -> planning.preprocess / planning.expand
  -> planning.FullILPPlanner or planning.HILPPlanner
  -> ilp.GurobiSolver
  -> planning.OnlineSession
```

目录结构：

```text
DARP/
├── src/darp/
│   ├── adapter/        # pyRDDLGym loading、grounded view、exact kernel、runtime。
│   ├── model/          # AND-OR tree、history、duration sidecar 和 duration evaluator。
│   ├── planning/       # online session、rollout baseline、full-ILP、HILP、Expand。
│   ├── ilp/            # DARP ILP schema 和 Gurobi adapter。
│   └── visualization/  # replay frame schema、graph、trace 解析、domain-specific decoder。
├── examples/
│   ├── rddl/           # 小型手写 RDDL 示例。
│   ├── durations/      # duration/risk sidecar 示例。
│   └── benchmarks/     # PROST/IPC benchmark RDDL 文件。
├── experiments/        # 可提交 config，忽略运行结果。
├── scripts/            # 安装、DARP/PROST 对比、可视化和汇总脚本。
└── tests/              # 单元测试和轻量集成测试。
```

## Roadmap

- [x] 使用 pyRDDLGym 作为标准 RDDL parser/grounder/simulator。
- [x] 建立 DARP AND-OR history tree、duration sidecar 和 exact finite kernel。
- [x] 实现 full-tree ILP 与 HILP partial-tree search 的 Gurobi 求解路径。
- [x] 支持 fixed duration、Gaussian percentile duration 和 sidecar risk budget。
- [x] 提供 DARP/PROST 对比 runner、CSV/JSON 输出、LaTeX 表格和 HTML replay。
- [x] 将 replay 可视化拆分为 frame schema、graph、trace 和 benchmark-specific decoder 模块。
- [ ] 改进 HILP 的 benchmark-scale pruning 与 large action-space handling。
- [ ] 支持 concurrent action combinations 和非 bool action。
- [ ] 补充更多 benchmark 的 PROST state parser 与可视化 renderer。
- [ ] 实现 PROST/rddlsim 风格在线协议兼容层。
- [ ] 如确有必要，基于 pyRDDLGym parser 扩展原生 durative-action RDDL 语法。

## 测试

```bash
python -m pytest
```

基础测试不要求本机有 Gurobi license；真实 full-ILP/HILP 实验需要可用 Gurobi。

## 当前限制

- 当前 exact kernel 主要面向有限、grounded、bool fluent/action 的 RDDL 问题。
- full-ILP 会展开完整剩余 horizon，规模随 action/observation history 指数增长。
- HILP 是 partial-tree refinement，不等价于全局最优证明，除非展开到与 full tree 等价或加入严格 bound/certificate。
- PROST 不支持 DARP duration sidecar；公平对比时通常使用 fixed duration=1。
- navigation 中的 `P(x,y)` 是 transition failure probability，不是 DARP 自动识别的 chance constraint。通用 chance constraint 必须通过显式 risk sidecar 或后续标准化建模接口传入。
