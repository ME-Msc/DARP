# DARP

Durative Action RDDL Planner 是一个 Python 研究原型，目标是在标准 RDDL 问题上实现 fixed-horizon POMDP / constrained POMDP / durative-action 规划算法，并逐步实现论文《Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions》中的 AND-OR tree、full-tree baseline 与 HILP 搜索流程。

main 分支只保留一条实现主线：pyRDDLGym 负责标准 RDDL parser、grounder 和 simulator；DARP 用自己的数据结构组织 AND-OR history tree，把固定 horizon CC-POMDP 搜索问题转换为论文中的 full ILP / HILP p-ILP，并以 Gurobi 作为唯一目标 ILP solver。durative action 当前通过 YAML/JSON sidecar 定义；未来如需扩展 RDDL 语法，再基于 pyRDDLGym parser 做继承式扩展。

## 功能状态

- `adapter/` 使用 pyRDDLGym 加载标准 RDDL domain/instance，并返回 `PyRDDLGymProblem(env, model, native_ast)`。
- `PyRDDLGymProblem.build_grounded_model()` 直接复用 pyRDDLGym `RDDLGrounder(...).ground()`，返回 pyRDDLGym 的 `RDDLGroundedModel`；`GroundedRDDLView` 封装它，DARP 不再自己实现 grounding。
- `GroundedRDDLView.build_and_or_interface()` 已能把 grounded action、observation scope 和 root history 组织成 AND-OR search interface。
- `adapter.ExactRDDLKernel` 基于 pyRDDLGym grounded CPF 为有限 bool 状态模型精确枚举 transition / observation / reward，并按论文 Lemma 3.3 递推 chance-constrained safe belief 与 `rho*(q)` risk 常量；online full-ILP/HILP 使用 exact Bayes belief update 维护 root belief 后再建树。
- `darp --domain --instance` 默认通过 pyRDDLGym + rollout baseline 执行快速 online trace；`--planner full-ilp` / `--planner hilp` 会切换到论文算法路径。
- `model/` 保留 DARP 自己的 `DurationModel`、duration sidecar 和 AND-OR tree 数据结构，并已接入 Phase 7 的 `tau(q)` 剪枝。
- `planning/` 已提供论文结构对齐的 `preprocess`、`Expand`、full-tree baseline 和 HILP-style partial-tree search；full-ILP 不再使用额外 lookahead depth 截断，而是展开到 RDDL 剩余 horizon / duration stopping condition；HILP 通过 Gurobi 求解当前 `E ∪ F` partial-tree p-ILP，并支持 `one-step-greedy` 与 `reachable-bellman` 两种 frontier utility heuristic。二者都不采样未来、不递归展开 observation tree，也不回退调用 full-ILP。
- `ilp/` 提供 DARP 自己的小型二元 ILP schema 和唯一 solver adapter：Gurobi。
- durative action 当前只通过 YAML/JSON sidecar 定义；未来如需原生语法，会继承 pyRDDLGym parser 做扩展解析。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

Gurobi 是论文 ILP/HILP 路线的唯一 solver 目标；运行 ILP/HILP 求解时安装：

```bash
python -m pip install -e ".[gurobi]"
```

## 命令行

查看帮助：

```bash
darp -h
```

加载标准 RDDL，并用 pyRDDLGym 执行默认 online trace：

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl
```

设置 rollout 的 lookahead 或 HILP 的 partial-tree 最大细化深度；`reachable-bellman` 模式也会把它作为局部 Bellman horizon 上限。`--particles` 只影响 rollout 的 POMDP 近似 belief：

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --lookahead-depth 4 \
  --particles 32
```

使用 duration sidecar 和 full-tree ILP planner：

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --duration examples/durations/tiny_grid.yaml \
  --planner full-ilp
```

使用 HILP partial-tree p-ILP planner：

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --duration examples/durations/tiny_grid.yaml \
  --planner hilp \
  --lookahead-depth 4 \
  --hilp-iterations 4 \
  --frontier-width 1 \
  --hilp-heuristic one-step-greedy
```

HILP heuristic 有两个模式：

- `one-step-greedy`：默认模式，使用 $$h_q^u := u_q = \rho^*(q)\sum_s b_q^*(s)U(s,a_q)$$，最快，但只看当前动作的一步期望奖赏。
- `reachable-bellman`：从当前 frontier action 的后继状态 support 出发，只在剩余 lookahead 内可达的状态集合上做 fully observable Bellman backup；它会看未来 reward，但不枚举 observation tree，适合 tiny grid 这类状态空间较小的问题。

`full-ilp` / `hilp` 是论文路线 planner，必须安装可用的 `gurobipy` 和 Gurobi license；缺少 Gurobi 时会直接报错。`rollout` 是不依赖 Gurobi 的 baseline。

注意：CC-POMDP 规划问题中的时间预算不是 Python 程序运行时间。DARP 使用 RDDL instance 的 `horizon` 加上 duration sidecar 中的动作持续时间，通过 `tau(q)` 判断 history 是否还能继续展开。

## PROST 对比实验

DARP/PROST 对比实验由通用 runner `scripts/darp_prost_compare.py` 负责进程管理、日志解析、metrics 汇总和表格输出；具体场景脚本只提供 RDDL 路径、duration sidecar 和必要的 PROST 状态解析逻辑。

tiny grid fixed duration=1 的对比实验命名为 `tiny_grid_fixed_1`，它由 `scripts/tiny_grid_fixed_1.py` 调用通用 runner。建议用脚本运行，而不是 notebook 直接启动进程。脚本默认会分别运行 DARP HILP 的 `one-step-greedy` 和 `reachable-bellman` 两种 heuristic 模式，再启动 rddlsim server 和 PROST client，并把 stdout、stderr、rddlsim logs、DARP JSON trace 和 summary 写到同一个目录：

```bash
python scripts/tiny_grid_fixed_1.py --seeds 0
```

脚本结束时会在终端打印按 metric 分行的 DARP/PROST 对比表，默认列为 `DARP-one-step-greedy`、`DARP-reachable-bellman` 和 `PROST`。指标包括 `total_reward`、`turns`、`runtime_s`、`rddl_load_ms`、`grounding_ms`、`and_or_interface_ms`、`initial_belief_ms`、`planner_elapsed_ms`、`decision_ms`、`frontier_expand_ms`、`heuristic_eval_ms`、`ilp_encode_ms`、`tree_ilp_build_ms`、`gurobi_call_ms`、`postprocess_ms`、`actions`、`states`、`expanded_nodes`、`performed_trials`、`ilp_vars`、`ilp_constraints`、`gurobi_ms`、`prost_parsing_ms`、`prost_simplifying_ms`、`prost_analyzing_ms`、`risk_budget` 和 `constraint_violation`。当前还没有稳定统计的指标会先保留为空列。

其中 `actions` 会分别来自 DARP JSON trace 和 PROST client stdout；`states` 在 DARP 侧来自 JSON trace，在 PROST 侧来自 client stdout 中的 `Current state` 位向量，最后一个状态由最后动作推断。DARP trace 中的 `elapsed_ms` 是每步 `choose_action()` 的 wall-clock 耗时，因此汇总为 `planner_elapsed_ms`；DARP 的 `decision_ms` 来自 planner 内部 `timing.decision_ms`，目前定义为 tree/partial-tree + ILP 构造、Gurobi 求解和动作抽取时间之和。PROST 的 `decision_ms` 来自 client stdout 中每步 `Search time` 的和。DARP 的 `grounding_ms` 对应 pyRDDLGym grounder，PROST 的 closest counterpart 是 `Instantiating`；DARP 的 `frontier_expand_ms`、`heuristic_eval_ms`、`ilp_encode_ms` 只对 HILP 有意义，PROST 侧保留为空或使用 PROST 自己的 parser/search phase 指标。`--open-terminals` 模式下，PROST 的 `runtime_s` 使用 PROST client 启动到结束的时间戳，不再使用外部等待窗口时间。

可以用 `--hilp-heuristics` 指定要比较的 DARP heuristic 模式：

```bash
python scripts/tiny_grid_fixed_1.py \
  --seeds 0 \
  --hilp-heuristics one-step-greedy,reachable-bellman
```

如果你想像手动实验一样看到多个终端窗口，可以使用：

```bash
python scripts/tiny_grid_fixed_1.py --seeds 0 --open-terminals
```

该模式会为每个 DARP heuristic 生成一个 `run_darp_<heuristic>.sh`，同时生成 `run_prost_server.sh` 和 `run_prost_client.sh`，并用 `gnome-terminal` 分别打开多个 DARP variant、rddlsim server 和 PROST client。PROST client 会等待 server 日志显示初始化完成后再启动；执行该命令的原终端会继续等待日志和 trace 写完，然后打印同一张 metrics 对比表。

默认 PROST 路径是 `/home/shaocong/Desktop/prost-planner`，rddlsim 路径来自 `RDDLSIM_ROOT` 或 `/home/shaocong/Desktop/rddlsim`，PROST helper Python 使用 conda 的 `rddl` 环境。`examples/rddl/tiny_grid_*` 本身使用 DARP/pyRDDLGym 和 PROST/rddlsim 都支持的共同 RDDL 子集：`location` 是 `object` 类型，对象写在 instance 的 `objects` 块中，domain 通过 non-fluent 描述 goal、risk、penalty direction 和 transition graph，而不是硬编码 enum 常量。tiny grid 的 reward 使用负代价语义：普通移动是 `-1`，非 goal 的 noop 是 `-2`，危险方向是 `-10`，到达 goal 后 reward 为 `0`；最大化 reward 因此等价于尽快到达 goal 并避免危险方向。

## Duration Sidecar

duration sidecar 只描述动作持续时间模型，不描述 RDDL 已经定义的内容。`horizon` 来自 RDDL instance；sidecar 中不要写 `horizon` 或 `version`。

最小 fixed-duration 示例：

```yaml
kind: fixed
default: 1
actions:
  move-east: 1
  move-south: 1
```

`default` 是针对未在 `actions` 中显式配置的 action 的默认 duration。上例中，如果 RDDL 中还有 `move-west` 但 sidecar 没写它，DARP 会使用 `default: 1` 作为它的持续时间。

fixed-duration C-POMDP 风险约束可以在同一个 sidecar 中声明；下面的例子把进入 `at___c22` 视为风险成本，并设置总风险预算：

```yaml
kind: fixed
default: 1
actions:
  move-east: 1
  move-south: 1
  move-west: 1
  move-north: 1
risk:
  budget: 0.25
  next_state_fluents:
    at___c22: 1
```

在 full-ILP 中，DARP 按论文 Lemma 3.3 使用 safe-belief chance constraint：ILP 右端项为 `R = Delta - r(b0)`，每个 action history 的风险系数为 `rho*(q) * r(b_q)`；`rho*(q)` 和 safe belief `b*` 会沿 AND-OR tree 递推，而不是用普通 belief 简单累加 expected cost。

Stochastic Duration with Percentile Risk Criteria 使用 `kind: gaussian` 和 `zeta`。`zeta` 对应论文里的 percentile threshold `\varsigma`，DARP 会根据 Algorithm 2 的 smoothed belief 边缘概率计算 action duration 的均值/方差并用于 `tau(q)`：

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
risk:
  budget: 0.25
  next_state_fluents:
    at___c22: 1
```

将在线 trace 写成 JSON：

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --output tiny_grid_pyrddlgym_trace.json
```

直接检查 pyRDDLGym problem 组件和 planner interface 边界：

```bash
python -m darp.adapter.loader \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl
```

## 执行流程

当前主路径刻意保持简单：

```text
darp CLI
  -> adapter.RDDLLoader.load(domain, instance)
  -> pyRDDLGym.make(...)
  -> PyRDDLGymProblem(env, model, native_ast)
  -> adapter.PyRDDLGymRuntime.reset/step
  -> planning.RolloutPlanner.choose_action（默认快速路径）
  -> planning.run_online_session 管理 action、observation、state、belief 和 trace
```

论文算法主路径：

```text
PyRDDLGymProblem.build_grounded_model()
  -> pyRDDLGym.core.compiler.model.RDDLGroundedModel
  -> adapter.GroundedRDDLView（复用 pyRDDLGym grounding 结果）
  -> model.ANDORNode / History
  -> model.DurationModel（来自 YAML/JSON sidecar 或默认单位 duration）
  -> planning.preprocess / planning.expand
  -> planning.FullILPPlanner / planning.HILPPlanner
  -> Gurobi full ILP / p-ILP solve
  -> OnlineSession 每步 take action 并从 pyRDDLGym env 接收 observation/reward/state
```

## 架构边界

- pyRDDLGym：负责标准 RDDL parser、semantic handling、grounder、Gym-style environment 和 simulator。
- DARP `adapter/`：固定作为 pyRDDLGym 适配层，负责 loading、grounded-model view、runtime reset/step 和轻量 belief helper。
- DARP `model/`：保存 DARP 自己的规划数据结构，例如 duration model、history、AND-OR tree node；这里不直接依赖 pyRDDLGym。
- DARP `planning/`：保存 rollout baseline、论文搜索 scaffold、HILP/full-tree planner、online session 和 trace；这里通过 adapter/runtime 接触 pyRDDLGym simulator。
- DARP `ilp/`：保存 binary ILP schema 和 Gurobi adapter；`planning/` 负责把 exact AND-OR tree/frontier 编码成 full ILP 或 HILP p-ILP。

## 文件结构

```text
DARP/
├── README.md                         # 中文主文档。
├── README-EN.md                      # 英文镜像文档。
├── pyproject.toml                    # Python 包配置和 Gurobi solver extra。
├── requirements.txt                  # 运行时依赖，包含 pyRDDLGym。
├── requirements-dev.txt              # 开发/测试依赖。
├── scripts/
│   ├── darp_prost_compare.py        # DARP/PROST 对比实验的通用 runner。
│   └── tiny_grid_fixed_1.py         # tiny_grid 场景、fixed duration=1 的 runner 配置脚本。
│
├── examples/
│   ├── benchmarks/                   # PROST/IPC RDDL MDP benchmark 数据集。
│   │   ├── README.md                 # benchmark 来源、目录约定和导入清单。
│   │   └── <domain-year>/            # 单个 benchmark domain 目录。
│   ├── durations/                    # DARP duration sidecar 示例。
│   │   ├── tiny_grid.yaml            # tiny grid fixed-duration + risk sidecar。
│   │   └── tiny_grid_gaussian.yaml   # tiny grid Gaussian percentile duration sidecar。
│   └── rddl/                         # 小型手写 RDDL 示例。
│       ├── tiny_grid_domain.rddl     # tiny grid 标准 RDDL domain。
│       ├── tiny_grid_instance.rddl   # tiny grid instance。
│       ├── factored_door_domain.rddl # partial-observation toy domain，供未来 adapter 回归使用。
│       └── factored_door_instance.rddl # factored door instance。
│
├── src/darp/
│   ├── __init__.py                   # 包版本入口。
│   ├── __main__.py                   # `darp` 顶层命令入口。
│   ├── adapter/                      # 当前唯一外部系统 pyRDDLGym 的适配层。
│   │   ├── __init__.py               # adapter package 入口。
│   │   ├── problem.py                # PyRDDLGymProblem 容器、load error 和 pyRDDLGym grounder 入口。
│   │   ├── loader.py                 # 使用 pyRDDLGym 加载标准 RDDL。
│   │   ├── grounded.py               # GroundedRDDLView，封装 pyRDDLGym grounded model。
│   │   ├── exact.py                  # 从 grounded CPF 精确枚举有限 transition/observation/reward/risk kernel。
│   │   └── runtime.py                # pyRDDLGym reset/step/action/belief runtime。
│   ├── model/                        # DARP 自己的规划数据结构。
│   │   ├── __init__.py               # model package 入口。
│   │   ├── and_or_tree.py            # AND-OR history tree 的基础节点和 history 结构。
│   │   ├── duration.py               # DurationModel、HistoryDurationEvaluator 和 tau(q) 计算。
│   │   └── duration_sidecar.py       # JSON/YAML duration sidecar loader。
│   ├── ilp/                          # DARP 二元 ILP schema 和 Gurobi solver adapter。
│   │   ├── __init__.py               # ilp package 入口。
│   │   ├── model.py                  # ILPVariable、ILPLinearConstraint、ILPModelSpec 和 solve result。
│   │   └── gurobi.py                 # Gurobi adapter；DARP 论文路线唯一 ILP solver。
│   └── planning/                     # planner 与在线执行编排。
│       ├── __init__.py               # planning package 入口。
│       ├── preprocess.py             # Algorithm 1 的 root frontier 初始化工具。
│       ├── expand.py                 # 论文 Expand 操作，计算 rho/u/r/tau、backward message 和 smoothed belief。
│       ├── ilp_tree.py               # 运行 Algorithm 1 preprocessing，并编码 full ILP 和 HILP p-ILP。
│       ├── full_ilp.py               # Gurobi-only full-tree ILP planner；按论文 Algorithm 1/2 生成 policy tree 后求解。
│       ├── hilp.py                   # HILP partial-tree search，按 Algorithm 3 维护 E/F frontier 并反复求解 Gurobi p-ILP。
│       ├── rollout.py                # 当前 pyRDDLGym rollout baseline planner。
│       └── session.py                # online session loop 和 trace 结构。
└── tests/
    ├── test_darp_entrypoint.py       # 顶层 CLI 和 pyRDDLGym online trace 测试。
    ├── test_and_or_tree.py           # DARP AND-OR tree 基础结构测试。
    ├── test_duration_sidecar.py      # duration sidecar 和 history duration 测试。
    ├── test_exact_kernel.py          # exact finite kernel、risk sidecar 和 Gaussian percentile duration 测试。
    ├── test_gurobi_ilp.py            # Phase 8 ILP schema、fake Gurobi、full ILP 和 HILP p-ILP 测试。
    ├── test_pyrddlgym_runtime.py     # pyRDDLGym runtime 与 simple online trace 测试。
    └── test_rddl_loader.py           # pyRDDLGym loader、summary 和 grounder 复用测试。
```

## 开发路线图

- [x] Phase 1：项目基础
  - [x] 1.1：README/README-EN、包配置和测试入口
  - [x] 1.2：最小 CLI、duration abstraction 和 pyRDDLGym runtime 测试
  - [x] 1.3：导入 PROST/IPC benchmark 数据集
- [x] Phase 2：pyRDDLGym-first RDDL 输入
  - [x] 2.1：把标准 RDDL parser/simulator 责任切换到 pyRDDLGym
  - [x] 2.2：删除 DARP 自有 parser 主线，统一使用 pyRDDLGym parser
  - [x] 2.3：提供 `PyRDDLGymProblem` summary 和 pyRDDLGym grounder 复用测试
- [x] Phase 3：pyRDDLGym generative runtime adapter
  - [x] 3.1：定义 DARP planner-facing runtime 协议，封装 pyRDDLGym `reset/step/model`
  - [x] 3.2：抽取 type/object/fluent/action metadata，并保留 pyRDDLGym 原生对象引用
  - [x] 3.3：实现初始 bool action 候选、noop/default action、action constraint 错误传播
  - [x] 3.4：实现 MDP/POMDP observation/state/belief 边界；不可枚举时使用采样/粒子接口
  - [x] 3.5：让 `darp --domain --instance` 能通过 pyRDDLGym runtime 执行 online step trace
- [x] Phase 4：pyRDDLGym grounded model view
  - [x] 4.1：封装 pyRDDLGym `RDDLGroundedModel`，暴露 state/action/observation/reward/cpf 读取接口
  - [x] 4.2：从 grounded model 和 runtime 构造 AND-OR tree 所需的 action/observation/history 接口
  - [x] 4.3：明确不支持或暂不支持的 RDDL 结构，并给出清晰错误
- [ ] Phase 5：可验证执行流程
  - [x] 5.1：把 `full-ilp` / `hilp` 论文 planner 接入 online session 和 CLI
  - [x] 5.2：在 CLI/JSON trace 中记录 planner、duration、decision value 和求解耗时
  - [ ] 5.3：调优 HILP/full-ILP 运行效率，使其适合作为默认 planner
  - [ ] 5.4：实现 offline policy JSON、replay 和 evaluation 流程
- [x] Phase 6：DurationModel 与 DARP sidecar
  - [x] 6.1：设计 YAML/JSON duration sidecar schema；sidecar 不包含 `version` / `horizon`
  - [x] 6.2：把 fixed、expected、Gaussian duration 接入 history tree 和 HILP `tau(q)` evaluator
  - [x] 6.3：明确 duration 暂时只由 YAML/JSON sidecar 定义，不修改标准 RDDL grammar
- [x] Phase 7：论文搜索算法骨架
  - [x] 7.1：实现 AND-OR history tree
  - [x] 7.2：实现论文 `Expand` 与 preprocessing
  - [x] 7.3：实现 exact policy-tree preprocessing/Expand scaffolding，为 full-ILP 编码提供常数
  - [x] 7.4：实现 HILP-style partial frontier search 的 E/F bookkeeping
- [x] Phase 8：Gurobi ILP 求解
  - [x] 8.1：实现 finite exact CC-POMDP tree 到 full ILP / p-ILP 的变量、目标和约束编码
  - [x] 8.2：用 Gurobi 作为唯一 solver 求解 exact/full-tree ILP
  - [x] 8.3：用 Gurobi 求解 HILP 当前 `E ∪ F` partial-tree p-ILP，并支持 `one-step-greedy` / `reachable-bellman` frontier heuristic 选择待展开节点
  - [x] 8.4：用 `ILPSolveResult` 记录 Gurobi status、runtime、MIP gap、objective 和 selected variables
  - [x] 8.5：接入 duration sidecar 的 safe-belief chance constraint、Algorithm 2 backward message / smoothed belief 和 Gaussian percentile `tau(q)`
  - [x] 8.6：让 online full-ILP/HILP 使用 `ExactBeliefState` 和 exact Bayes update 维护 root belief，不再依赖 particle belief
- [ ] Phase 9：benchmark、实验与语法扩展
  - [ ] 9.1：实现 benchmark runner 和 pyRDDLGym/rddlrepository 导入检查
  - [ ] 9.2：实现 PROST/rddlsim 风格在线协议兼容层
  - [ ] 9.3：补充论文风格实验脚本
  - [ ] 9.4：扩展 action-space 枚举，支持 concurrent action combination 和非 bool action
  - [ ] 9.5：实现 augmented-state chance-constrained duration，并补齐更复杂随机表达式/连续分布支持
  - [ ] 9.6：如需原生 durative-action 语法，继承 pyRDDLGym parser 实现扩展解析

## 测试

```bash
python -m pytest
```

Phase 8 的单元测试使用 fake `gurobipy` 覆盖 DARP 自己的 ILP 编码和 adapter 边界，因此基础测试不要求本机安装 Gurobi；真实求解实验仍需要 `pip install -e ".[gurobi]"` 和有效 license。

## 当前限制

- RDDL 输入目前通过 pyRDDLGym generative runtime 执行 online trace，DARP 不再维护独立 `PlanningProblem` 编译路线。
- 默认 CLI planner 仍是快速 rollout baseline；`hilp` 已使用 partial-tree p-ILP 避免 full horizon 枚举，但仍需要 benchmark-scale pruning 后才能作为默认 planner；`full-ilp` 会完整展开到 RDDL 剩余 horizon，规模随 action/observation history 指数增长。
- 当前 grounded AND-OR 接口和 pyRDDLGym rollout baseline 只枚举 noop 与单个 bool action；多 action 组合和非 bool action 会给出清晰 unsupported 错误，是后续 planner/action-space 工作。
- rollout baseline 的 POMDP belief 仍采用轻量粒子近似；online full-ILP/HILP 使用 exact Bayes belief update。benchmark 级 POMDP 仍需要更完整的初始 belief 建模、可达状态剪枝和大规模 belief 表示。
- DARP 复用 pyRDDLGym grounding；有限 bool grounded 模型可由 `ExactRDDLKernel` 精确枚举，连续分布、大规模状态空间和复杂随机表达式仍是后续 benchmark 工作。
- DARP-RDDL 原生新语法暂不在 main 分支维护；durative action 当前只通过 YAML/JSON sidecar 接入。
- Phase 8 ILP 已支持有限 transition/observation 的 exact branch、Algorithm 2 smoothed belief、sidecar safe-belief chance constraint 和 duration percentile `tau(q)`；augmented-state chance-constrained duration、benchmark-scale pruning、连续分布和复杂随机表达式留到 Phase 9。
