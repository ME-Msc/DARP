# DARP

Durative Action RDDL Planner 是一个 Python 研究原型，目标是在标准 RDDL 问题上实现 fixed-horizon POMDP / constrained POMDP / durative-action 规划算法，并逐步实现论文《Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions》中的 AND-OR tree、full-tree baseline 与 HILP 搜索流程。

main 分支只保留一条实现主线：pyRDDLGym 负责标准 RDDL parser、grounder 和 simulator；DARP 用自己的数据结构组织 AND-OR history tree，把固定 horizon CC-POMDP 搜索问题转换为论文中的 full ILP / HILP p-ILP，并以 Gurobi 作为唯一目标 ILP solver。durative action 当前通过 YAML/JSON sidecar 定义；未来如需扩展 RDDL 语法，再基于 pyRDDLGym parser 做继承式扩展。

## 功能状态

- `adapter/` 使用 pyRDDLGym 加载标准 RDDL domain/instance，并返回 `PyRDDLGymProblem(env, model, native_ast)`。
- `PyRDDLGymProblem.build_grounded_model()` 直接复用 pyRDDLGym `RDDLGrounder(...).ground()`，返回 pyRDDLGym 的 `RDDLGroundedModel`；`GroundedRDDLView` 封装它，DARP 不再自己实现 grounding。
- `GroundedRDDLView.build_and_or_interface()` 已能把 grounded action、observation scope 和 root history 组织成 AND-OR search interface。
- `darp --domain --instance` 默认通过 pyRDDLGym + rollout baseline 执行快速 online trace；`--planner full-ilp` / `--planner hilp` 会切换到论文算法路径。
- `model/` 保留 DARP 自己的 `DurationModel`、duration sidecar 和 AND-OR tree 数据结构，并已接入 Phase 7 的 `tau(q)` 剪枝。
- `planning/` 已提供论文结构对齐的 `preprocess`、`Expand`、full-tree baseline 和 HILP-style partial frontier search；Phase 8 已把生成式 full-tree ILP 与 HILP frontier p-ILP 接到 Gurobi。
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

设置 lookahead 和 particle 数量：

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
  --planner full-ilp \
  --lookahead-depth 4
```

使用 HILP frontier p-ILP planner：

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --duration examples/durations/tiny_grid.yaml \
  --planner hilp \
  --lookahead-depth 3 \
  --hilp-iterations 1 \
  --frontier-width 1
```

`full-ilp` / `hilp` 会在安装 `gurobipy` 时调用 Gurobi；未安装时默认使用生成树 DP fallback 以便调试。使用 `--require-gurobi` 可以要求缺少 Gurobi 时直接失败。

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
- DARP `ilp/`：保存 binary ILP schema 和 Gurobi adapter；`planning/` 负责把生成式 AND-OR tree/frontier 编码成 full ILP 或 HILP p-ILP。

## 文件结构

```text
DARP/
├── README.md                         # 中文主文档。
├── README-EN.md                      # 英文镜像文档。
├── pyproject.toml                    # Python 包配置和 Gurobi solver extra。
├── requirements.txt                  # 运行时依赖，包含 pyRDDLGym。
├── requirements-dev.txt              # 开发/测试依赖。
│
├── examples/
│   ├── benchmarks/                   # PROST/IPC RDDL MDP benchmark 数据集。
│   │   ├── README.md                 # benchmark 来源、目录约定和导入清单。
│   │   └── <domain-year>/            # 单个 benchmark domain 目录。
│   ├── durations/                    # DARP duration sidecar 示例。
│   │   └── tiny_grid.yaml            # tiny grid fixed-duration sidecar。
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
│       ├── preprocess.py             # 论文搜索前处理，初始化 root 和 frontier。
│       ├── expand.py                 # 论文 Expand 操作，计算 rho/u/r/tau 风格指标。
│       ├── ilp_tree.py               # 把生成式 AND-OR tree/frontier 编码为 full ILP 和 HILP p-ILP。
│       ├── full_ilp.py               # Gurobi full-tree ILP planner，缺少 Gurobi 时可 fallback。
│       ├── hilp.py                   # HILP partial frontier search，使用 Gurobi p-ILP 选择 frontier。
│       ├── rollout.py                # 当前 pyRDDLGym rollout baseline planner。
│       └── session.py                # online session loop 和 trace 结构。
└── tests/
    ├── test_darp_entrypoint.py       # 顶层 CLI 和 pyRDDLGym online trace 测试。
    ├── test_and_or_tree.py           # DARP AND-OR tree 基础结构测试。
    ├── test_duration_sidecar.py      # duration sidecar 和 history duration 测试。
    ├── test_gurobi_ilp.py            # Phase 8 ILP schema、fake Gurobi、full ILP 和 HILP p-ILP 测试。
    ├── test_phase7_search.py         # Phase 7 preprocess、Expand、full-tree 和 HILP 测试。
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
  - [x] 5.2：在 CLI/JSON trace 中记录 planner、duration、decision fallback 和 time-budget 状态
  - [ ] 5.3：调优 HILP/full-ILP 运行效率，使其适合作为默认 planner
  - [ ] 5.4：实现 offline policy JSON、replay 和 evaluation 流程
- [x] Phase 6：DurationModel 与 DARP sidecar
  - [x] 6.1：设计 YAML/JSON duration sidecar schema
  - [x] 6.2：把 fixed、expected、Gaussian duration 接入 history tree 和 HILP `tau(q)` evaluator
  - [x] 6.3：明确 duration 暂时只由 YAML/JSON sidecar 定义，不修改标准 RDDL grammar
- [x] Phase 7：论文搜索算法骨架
  - [x] 7.1：实现 AND-OR history tree
  - [x] 7.2：实现论文 `Expand` 与 preprocessing
  - [x] 7.3：实现 generated full-tree DP baseline，作为 Gurobi 缺失时的 fallback 和诊断值
  - [x] 7.4：实现 HILP-style partial frontier search 的 E/F bookkeeping
- [x] Phase 8：Gurobi ILP 求解
  - [x] 8.1：实现生成式 CC-POMDP tree 到 full ILP / p-ILP 的变量、目标和约束编码
  - [x] 8.2：用 Gurobi 求解 generated full-tree ILP，并保留无 Gurobi fallback
  - [x] 8.3：用 Gurobi 求解 HILP 每轮 frontier-selection p-ILP
  - [x] 8.4：用 `ILPSolveResult` 记录 Gurobi status、runtime、MIP gap、objective 和 selected variables
- [ ] Phase 9：benchmark、实验与语法扩展
  - [ ] 9.1：实现 benchmark runner 和 pyRDDLGym/rddlrepository 导入检查
  - [ ] 9.2：实现 PROST/rddlsim 风格在线协议兼容层
  - [ ] 9.3：补充论文风格实验脚本
  - [ ] 9.4：如需原生 durative-action 语法，继承 pyRDDLGym parser 实现扩展解析

## 测试

```bash
python -m pytest
```

Phase 8 的单元测试使用 fake `gurobipy` 覆盖 DARP 自己的 ILP 编码和 adapter 边界，因此基础测试不要求本机安装 Gurobi；真实求解实验仍需要 `pip install -e ".[gurobi]"` 和有效 license。

## 当前限制

- RDDL 输入目前通过 pyRDDLGym generative runtime 执行 online trace，DARP 不再维护独立 `PlanningProblem` 编译路线。
- 默认 CLI planner 仍是快速 rollout baseline；`full-ilp` / `hilp` 已接入 online session，但在 pyRDDLGym deepcopy 和生成树展开下还需要性能调优后才能作为默认 planner。
- 当前 grounded AND-OR 接口和 pyRDDLGym rollout baseline 只枚举 noop 与单个 bool action；多 action 组合和非 bool action 会给出清晰 unsupported 错误，是后续 planner/action-space 工作。
- 当前 POMDP belief 采用轻量粒子近似，适合调试 runtime 边界；benchmark 级 POMDP 评估需要后续 likelihood weighting/resampling。
- DARP 复用 pyRDDLGym grounding，不重新实现 RDDL grounding 或有限状态枚举。
- DARP-RDDL 原生新语法暂不在 main 分支维护；durative action 当前只通过 YAML/JSON sidecar 接入。
- Phase 8 ILP 目前基于 pyRDDLGym 生成/采样到的 observation branch；完整 stochastic observation 枚举、risk/cost fluent 抽取和 benchmark-scale constrained rows 留到 Phase 9。
