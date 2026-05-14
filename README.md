# DARP

Durative Action RDDL Planner 是一个 Python 研究原型，目标是在标准 RDDL 问题上实现 fixed-horizon POMDP / constrained POMDP / durative-action 规划算法，并逐步实现论文《Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions》中的 AND-OR tree、full ILP 与 HILP 搜索流程。

main 分支采用 **pyRDDLGym-first** 架构：标准 RDDL 的解析、语义检查、grounding 和仿真能力交给 pyRDDLGym；DARP 专注把 pyRDDLGym 的 `model/env/grounded_model` 组织成求解器可用的 runtime 与 search 输入，再接入 `DurationModel`、AND-OR tree、ILP/full ILP/HILP 和在线 session。DARP 自有 parser、表达式 parser、AST visualizer 和 EBNF 已保存在 archive 分支，不再在 main 上继续维护。

## 功能状态

- `adapter/` 使用 pyRDDLGym 加载标准 RDDL domain/instance，并返回 `PyRDDLGymProblem(env, model, native_ast)`。
- `PyRDDLGymProblem.build_grounded_model()` 直接复用 pyRDDLGym `RDDLGrounder(...).ground()`，返回 pyRDDLGym 的 `RDDLGroundedModel`；`GroundedRDDLView` 封装它，DARP 不再自己实现 grounding。
- `GroundedRDDLView.build_and_or_interface()` 已能把 grounded action、observation scope 和 root history 组织成 AND-OR search interface。
- `darp --domain --instance` 已能通过 pyRDDLGym 执行 online trace；当前 planner 是小规模 rollout baseline，用于验证 runtime/session 主路径。
- `model/` 保留 DARP 自己的 `DurationModel`、duration sidecar 和 AND-OR tree 数据结构，为后续 HILP 的 `tau(q)` 计算服务。
- 后续 AND-OR tree / ILP / HILP 将直接消费 pyRDDLGym runtime 与 grounded model view，而不是先转换成 DARP 自有 `PlanningProblem`。
- DARP-RDDL 原生扩展语法暂不在 main 实现；durative action 优先通过 YAML/JSON sidecar 或 Python plugin 接入。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements-dev.txt
```

可选求解器依赖仍保留为未来 ILP backend 使用：

```bash
python -m pip install -e ".[highs]"     # future HiGHS backend
python -m pip install -e ".[gurobi]"    # future Gurobi backend
```

## 命令行

查看帮助：

```bash
darp -h
```

加载标准 RDDL，并用 pyRDDLGym 执行 online trace：

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl
```

设置 rollout lookahead：

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --lookahead-depth 4 \
  --particles 32
```

将在线 trace 写成 JSON：

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --output tiny_grid_pyrddlgym_trace.json
```

直接检查 pyRDDLGym problem 组件和未来 search 边界：

```bash
python -m darp.adapter.loader \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl
```

## Duration Sidecar

Phase 6 不修改标准 RDDL grammar，而是用 YAML/JSON sidecar 描述动作时长。当前支持：

- `fixed`：每个 action 一个固定 duration。
- `expected`：按 belief 对 `(state, action)` duration 做期望。
- `gaussian`：按 belief 累计 mean/variance，并用 `tau(q)` 计算 horizon 可行度。
- `plugin`：通过 Python factory 返回自定义 `DurationModel`。

如果环境安装了 PyYAML，DARP 会用 PyYAML 读取 YAML；否则使用内置的 mapping-only YAML 子集解析器，足够读取项目 sidecar schema。

示例：

```yaml
version: 1
duration_model:
  kind: fixed
  horizon: 8
  default: 1
  actions:
    move-east: 1
    move-south: 1
```

Python 中加载：

```python
from darp.model.duration_sidecar import load_duration_sidecar

sidecar = load_duration_sidecar("examples/durations/tiny_grid.yaml")
evaluator = sidecar.evaluator()
```

## 执行流程

当前主路径刻意保持简单：

```text
darp CLI
  -> adapter.RDDLLoader.load(domain, instance)
  -> pyRDDLGym.make(...)
  -> PyRDDLGymProblem(env, model, native_ast)
  -> adapter.PyRDDLGymRuntime.reset/step
  -> planning.RolloutPlanner.choose_action
  -> planning.run_online_session 管理 action、observation、state、belief 和 trace
```

后续论文算法主路径：

```text
PyRDDLGymProblem.build_grounded_model()
  -> pyRDDLGym.core.compiler.model.RDDLGroundedModel
  -> adapter.GroundedRDDLView（复用 pyRDDLGym grounding 结果）
  -> model.ANDORNode / History
  -> ILP/full ILP/HILP planner
  -> OnlineSession 每步 take action 并从 pyRDDLGym env 接收 observation/reward/state
```

## 架构边界

- pyRDDLGym：负责标准 RDDL parser、semantic handling、grounder、Gym-style environment 和 simulator。
- DARP `adapter/`：隔离 pyRDDLGym 依赖，负责 loading、grounded-model view、runtime reset/step 和轻量 belief helper；如果未来接入其他外部系统，再拆成多个 adapter。
- DARP `model/`：保存 DARP 自己的规划数据结构，例如 duration model、history、AND-OR tree node；这里不直接依赖 pyRDDLGym。
- DARP `planning/`：保存 planner、online session、trace 与后续 planner registry；这里通过 adapter/runtime 接触外部仿真器。
- 未来 `search/` / `ilp/` 或 `planning/search/` / `planning/ilp/`：直接基于 `GroundedRDDLView`、runtime 和 `model/` 数据结构实现 AND-OR tree、full ILP、HILP、HiGHS/Gurobi backend。

## 文件结构

```text
DARP/
├── README.md                         # 中文主文档。
├── README-EN.md                      # 英文镜像文档。
├── pyproject.toml                    # Python 包配置和可选求解器依赖。
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
│   │   └── duration_sidecar.py       # JSON/YAML duration sidecar loader 和 plugin 入口。
│   └── planning/                     # planner 与在线执行编排。
│       ├── __init__.py               # planning package 入口。
│       ├── rollout.py                # 当前 pyRDDLGym rollout baseline planner。
│       └── session.py                # online session loop 和 trace 结构。
└── tests/
    ├── test_darp_entrypoint.py       # 顶层 CLI 和 pyRDDLGym online trace 测试。
    ├── test_and_or_tree.py           # DARP AND-OR tree 基础结构测试。
    ├── test_duration_sidecar.py      # duration sidecar、plugin 和 history duration 测试。
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
  - [x] 2.2：移除 main 上的 DARP 自有 parser/AST/expression/visualizer 维护面，并保存在 archive 分支
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
- [ ] Phase 5：可验证 baseline solver（vNext，当前不阻塞 Phase 6/7）
  - [ ] 5.1：为 rollout、AND-OR、full ILP、HILP 增加 planner registry
  - [ ] 5.2：实现统一 trace 输出、time-budget fallback 和 trace formatter
  - [ ] 5.3：实现 offline policy JSON、replay 和 evaluation 流程
  - [ ] 说明：Phase 5 会在至少出现 AND-OR/full ILP/HILP 中的一个新 planner 后推进，避免当前只有 rollout baseline 时过早工程化。
- [x] Phase 6：DurationModel 与 DARP sidecar
  - [x] 6.1：设计 YAML/JSON duration sidecar schema
  - [x] 6.2：把 fixed、expected、Gaussian duration 接入 history tree 和 HILP `tau(q)` evaluator
  - [x] 6.3：预留 Python plugin 接口，不修改标准 RDDL grammar
- [ ] Phase 7：论文搜索算法
  - [ ] 7.1：实现 AND-OR history tree
  - [ ] 7.2：实现论文 `Expand` 与 preprocessing
  - [ ] 7.3：实现 full ILP baseline
  - [ ] 7.4：实现 HILP partial-ILP search
- [ ] Phase 8：ILP backend
  - [ ] 8.1：实现 ILP model/backend 协议与内置 backend
  - [ ] 8.2：接入 HiGHS
  - [ ] 8.3：接入 Gurobi
- [ ] Phase 9：benchmark 与 PROST/rddlsim 兼容
  - [ ] 9.1：实现 benchmark runner 和 pyRDDLGym/rddlrepository 导入检查
  - [ ] 9.2：实现 rddlsim/PROST 风格 online protocol adapter
  - [ ] 9.3：补充论文风格实验脚本
  - [ ] 9.4：评估 pyRDDLGym visualizer 与 DARP planner trace 的集成方式

## 测试

```bash
python -m pytest
```

## 当前限制

- RDDL 输入目前通过 pyRDDLGym generative runtime 执行 online trace，DARP 不再维护独立 `PlanningProblem` 编译路线。
- 当前 grounded AND-OR 接口和 pyRDDLGym rollout baseline 只枚举 noop 与单个 bool action；多 action 组合和非 bool action 会给出清晰 unsupported 错误，是后续 planner/action-space 工作。
- 当前 POMDP belief 采用轻量粒子近似，适合调试 runtime 边界；benchmark 级 POMDP 评估需要后续 likelihood weighting/resampling。
- DARP 复用 pyRDDLGym grounding，不重新实现 RDDL grounding 或有限状态枚举。
- DARP-RDDL 原生新语法暂不在 main 分支维护；durative action 已通过 sidecar/plugin 接入。
- AND-OR tree、full ILP、HILP、HiGHS/Gurobi backend 仍是后续阶段。
