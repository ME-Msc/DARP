# DARP

Durative Action RDDL Planner 是一个 Python 研究原型，目标是在标准 RDDL 问题上实现 fixed-horizon POMDP / constrained POMDP / durative-action 规划算法，并逐步实现论文《Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions》中的 AND-OR tree、full ILP 与 HILP 搜索流程。

main 分支采用 **pyRDDLGym-first** 架构：标准 RDDL 的解析、基础语义检查和仿真能力交给 pyRDDLGym；DARP 专注把 pyRDDLGym 的 `model/env` 适配为 planner runtime，并在小规模可枚举问题上生成研究用 `PlanningProblem`，再接入 `DurationModel`、在线/离线规划算法。DARP 自有 parser、表达式 parser、AST visualizer 和 EBNF 已保存在 archive 分支，不再在 main 上继续维护。

## 功能状态

- 使用 pyRDDLGym 加载标准 RDDL domain/instance，并返回 `RDDLEnv`、`RDDLLiftedModel` 和 pyRDDLGym 原生 AST。
- `darp --domain --instance` 已能通过 pyRDDLGym 执行 online trace；当前 planner 是小规模 rollout baseline，用于验证 runtime 主路径。
- 提供 `PyRDDLGymPlanningAdapter` 接口边界；下一步优先封装 pyRDDLGym 的 generative runtime，再只对小规模离散问题做可选枚举。
- 保留 DARP 内部显式 `PlanningProblem`、`DurationModel` 和 finite-horizon DP/belief helper，用于未来可枚举 RDDL、ILP/HILP 与算法单元测试。
- 当前 RDDL 输入先走 pyRDDLGym generative runtime；从 pyRDDLGym 模型生成 DARP `PlanningProblem` 尚未实现。
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

直接检查 pyRDDLGym artifacts：

```bash
python -m darp.rddl.compiler \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl
```

## 架构边界

- pyRDDLGym：负责标准 RDDL parser、semantic handling、Gym-style environment 和基础 simulator 能力。
- DARP `rddl/`：只负责加载 pyRDDLGym artifacts，并预留 pyRDDLGym runtime 与小规模 `model/env -> PlanningProblem` 的适配边界。
- DARP `core/`：保存 planner 使用的显式有限 horizon 模型、typed identifiers 和 duration abstraction。
- DARP `online.py`：保留显式 `PlanningProblem` 上的 DP/belief helper，供未来可枚举 RDDL 和算法测试使用。
- 未来 planner 接口：优先支持 pyRDDLGym generative step/reset，再在可枚举的小规模离散问题上生成显式 `PlanningProblem`。
- 未来 `search/` / `ilp/`：在稳定 planner 接口上实现 AND-OR tree、full ILP、HILP、HiGHS/Gurobi backend。

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
│   └── rddl/                         # 小型手写 RDDL 示例。
│       ├── tiny_grid_domain.rddl     # tiny grid 标准 RDDL domain。
│       ├── tiny_grid_instance.rddl   # tiny grid instance。
│       ├── factored_door_domain.rddl # partial-observation toy domain，供未来 adapter 回归使用。
│       └── factored_door_instance.rddl # factored door instance。
│
├── src/darp/
│   ├── __init__.py                   # 包版本入口。
│   ├── __main__.py                   # `darp` 顶层命令入口。
│   ├── online.py                     # 显式 PlanningProblem 上的 finite-horizon online replanning。
│   │
│   ├── core/
│   │   ├── __init__.py               # core 子包入口。
│   │   ├── duration.py               # DurationModel 接口和 fixed/expected/Gaussian duration 模型。
│   │   ├── problem.py                # DARP 显式 PlanningProblem 模型。
│   │   └── types.py                  # State、Action、Observation 等类型别名。
│   │
│   ├── rddl/
│   │   ├── __init__.py               # pyRDDLGym-backed RDDL 子包入口。
│   │   ├── artifacts.py              # RDDLArtifacts 容器和 RDDLLoadError。
│   │   ├── loader.py                 # 使用 pyRDDLGym 加载标准 RDDL 并返回 artifacts。
│   │   ├── runtime.py                # pyRDDLGym reset/step runtime 和 rollout online trace。
│   │   └── compiler.py               # pyRDDLGym runtime 与可选 PlanningProblem 枚举的未来 adapter 边界。
└── tests/
    ├── test_darp_entrypoint.py       # 顶层 CLI 和 pyRDDLGym online trace 测试。
    ├── test_online.py                # online replanning 和 belief 更新测试。
    ├── test_pyrddlgym_runtime.py     # pyRDDLGym runtime 与 simple online trace 测试。
    └── test_rddl_loader.py           # pyRDDLGym loader 与未来 adapter 边界测试。
```

## 开发路线图

- [x] Phase 1：项目基础
  - [x] 1.1：README/README-EN、包配置和测试入口
  - [x] 1.2：显式 `PlanningProblem`、belief helper 和 online DP 单元测试
  - [x] 1.3：导入 PROST/IPC benchmark 数据集
- [x] Phase 2：pyRDDLGym-first RDDL 输入
  - [x] 2.1：把标准 RDDL parser/simulator 责任切换到 pyRDDLGym
  - [x] 2.2：移除 main 上的 DARP 自有 parser/AST/expression/visualizer 维护面，并保存在 archive 分支
  - [x] 2.3：提供 pyRDDLGym artifact summary 和 adapter 边界测试
- [x] Phase 3：pyRDDLGym generative runtime adapter
  - [x] 3.1：定义 DARP planner-facing runtime 协议，封装 pyRDDLGym `reset/step/model`
  - [x] 3.2：抽取 type/object/fluent/action metadata，并保留 pyRDDLGym 原生对象引用
  - [x] 3.3：实现初始 bool action 候选、noop/default action、action constraint 错误传播
  - [x] 3.4：实现 MDP/POMDP observation/state/belief 边界；不可枚举时使用采样/粒子接口
  - [x] 3.5：让 `darp --domain --instance` 能通过 pyRDDLGym runtime 执行 online step trace
- [ ] Phase 4：小规模离散 RDDL 到显式 `PlanningProblem`
  - [ ] 4.1：实现有限离散可枚举性检查；连续、过大或未支持结构要清晰报错
  - [ ] 4.2：对 deterministic / 已支持有限随机结构抽取 reward、transition、observation table
  - [ ] 4.3：把可枚举 RDDL 转为 `PlanningProblem`，并用 tiny/factored 示例回归测试
- [ ] Phase 5：可验证 baseline solver
  - [ ] 5.1：让 baseline planner 同时支持 pyRDDLGym generative runtime 和显式 `PlanningProblem`
  - [ ] 5.2：实现 planner registry、统一 trace 输出和 time-budget fallback
  - [ ] 5.3：实现 offline policy JSON 与 replay/evaluation 流程
- [ ] Phase 6：DurationModel 与 DARP sidecar
  - [ ] 6.1：设计 YAML/JSON duration sidecar schema
  - [ ] 6.2：把 fixed、expected、Gaussian duration 接入 runtime adapter 与显式 `PlanningProblem`
  - [ ] 6.3：预留 Python plugin 接口，不修改标准 RDDL grammar
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

- RDDL 输入目前通过 pyRDDLGym generative runtime 执行 online trace，不会生成 DARP `PlanningProblem`。
- pyRDDLGym rollout baseline 目前只枚举 noop 与单个 bool action；多 action 组合和非 bool action 是后续 planner/action-space 工作。
- 当前 POMDP belief 采用轻量粒子近似，适合调试 runtime 边界；benchmark 级 POMDP 评估需要后续 likelihood weighting/resampling。
- 通用 RDDL 不能假设可以完整枚举成表格 MDP/POMDP；main 分支先实现 generative runtime，小规模离散问题再走显式枚举路径。
- 显式 `PlanningProblem` DP helper 仅用于算法单元测试和未来小规模枚举路径，不替代 pyRDDLGym runtime。
- DARP-RDDL 原生新语法暂不在 main 分支维护；durative action 优先通过 sidecar/plugin 实现。
- AND-OR tree、full ILP、HILP、HiGHS/Gurobi backend 仍是后续阶段。
