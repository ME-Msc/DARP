# DARP

Durative Action RDDL Planner 是一个 Python 研究原型，用于把 RDDL 问题解析、编译为 finite-horizon POMDP / C-POMDP / CC-POMDP 风格的规划模型，并逐步实现论文《Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions》中的 full ILP 与 HILP 搜索流程。

当前代码重点是标准 RDDL 输入管线、DARP 内部小规模 simulator、可交互 HTML visualizer，以及本地 PROST-like online solve loop。后续会继续补齐外部 simulator 协议、AND-OR tree、full ILP、HILP、HiGHS/Gurobi backend 和 durative action 接口。

## 功能状态

- 解析标准 RDDL domain/instance 文件，并生成 DARP 自有 `RDDLASTNode` AST。
- 通过 `RDDLFrontend` 统一 `darp`、`pyrddl`、`pyrddlgym` 三类 frontend。
- 将当前支持范围内的 RDDL CPF/reward 表达式 grounding 为最小 `PlanningProblem`。
- 使用 DARP 内部 simulator 执行小规模显式 transition/observation/reward 表。
- 通过 `darp --domain DOMAIN.rddl --instance INSTANCE.rddl` 运行默认非可视化 online solve loop，并在终端打印可读 trace。
- 使用 observation model 做跨 step 贝叶斯 belief carryover，并在硬时间预算耗尽时返回可追踪 fallback action。
- 通过 `--visualizer` 启动实时 HTML，可查看源码、AST，并显示由 DARP planner 选择动作、内部 simulator 推进状态的执行状态机。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --no-build-isolation --no-deps -e .
python -m pip install -r requirements-dev.txt
```

如果系统 Python 缺少 `ensurepip`，可以改用：

```bash
virtualenv --clear .venv
source .venv/bin/activate
python -m pip install --no-build-isolation --no-deps -e .
python -m pip install -r requirements-dev.txt
```

可选依赖：

```bash
python -m pip install -e ".[rddl]"      # pyRDDLGym
python -m pip install -e ".[pyrddl]"    # pyrddl
python -m pip install -e ".[highs]"     # future HiGHS backend
python -m pip install -e ".[gurobi]"    # future Gurobi backend
```

## 命令行

安装为 editable package 后可以使用：

```bash
darp -h
```

也可以不依赖 console script，直接使用：

```bash
python -m darp -h
```

当前只有一条主命令：

```bash
darp [--domain DOMAIN.rddl --instance INSTANCE.rddl] [--visualizer] [options]
```

默认行为：

| 默认项 | 默认值 | 说明 |
| --- | --- | --- |
| mode | `online` | 每步根据当前 belief 重新规划 action。 |
| visualizer | disabled | 默认在命令行打印可读 trace；只有显式传入 `--visualizer` 才打开网页。 |
| simulator | `darp` | 默认使用 DARP 内部 simulator 更新 state、observation 和 reward。 |
| frontend | `darp` | 默认使用 DARP 自有 parser/编译链路。 |
| host | `127.0.0.1` | 默认只监听本机。 |
| port | `0` | 默认自动选择空闲端口。 |

主命令参数说明：

| 参数 | 必需 | 说明 |
| --- | --- | --- |
| `--domain PATH` | 否 | RDDL domain 文件路径；和 `--instance` 同时提供时编译 RDDL，否则非可视化模式使用内置 demo。 |
| `--instance PATH` | 否 | RDDL instance 文件路径；必须和 `--domain` 同时提供。 |
| `--mode online` | 否 | 当前 Phase 3 支持本地 online solve loop，默认 `online`。 |
| `--frontend {darp,pyrddl,pyrddlgym}` | 否 | RDDL parser/compiler frontend，默认 `darp`。 |
| `--simulator {darp,rddlgym,pyrddlgym}` | 否 | visualizer runtime simulator，默认 `darp`；非可视化模式当前只支持 DARP 内部 simulator。 |
| `--seed N` | 否 | DARP 内部 simulator 的随机种子，默认 `0`。 |
| `--host HOST` | 否 | visualizer HTTP host，默认 `127.0.0.1`。 |
| `--port PORT` | 否 | visualizer HTTP port，默认 `0`，表示自动选择空闲端口。 |
| `--no-open` | 否 | 只启动服务，不自动打开浏览器。 |
| `--visualizer` | 否 | 启动实时 HTML visualizer；使用时必须提供 `--domain` 和 `--instance`。 |
| `--time-budget-ms MS` | 否 | 每次决策的硬时间预算，超时时返回 fallback action 并在 trace 中标记。 |
| `--output PATH` | 否 | 非可视化模式下将完整 JSON trace 写入文件；不指定时不会输出 JSON。 |
| `-h`, `--help` | 否 | 显示帮助信息。 |

`--frontend` 和 `--simulator` 不重复：`--frontend` 决定“如何把 RDDL 文本解析/编译成 DARP 可用模型”，`--simulator` 决定“可视化运行时由谁接收 action 并推进 state/observation/reward”。大多数情况下二者都不用写，默认都是 `darp`。

online solve 的执行步数来自 RDDL `horizon` 编译得到的 `problem.max_depth`，不会通过命令行手动截断。

`--seed` 用于控制 DARP 内部 simulator 的随机采样。当前 tiny grid 是确定性的，所以 seed 不改变轨迹；后续支持随机初始 belief、随机 transition 或随机 observation 后，seed 会保证调试、测试和 benchmark 可以复现同一条 sampled trajectory。

## 使用示例

使用 RDDL tiny grid 打印非可视化终端 trace：

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --seed 7
```

使用内置 demo 打印非可视化终端 trace：

```bash
darp --seed 7
```

将完整 JSON trace 写入文件：

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --seed 7 \
  --output tiny_grid_trace.json
```

启动在线可视化：DARP 选择 action，内部 simulator 推进状态，浏览器显示 RDDL 文本、AST 和运行状态机。

```bash
darp \
  --visualizer \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl
```

在固定端口启动可视化，不自动打开浏览器：

```bash
darp \
  --visualizer \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --host 127.0.0.1 \
  --port 8080 \
  --no-open
```

指定 frontend 编译 RDDL，仍然使用 DARP 内部 simulator：

```bash
darp \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --frontend darp
```

可视化时加载 pyRDDLGym simulator，但隐藏 DARP 内部状态机：

```bash
darp \
  --visualizer \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --simulator rddlgym
```

## 架构

- `rddl/`：RDDL parser frontend、加载、AST、表达式 grounding 与 `PlanningProblem` 编译。
- `core/`：当前阶段所需的最小规划问题、类型和 duration 数据结构。
- `online.py`：本地 PROST-like online solve loop、非可视化终端 trace、可选 JSON trace 和有限 horizon 在线 planner。
- `sim/`：DARP 内部 simulator，以及后续外部 simulator 适配层。
- `search/`：后续实现 AND-OR tree、Expand、full ILP 与 HILP 搜索算法。
- `ilp/`：后续实现 internal、HiGHS、Gurobi 等 ILP backend。

`search/` 和 `ilp/` 的区别是：`search/` 负责“如何搜索 policy tree”，`ilp/` 负责“如何求解搜索过程中构造出的 ILP/p-ILP 子问题”。HiGHS/Gurobi 是底层 ILP backend，不是 HILP 的替代算法。

## RDDLFrontend

`RDDLFrontend` 是 DARP 面向 parser 的统一协议。每个 frontend 都返回 `ParsedRDDL`：

- `ast`：DARP 自己的 `RDDLASTNode`，供 compiler 和 visualizer 使用。
- `native_ast`：第三方 parser 的原生 AST，可用于调试或后续继承。
- `env/model`：pyRDDLGym 等 frontend 的可执行对象。
- `metadata`：frontend、版本和解析信息。

当前可选 frontend：

- `darp`：DARP 自有基础 parser，当前默认用于本项目内部链路。
- `pyrddl`：复用 `pyrddl.parser.RDDLParser`。
- `pyrddlgym`：复用 pyRDDLGym 的 parser/simulator 生态。

## 项目结构

```text
DARP/
├── README.md                         # 中文主文档，面向开发和维护。
├── README-EN.md                      # 英文镜像文档，面向国际团队协作。
├── pyproject.toml                    # Python 包元数据、console script 和可选依赖。
├── requirements.txt                  # 运行时依赖记录；当前核心尽量保持轻量。
├── requirements-dev.txt              # 开发和测试依赖记录。
│
├── examples/                         # 示例输入目录。
│   └── rddl/                         # RDDL 示例文件。
│       ├── tiny_grid_domain.rddl     # 3x3 tiny-grid domain，包含 CPF/reward 动态。
│       └── tiny_grid_instance.rddl   # 3x3 tiny-grid instance，包含对象和 horizon 等配置。
│
├── src/
│   └── darp/                         # DARP Python 包主体。
│       ├── __init__.py               # 包版本和顶层元信息。
│       ├── __main__.py               # `darp` 顶层命令入口，根据 `--visualizer` 切换网页可视化或终端 trace。
│       ├── online.py                 # 本地 online solve loop、终端 trace、可选 JSON trace 和有限 horizon 动态规划 planner。
│       │
│       ├── core/                     # 当前阶段的最小规划模型。
│       │   ├── __init__.py           # core 子包入口。
│       │   ├── types.py              # state、action、observation、transition 等共享类型别名。
│       │   ├── duration.py           # 动作时长接口和固定时长模型。
│       │   └── problem.py            # `PlanningProblem` 数据结构和 tiny-grid 内置模型。
│       │
│       ├── rddl/                     # RDDL 解析、加载、编译和可视化。
│       │   ├── __init__.py           # rddl 子包入口。
│       │   ├── ast.py                # DARP 自有 `RDDLASTNode` AST 节点。
│       │   ├── basic_parser.py       # 无第三方依赖的基础 RDDL 结构 parser。
│       │   ├── lexicon.py            # RDDL 关键字、块名和词法符号定义。
│       │   ├── expressions.py        # 标准 RDDL 表达式解析与求值，用于 grounding。
│       │   ├── frontend.py           # `RDDLFrontend` 协议和 `ParsedRDDL` 容器。
│       │   ├── extended.py           # DARP 自有 frontend，并预留未来扩展语法。
│       │   ├── pyrddl_frontend.py    # `pyrddl` frontend 适配。
│       │   ├── pyrddlgym_frontend.py # `pyRDDLGym` frontend 适配。
│       │   ├── loader.py             # 根据 frontend 名称加载 RDDL。
│       │   ├── compiler.py           # 将 `ParsedRDDL` 编译为 `PlanningProblem`。
│       │   └── visualizer.py         # 实时 HTML visualizer 和内部 simulator 状态机面板。
│       │
│       └── sim/                      # simulator 适配层。
│           ├── __init__.py           # sim 子包入口。
│           └── local.py              # 基于显式表的 DARP 内部小规模 simulator。
│
└── tests/                            # 当前阶段测试。
    ├── test_basic_rddl_parser.py     # parser 和 HTML visualizer 测试。
    ├── test_darp_entrypoint.py       # 顶层 `darp` CLI 参数和 `-h` 测试。
    ├── test_rddl_frontends.py        # frontend loader 与第三方 parser 适配测试。
    ├── test_rddl_compiler.py         # RDDL 到 `PlanningProblem` 的编译测试。
    ├── test_rddl_grounding.py        # CPF/reward grounding 行为测试。
    ├── test_local_simulator.py       # DARP 内部 simulator 测试。
    ├── test_online.py                # 本地 online solve loop 和 belief 更新测试。
    └── test_compiler_simulator_interaction.py # compiler 与 simulator 联动测试。
```

## 开发路线图

DARP 的目标不是针对 tiny grid 写死策略，而是逐步实现一个可扩展的 RDDL 求解器。下面路线图已经按实现优先级排序：先让本地求解闭环可靠，再扩大通用 RDDL 建模能力，然后实现可验证 baseline，再实现论文搜索算法，之后接入外部 simulator，最后扩展 durative action 与 DARP-RDDL 新语法。

- [x] Phase 1：项目基础
  - [x] 1.1：项目计划、README/README-EN 和文件结构说明
  - [x] 1.2：Python 包配置、requirements 和 `.venv` 使用方式
  - [x] 1.3：最小 RDDL 示例与 pytest 测试入口
- [x] Phase 2：RDDL 输入管线
  - [x] 2.1：基础 RDDL parser 与交互式 HTML AST visualizer
  - [x] 2.2：通过 `RDDLFrontend` 对齐 `darp`、`pyrddl`、`pyrddlgym`
  - [x] 2.3：将 `ParsedRDDL` 结构化编译为最小 `PlanningProblem`
  - [x] 2.4：补齐标准 RDDL CPF/reward 表达式 grounding，并用 DARP 内部 simulator 验证状态推进
    - [x] 2.4.1：为 tiny grid 补齐 CPF/reward grounding
    - [x] 2.4.2：实现 DARP 内部 simulator 并跑通 tiny grid 小实验
    - [x] 2.4.3：补齐通用标准 RDDL CPF/reward 表达式 grounding，不引入新语法
- [x] Phase 3：PROST-like 实时执行流程
  - [x] 3.1：实现本地 online solve loop：每步 replan、输出 action、接收 observation
  - [x] 3.2：统一 `darp` 顶层入口：默认输出非可视化终端 trace，`--visualizer` 启动网页，`--output` 写 JSON
  - [x] 3.3：完善跨 step 的贝叶斯 belief carryover、初始 observation 采样与硬时间预算 fallback
- [ ] Phase 4：通用 RDDL 问题建模
  - [ ] 4.1：稳定 `PlanningProblem`、typed identifiers 和模型校验
  - [ ] 4.2：支持多个 state fluent 与 factored state，替换当前 one-hot compact state 假设
  - [ ] 4.3：支持随机 CPF、非 identity observation、初始 belief 分布和 action 约束
  - [ ] 4.4：用 pyRDDLGym/rddlsim 小 domain 对照验证 compiler 与 simulator 语义
- [ ] Phase 5：可验证 baseline solver
  - [ ] 5.1：完善显式 state finite-horizon DP baseline，支持 offline policy 与 online replanning
  - [ ] 5.2：实现 planner registry、统一 trace 输出和算法切换参数
  - [ ] 5.3：加入 stochastic/tie-break 策略和 seed-driven reproducibility 测试
- [ ] Phase 6：论文搜索算法
  - [ ] 6.1：在 `and_or_tree.py` 中完善 AND-OR history tree
  - [ ] 6.2：实现论文中的 `Expand` 与 full-tree preprocessing
  - [ ] 6.3：实现 full ILP baseline
  - [ ] 6.4：实现 HILP partial-ILP search
- [ ] Phase 7：ILP 求解层
  - [ ] 7.1：实现 ILP model/backend 协议与内置 backend
  - [ ] 7.2：接入可选 HiGHS backend
  - [ ] 7.3：接入可选 Gurobi backend
- [ ] Phase 8：外部 simulator 与 PROST 兼容
  - [ ] 8.1：设计 rddlsim/PROST 风格 online protocol adapter
  - [ ] 8.2：实现外部 simulator client，并和本地 online planner 共用 action/observation 接口
  - [ ] 8.3：加入外部 simulator 集成测试和 benchmark runner
- [ ] Phase 9：Durative action 与 DARP-RDDL 新语法
  - [ ] 9.1：设计 YAML/JSON sidecar schema 和 compiler/runtime 接口
  - [ ] 9.2：实现 fixed、expected、Gaussian duration model 接入
  - [ ] 9.3：把论文中的 duration/smoothed-belief 约束接入 HILP
  - [ ] 9.4：设计并实现 DARP-RDDL 原生语法扩展
- [ ] Phase 10：输出、接口与实验
  - [ ] 10.1：完善 offline policy JSON 与 trace 输出
  - [ ] 10.2：实现 benchmark 与论文风格实验
  - [ ] 10.3：整理公开 API 与算法 registry

## 测试

```bash
python -m pytest
```

## 当前限制

- 当前 compiler 面向小规模、离散、紧凑 one-hot state fluent 的 RDDL 问题。
- 当前内部 simulator 基于显式 transition/observation/reward 表，不是通用高性能 RDDL simulator。
- 多 state fluent、factored state、大规模/连续 belief 表示、DARP-RDDL 新语法、durative action 原生语法、HILP、HiGHS/Gurobi backend 仍在后续阶段。
