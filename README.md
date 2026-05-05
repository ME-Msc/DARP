# DARP

Durative Action RDDL Planner 是一个 Python 研究原型，用于把 RDDL 问题解析、编译为 finite-horizon POMDP / C-POMDP / CC-POMDP 风格的规划模型，并逐步实现论文《Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions》中的 full ILP 与 HILP 搜索流程。

当前代码重点是标准 RDDL 输入管线、DARP 内部小规模 simulator、可交互 HTML visualizer，以及本地 PROST-like online solve loop。后续会继续补齐外部 simulator 协议、AND-OR tree、full ILP、HILP、HiGHS/Gurobi backend 和 durative action 接口。

## 功能状态

- 解析标准 RDDL domain/instance 文件，并生成 DARP 自有 `RDDLASTNode` AST。
- 通过 `RDDLFrontend` 统一 `darp`、`pyrddl`、`pyrddlgym` 三类 frontend。
- 将当前支持范围内的 RDDL CPF/reward 表达式 grounding 为最小 `PlanningProblem`。
- 使用 DARP 内部 simulator 执行小规模显式 transition/observation/reward 表。
- 通过顶层 `darp --visualizer` 启动实时 HTML，可查看源码、AST，并可选显示由 DARP online planner 选择动作、内部 simulator 推进状态的执行状态机。
- 通过 `darp solve --mode online` 运行本地在线循环：每步根据当前 belief 重新选择 action，再由 simulator 返回 observation/reward。

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

当前顶层入口的核心形式是：

```bash
darp --visualizer --domain DOMAIN.rddl --instance INSTANCE.rddl [options]
darp solve --mode online [--domain DOMAIN.rddl --instance INSTANCE.rddl] [options]
```

visualizer 参数说明：

| 参数 | 必需 | 说明 |
| --- | --- | --- |
| `--visualizer` | 是 | 启动实时 HTML visualizer。 |
| `--domain PATH` | 是 | RDDL domain 文件路径。 |
| `--instance PATH` | 是 | RDDL instance 文件路径。 |
| `--with-simulator [darp\|rddlgym\|pyrddlgym]` | 否 | 启用 simulator；不写值时默认 `darp`，显示 DARP 内部状态机；指定 `rddlgym`/`pyrddlgym` 时加载 pyRDDLGym，但不显示 DARP 内部状态机。 |
| `--frontend {darp,pyrddl,pyrddlgym}` | 否 | DARP 内部 simulator 编译 RDDL 时使用的 frontend，默认 `darp`。 |
| `--host HOST` | 否 | visualizer HTTP host，默认 `127.0.0.1`。 |
| `--port PORT` | 否 | visualizer HTTP port，默认 `0`，表示自动选择空闲端口。 |
| `--no-open` | 否 | 只启动服务，不自动打开浏览器。 |
| `-h`, `--help` | 否 | 显示帮助信息。 |

online solve 参数说明：

| 参数 | 必需 | 说明 |
| --- | --- | --- |
| `solve` | 是 | 运行命令行求解流程。 |
| `--mode online` | 否 | 当前 Phase 3.1 支持本地 online solve loop，默认 `online`。 |
| `--domain PATH` | 否 | RDDL domain 文件路径；和 `--instance` 同时提供时编译 RDDL，否则使用内置 tiny demo。 |
| `--instance PATH` | 否 | RDDL instance 文件路径；必须和 `--domain` 同时提供。 |
| `--frontend {darp,pyrddl,pyrddlgym}` | 否 | 编译显式 RDDL 输入时使用的 frontend，默认 `darp`。 |
| `--steps N` | 否 | 最大在线决策步数；默认使用 `problem.max_depth`。 |
| `--seed N` | 否 | DARP 内部 simulator 的随机种子，默认 `0`。 |
| `--time-budget-ms MS` | 否 | 记录到 JSON trace 中的软单步时间预算。 |
| `--output PATH` | 否 | 将 JSON trace 写入文件，同时仍在终端打印。 |

## 使用示例

只查看源码和 AST：

```bash
darp \
  --visualizer \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl
```

启动 DARP 内部 simulator；右侧 HTML 面板只推进环境，action 由 DARP online planner 根据当前 belief 选择：

```bash
darp \
  --visualizer \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --with-simulator
```

指定 frontend 编译 RDDL，再使用 DARP 内部 simulator：

```bash
darp \
  --visualizer \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --frontend darp \
  --with-simulator darp
```

加载 pyRDDLGym simulator，但隐藏 DARP 内部状态机：

```bash
darp \
  --visualizer \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --with-simulator rddlgym
```

在固定端口启动，不自动打开浏览器：

```bash
darp \
  --visualizer \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --with-simulator \
  --host 127.0.0.1 \
  --port 8080 \
  --no-open
```

使用内置 demo 运行本地 online solve loop：

```bash
darp solve --mode online --steps 2 --seed 7
```

使用 RDDL tiny grid 运行本地 online solve loop：

```bash
darp solve \
  --mode online \
  --domain examples/rddl/tiny_grid_domain.rddl \
  --instance examples/rddl/tiny_grid_instance.rddl \
  --steps 4 \
  --seed 7
```

## 架构

- `rddl/`：RDDL parser frontend、加载、AST、表达式 grounding 与 `PlanningProblem` 编译。
- `core/`：当前阶段所需的最小规划问题、类型和 duration 数据结构。
- `online.py`：本地 PROST-like online solve loop 和有限 horizon 在线 planner。
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
│       ├── __main__.py               # `darp` 顶层命令入口，负责解析 `--visualizer` 等参数。
│       ├── online.py                 # 本地 online solve loop 和有限 horizon 动态规划 planner。
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
- [ ] Phase 3：PROST-like 实时执行流程
  - [x] 3.1：实现本地 online solve loop：每步 replan、输出 action、接收 observation
  - [ ] 3.2：接入 rddlsim/PROST 风格外部 simulator 协议
  - [ ] 3.3：完善跨 step 的 belief/state carryover 与硬时间预算控制
- [ ] Phase 4：规划核心模型
  - [ ] 4.1：稳定 `PlanningProblem`、typed identifiers 和模型校验
  - [ ] 4.2：完善 history、belief、constraints 和 policy tree 基础结构
  - [ ] 4.3：扩展多约束、chance-risk 与连续/大状态空间接口
- [ ] Phase 5：搜索算法
  - [ ] 5.1：在 `and_or_tree.py` 中完善 AND-OR history tree
  - [ ] 5.2：实现论文中的 `Expand` 与 full-tree preprocessing
  - [ ] 5.3：实现 full ILP baseline
  - [ ] 5.4：实现 HILP partial-ILP search
- [ ] Phase 6：ILP 求解层
  - [ ] 6.1：实现 ILP model/backend 协议与内置 backend
  - [ ] 6.2：接入可选 HiGHS backend
  - [ ] 6.3：接入可选 Gurobi backend
- [ ] Phase 7：Durative action sidecar
  - [ ] 7.1：设计 YAML/JSON sidecar schema 和 compiler/runtime 接口
  - [ ] 7.2：实现 fixed、expected、Gaussian duration model 接入
  - [ ] 7.3：把论文中的 duration/smoothed-belief 约束接入 HILP
- [ ] Phase 8：DARP-RDDL 新语法
  - [ ] 8.1：设计 DARP-RDDL 语法扩展
  - [ ] 8.2：决定并实现 parser 继承、fork 或自研 grammar
  - [ ] 8.3：把 sidecar 能力迁移为可选原生语法
- [ ] Phase 9：输出、接口与实验
  - [ ] 9.1：完善 offline policy JSON 与 trace 输出
  - [ ] 9.2：实现 benchmark 与论文风格实验
  - [ ] 9.3：整理公开 API 与算法 registry

## 测试

```bash
python -m pytest
```

## 当前限制

- 当前 compiler 面向小规模、离散、紧凑 one-hot state fluent 的 RDDL 问题。
- 当前内部 simulator 基于显式 transition/observation/reward 表，不是通用高性能 RDDL simulator。
- DARP-RDDL 新语法、durative action 原生语法、HILP、HiGHS/Gurobi backend 仍在后续阶段。
