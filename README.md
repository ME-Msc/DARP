# DARP

Durative Action RDDL Planner 是一个面向研究原型的 Python 项目，目标是从 RDDL 问题出发，构建论文《Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions》中使用的 finite-horizon (C)C-POMDP + durative action 模型，并提供 full ILP 与 HILP 两类策略搜索入口。

## 项目目标

DARP 第一版优先实现一条可运行、可读、可扩展的研究链路：

- 使用 Python 表达 POMDP / C-POMDP / CC-POMDP、belief、history、duration 和 policy tree。
- 使用 AND-OR history tree 表达论文中的 `O~` observation histories 与 `A~` action histories。
- 实现论文中的 `Expand`、full ILP baseline 和 HILP partial-ILP search。
- 默认使用项目内置的小规模 ILP backend 独立运行，后续可选接入 HiGHS 和 Gurobi。
- 支持 offline policy tree 输出，并预留 online replanning 接口。

## 论文算法对应关系

- `core/history.py` 对应论文中的 action-observation history `q`。
- `search/and_or_tree.py` 对应论文图 1 和图 2 的 AND-OR tree。
- `search/expand.py` 对应论文 Algorithm 2 `Expand`，负责计算 `u_q`、`r_q`、`rho(q)`、belief 与 `tau(q)`。
- `search/preprocess.py` 对应 full ILP 需要的完整 preprocessing。
- `search/full_ilp.py` 构建并求解完整 ILP。
- `search/hilp.py` 对应论文 Algorithm 3 `HILP`，逐步扩展 frontier 并反复求解 p-ILP。
- `ilp/` 只负责表达和求解 ILP/p-ILP 子问题。

## 架构说明

`rddl/`、`search/` 和 `ilp/` 是三层不同职责：

- `rddl/` 是解析和编译入口，回答“如何把标准 RDDL 或 DARP-RDDL 扩展语法变成统一的中间表示”。
- `search/` 是规划/搜索算法层，回答“如何探索 policy tree”。HILP、full ILP wrapper、online replanning 都属于这一层。
- `ilp/` 是数学规划子问题层，回答“给定一个 ILP/p-ILP 模型如何求解”。internal、HiGHS、Gurobi 都是这一层的 backend。

因此，HiGHS 和 Gurobi 不是 HILP 的替代算法，而是 HILP 在每一轮 p-ILP 中可以调用的底层求解器。

### RDDLFrontend 协议

`RDDLFrontend` 是 DARP 面向 parser 的统一协议，不是一个具体 parser。它规定任何解析器都要实现：

- `name`：frontend 名称，例如 `pyrddlgym`、`pyrddl`、`darp`。
- `supports_extended_syntax`：是否支持 DARP 扩展语法。
- `parse(domain, instance) -> ParsedRDDL`：把 domain/instance 解析成统一容器。

`ParsedRDDL` 是 compiler 看到的唯一输入形状，里面可以装 `ast`、`model`、`env` 和 metadata。这样后续可以先复用 `pyrddl`，也可以复用或继承 `pyRDDLGym` 的 parser；如果将来 DARP 自己实现 parser，只要仍然返回 `ParsedRDDL`，后面的 `compiler.py`、`core/`、`search/`、`ilp/` 都不需要跟着改。

当前预留三种 frontend：

- `pyrddlgym`：默认标准 RDDL frontend，适合复用 pyRDDLGym 的 parser/simulator 生态。
- `pyrddl`：直接调用旧 `pyrddl.parser.RDDLParser`，适合作为 DARP 自研 parser 或 fork 的起点。
- `darp`：DARP 自有基础 parser frontend，当前可解析 RDDL 的文件/块/语句结构，并为后续 DARP-RDDL 扩展语法预留入口。

基础 parser 可以在命令行中直接验证解析结果，并输出 Graphviz DOT 格式的 AST：

```bash
python -m darp.rddl.basic_parser \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl \
  --dot
```

如需保存 DOT 文件：

```bash
python -m darp.rddl.basic_parser \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl \
  --dot-output tiny_grid_ast.dot
```

## 文件和文件夹职责

```text
DARP/
├── README.md                       # 中文主文档，说明项目目标、论文对应关系、架构、路线图和运行方式。
├── README-EN.md                    # 英文镜像文档，供团队其他成员阅读。
├── LICENSE                         # 项目的 Apache-2.0 许可证文本。
├── .gitignore                      # 忽略 Python 缓存、虚拟环境、构建产物和本地配置。
├── .codex                          # 本地 Codex 工作配置占位文件。
├── pyproject.toml                  # Python 包元数据、依赖、CLI 入口和可选 backend extras。
├── requirements.txt                # 记录运行时核心依赖；当前核心只依赖 Python 标准库。
├── requirements-dev.txt            # 记录开发和测试依赖，例如 pytest。
│
├── examples/                       # 保存最小 RDDL 示例和 duration 配置，用于 demo 和测试。
│   ├── rddl/                       # 保存 RDDL domain 与 instance 文件。
│   │   ├── tiny_grid_domain.rddl   # 用于演示的 tiny grid RDDL domain 占位示例。
│   │   └── tiny_grid_instance.rddl # 用于演示的 tiny grid RDDL instance 占位示例。
│   └── durations/                  # 保存 durative action 的 sidecar 配置文件。
│       └── tiny_grid.yaml          # 定义 tiny grid 中动作持续时间的 sidecar 配置。
│
├── src/
│   └── darp/                       # DARP 的主 Python 包。
│       ├── __init__.py             # 定义包版本和顶层导出。
│       ├── cli.py                  # 命令行入口，提供 solve 和 evaluate 模式。
│       │
│       ├── rddl/                   # RDDL parser frontend、加载与编译相关代码。
│       │   ├── __init__.py         # 标记 RDDL 子包并保留 parser/编译阶段 TODO。
│       │   ├── ast.py              # 定义基础 RDDL AST 节点和 Graphviz DOT 导出。
│       │   ├── basic_parser.py     # 实现无第三方依赖的基础 RDDL 结构 parser 和命令行入口。
│       │   ├── frontend.py         # 定义 RDDLFrontend 协议和 ParsedRDDL 统一容器。
│       │   ├── pyrddlgym_frontend.py # 复用 pyRDDLGym 解析标准 RDDL 并可返回环境对象。
│       │   ├── pyrddl_frontend.py  # 复用 pyrddl.parser.RDDLParser 直接生成 AST。
│       │   ├── extended.py         # 使用 DARP 自有 parser，并预留未来 DARP-RDDL 扩展语法。
│       │   ├── loader.py           # 根据 --rddl-frontend 选择具体 parser frontend。
│       │   ├── compiler.py         # 将 ParsedRDDL 编译为 DARP 的 PlanningProblem。
│       │   └── durations.py        # 读取 duration sidecar 配置。
│       │
│       ├── core/                   # 与具体求解算法无关的规划数据结构。
│       │   ├── __init__.py         # 标记 core 子包并预留稳定 API 导出。
│       │   ├── types.py            # 集中定义状态、动作、观测和概率分布等类型别名。
│       │   ├── problem.py          # 定义 finite-horizon POMDP/(C)C-POMDP 问题接口。
│       │   ├── history.py          # 定义论文中的 observation history 和 action history。
│       │   ├── belief.py           # 实现 belief update、safe belief 和风险概率计算。
│       │   ├── duration.py         # 实现 fixed、expected/state-dependent 和 Gaussian percentile duration 模型。
│       │   ├── constraints.py      # 定义 expected-cost 与 chance-risk 约束。
│       │   └── policy.py           # 表示求解结果、动作序列、policy tree 和 JSON 导出结构。
│       │
│       ├── search/                 # 规划算法层，负责搜索和展开 policy space。
│       │   ├── __init__.py         # 标记 search 子包并预留算法注册信息。
│       │   ├── base.py             # 定义通用 Planner 接口。
│       │   ├── and_or_tree.py      # 定义共享的 AND-OR history tree 结构。
│       │   ├── expand.py           # 实现论文 Expand 步骤并计算 ILP 常量。
│       │   ├── preprocess.py       # 为 full ILP baseline 展开完整有限树。
│       │   ├── full_ilp.py         # 构建并求解完整 ILP，不使用 HILP frontier 剪枝。
│       │   ├── hilp.py             # 实现论文 Algorithm 3 的 HILP partial-ILP 启发式搜索。
│       │   ├── heuristics.py       # 提供 frontier utility/risk 启发式。
│       │   └── online_replanner.py # 封装 PROST 风格的 online replanning 循环。
│       │
│       ├── ilp/                    # ILP/p-ILP 模型表达与底层求解 backend。
│       │   ├── __init__.py         # 标记 ilp 子包并保留 backend 路线图 TODO。
│       │   ├── model.py            # 定义 solver-neutral 的变量、目标函数和线性约束。
│       │   ├── backend.py          # 定义 internal、HiGHS 和 Gurobi 共用的 backend 协议。
│       │   ├── internal.py         # 实现内置小规模 binary ILP 求解器，保证项目可独立运行。
│       │   ├── highs.py            # 封装可选 HiGHS backend。
│       │   ├── gurobi.py           # 封装可选 Gurobi backend。
│       │   └── factory.py          # 根据 CLI/config 选择 ILP backend。
│       │
│       ├── sim/                    # 本地和外部仿真交互接口。
│       │   ├── __init__.py         # 标记 sim 子包并预留仿真器注册。
│       │   ├── local.py            # 基于 PlanningProblem 的本地采样仿真器。
│       │   ├── rddlsim_client.py   # 预留 rddlsim 风格外部 TCP 仿真器客户端。
│       │   └── protocol.py         # 定义仿真器交互协议。
│       │
│       └── output/                 # 求解结果和轨迹输出工具。
│           ├── __init__.py         # 标记 output 子包并预留 benchmark 输出工具。
│           ├── json_policy.py      # 将 policy/action sequence 写成 JSON。
│           └── trace.py            # 记录 solve/evaluate 轨迹。
│
└── tests/                          # 单元测试和端到端测试。
    ├── test_basic_rddl_parser.py   # 测试基础 RDDL parser 能生成 AST 和 DOT。
    ├── test_history.py             # 测试 observation/action history 的拼接、父节点和深度。
    ├── test_belief.py              # 测试 belief prediction、observation update 和风险概率。
    ├── test_duration.py            # 测试 fixed 和 Gaussian duration 的 tau 计算。
    ├── test_internal_backend.py    # 测试内置 binary ILP backend 的小模型求解。
    ├── test_and_or_tree.py         # 测试 AND-OR tree 从根节点展开。
    ├── test_preprocess_expand.py   # 测试 full-tree preprocessing 与 Expand 集成。
    ├── test_hilp_tiny_grid.py      # 测试 HILP 与 full ILP 在 tiny grid 上一致。
    └── test_cli.py                 # 测试 CLI solve 输出可解析 JSON。
```

## 开发路线图

- [x] Phase 1：项目脚手架、CLI、测试框架、示例文件
- [x] Phase 2.1：实现基础 RDDL parser，并支持命令行解析成功提示和 AST DOT 输出
- [ ] Phase 2.2：通过 RDDLFrontend 对齐 pyrddl/pyRDDLGym frontend
- [ ] Phase 2.3：将 ParsedRDDL 编译为 PlanningProblem
- [x] Phase 3：实现核心 POMDP/(C)C-POMDP 问题模型
- [x] Phase 4：在 `and_or_tree.py` 中实现 AND-OR tree
- [x] Phase 5：实现论文中的 `Expand` 与 preprocessing
- [x] Phase 6：实现内置 ILP backend
- [x] Phase 7：实现 full ILP baseline
- [x] Phase 8：实现 HILP partial-ILP search
- [x] Phase 9：输出 offline policy JSON
- [x] Phase 10：实现 online replanning 模式
- [ ] Phase 11：接入可选 HiGHS backend
- [ ] Phase 12：接入可选 Gurobi backend
- [ ] Phase 13：实现 benchmark 与论文风格实验

## 安装与运行

开发模式安装：

```bash
virtualenv .venv
source .venv/bin/activate
python -m pip install --no-build-isolation --no-deps -e .
python -m pip install -r requirements-dev.txt
```

如果你的机器已经安装了 `python3-venv`，也可以用 `python -m venv .venv` 创建虚拟环境。本机当前是通过 `virtualenv --clear .venv` 创建 `.venv`，因为系统 Python 缺少 `ensurepip`。

仅安装运行时依赖：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

可选外部依赖仍记录在 `pyproject.toml` extras 中：

```bash
python -m pip install -e ".[rddl]"
python -m pip install -e ".[pyrddl]"
python -m pip install -e ".[highs]"
python -m pip install -e ".[gurobi]"
```

运行 tiny grid 的 HILP：

```bash
darp solve --algorithm hilp --solver internal
```

运行 full ILP baseline：

```bash
darp solve --algorithm full-ilp --solver internal
```

输出 JSON policy：

```bash
darp solve --algorithm hilp --solver internal --output policy.json
```

运行测试：

```bash
python -m pytest
```

验证基础 RDDL parser 并在终端显示 DOT AST：

```bash
python -m darp.rddl.basic_parser \
  examples/rddl/tiny_grid_domain.rddl \
  examples/rddl/tiny_grid_instance.rddl \
  --dot
```

## 当前限制与后续计划

- 当前基础 parser 只解析 RDDL 的文件、块、赋值和语句结构，用于验证 AST 与 DOT 可视化；完整 RDDL 表达式语义仍在 Phase 2 后续步骤。
- 当前默认 tiny grid 使用内置 Python 问题模型；`RDDLFrontend` 解析层已预留，完整 RDDL-to-PlanningProblem 编译仍在 Phase 2。
- DARP-RDDL 扩展语法还未定义；当前建议先用 sidecar 配置表达 duration/risk/HILP 参数。
- 内置 ILP backend 使用穷举式 binary search，只适合小规模问题和测试，不追求性能。
- HiGHS/Gurobi backend 文件已预留并提供依赖检测，完整性能复现实验放在后续阶段。
- Gaussian percentile duration 已有可运行近似实现，后续会继续补齐论文中的 smoothed belief 细节。
- 当前支持单个 expected-cost 或 chance-risk 约束；多约束接口会在 benchmark 阶段扩展。
