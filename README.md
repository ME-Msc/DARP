# DARP

DARP（Dual-space search for chance-constrained POMDPs with durative actions）是论文算法的 Python 研究实现。项目用 pyRDDLGym 解析和 grounding RDDL，用稀疏 exact kernel 计算 belief/transition/observation，并用 Gurobi 求解 full-ILP 或 HILP 的策略选择问题。

当前实现重点是：让代码结构与论文 Algorithm 1–3 对齐、只展开当前候选策略实际需要的 history 节点，以及把可达状态和数值转移延迟到首次访问时计算。

## 一键复现

在仓库根目录执行：

```bash
bash tools/install_linux_deps.sh
tools/run_repro.sh
```

第一条命令用 Python >= 3.12（本次锁定环境为 3.13）创建或更新本地 `.venv`，并按 `requirements-lock.txt` 安装固定版本依赖。第二条命令先运行全部测试，再执行论文 5×5 grid 上的 HILP/RAO*-style 快速对比与 horizon-2 exact-prefix oracle，输出到 `outputs/darp-review/`。

HILP 和 full-ILP 需要有效的本机 Gurobi license。若还需要安装 Ubuntu/Debian 系统包：

```bash
INSTALL_SYSTEM_DEPS=1 bash tools/install_linux_deps.sh
```

## 对比实验

默认代表性场景实验（不是论文 Table 2 数值复刻）：

```bash
.venv/bin/python -m experiments.scripts.run_paper_grid \
  --horizons 3 \
  --risk-budgets 0.1 0.2 0.3 \
  --repetitions 3 \
  --algorithms hilp rao-star-style \
  --expansion-rounds 100 \
  --output outputs/darp-review/paper_grid_results.csv \
  --summary-output outputs/darp-review/paper_grid_summary.csv
```

`experiments/baselines/rao_star.py` 是面向这个有限时域 benchmark 的独立、穷举式 RAO*-semantics belief-hypergraph/Pareto baseline，不是官方 RAO* 的启发式展开实现。两种算法使用相同的普通 belief utility、safe-conditioned risk 递推和 Manhattan admissible terminal tail；因此默认结果标记为 `heuristic/admissible-tail`，只能衡量相对这个 exhaustive comparator 的本机表现，不能误读为完整最优性证明或“快于官方 RAO*”。

full-ILP 会完整展开 action-observation history tree，随 horizon 指数增长，因此保留为小规模 exact-prefix oracle，而不进入默认计时矩阵。oracle 模式关闭 terminal tail，确保三个算法优化同一个有限前缀目标：

```bash
.venv/bin/python -m experiments.scripts.run_paper_grid \
  --horizons 2 \
  --risk-budgets 0.1 \
  --algorithms hilp rao-star-style \
  --include-full-ilp \
  --full-ilp-max-horizon 2 \
  --disable-terminal-tail \
  --output outputs/darp-review/paper_grid_oracle_h2.csv
```

更多实验字段和假设见 `experiments/README.md`。

## DARP 命令行

安装后可查看完整参数：

```bash
.venv/bin/darp -h
```

运行 tiny grid 的 HILP：

```bash
.venv/bin/darp \
  --domain experiments/inputs/rddl/tiny_grid_domain.rddl \
  --instance experiments/inputs/rddl/tiny_grid_instance.rddl \
  --duration experiments/inputs/durations/fixed_1.yaml \
  --planner hilp \
  --hilp-heuristic reachable-bellman \
  --heuristic-lookahead-depth 2 \
  --expansion-rounds 1 \
  --frontier-width 1 \
  --output /tmp/darp_tiny_grid_hilp.json
```

## 代码与论文的对应关系

```text
RDDL + duration sidecar
  -> adapter.GroundedRDDLView / ExactRDDLKernel
  -> planning.preprocess       # Algorithm 1：根 history 与候选动作
  -> planning.expand           # Algorithm 2：普通流与 safe 流的 belief/risk 递推
  -> planning.ilp_tree         # policy variables、flow/risk 系数
  -> planning.hilp             # Algorithm 3：solve p-ILP -> 仅展开 incumbent frontier
  -> ilp.GurobiSolver          # MIP warm start 与确定性单线程求解
```

性能相关设计：

- pyRDDLGym 仍会先 grounding 参数化 fluent/CPF；该语法层结果只构造一次并缓存。
- DARP 不预枚举完整状态空间。可达状态、非零 transition row、reward 和 observation row 在树展开触达时按需生成并缓存。
- Algorithm 2 明确分离普通概率流（utility）和 safe-conditioned 概率流（chance risk），不会因安全质量为零而错误删除仍影响收益的分支。
- fixed duration 采用常数时间进度更新；只有 stochastic duration 才执行论文的 backward smoothing。
- HILP 每轮只展开当前 ILP incumbent 中 `x_q = 1` 的 frontier，避免全局贪心展开造成的历史回归。
- full-tree 构建带显式节点上限和求解时限，避免无界占用资源。

## 项目结构

```text
DARP/
├── src/darp/            # adapter、模型、规划器与 ILP 后端
├── experiments/         # paper-grid、RAO* baseline 与自动化 runner
├── docs/                # 原论文、原始代码、符号表与实现审查报告
├── tools/               # 环境安装与一键复现脚本
└── tests/               # 单元和集成回归测试
```

## 当前边界

- exact 路径面向有限、grounded、布尔 fluent/action、单动作执行的问题；遇到并发/非布尔动作、action precondition、invariant、termination 或非 1 discount 会 fail closed，而不是静默给出错误答案。
- 当前“按需 grounding”发生在数值可达状态层，不是 lifted symbolic grounding。真正的增量符号 grounding 需要进一步做 CPF dependency slicing，并绕开 pyRDDLGym 的整体 grounder。
- online exact replanning 不能在每一步重置全局 chance budget 或 durative progress；当前对这两类多步配置显式拒绝。论文级离线策略树求解不受此限制。
- full-ILP 保留作很小 horizon 的 exact oracle；实际比较默认使用 HILP 和 RAO*-style baseline。

完整设计取舍、Git 历史审查和实验解释见 `docs/IMPLEMENTATION_REVIEW.md`。
