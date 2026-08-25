# DARP

DARP 是论文 *Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions* 的 Python 研究实现。它用 pyRDDLGym 解析 RDDL，以稀疏按需 exact kernel 传播 belief、普通概率质量和 safe-conditioned 质量，并用 Gurobi 求解 full-ILP 或 HILP。

当前仓库只保留论文 Algorithm 1–3 的核心实现，以及未来论文中可实际呈现的“DARP 与固定上游 RAO* QuadModel”共享模型实验。

## 安装

```bash
bash tools/install.sh
```

安装脚本要求 artifact 固定的 CPython 3.12.3，并从 `requirements-lock.txt` 创建 `.venv`。HILP/full-ILP 需要有效的本机 Gurobi license。

## 唯一保留的外部对照：RAO* QuadModel

实验直接载入固定上游仓库自己的 `models/quad_model.py::QuadModel`。DARP 与 RAO* 调用同一组 `actions`、`state_transitions`、`observations`、`values`、`state_risk` 和 `is_terminal` 回调；DARP 仅将上游最小化目标取负以适配自身最大化接口。双方策略随后由独立 evaluator 再次调用上游回调，重算 native cost 与 first-entry execution risk。

公开的 [`JarvisIsFriday/RAOStar`](https://github.com/JarvisIsFriday/RAOStar) 是 DARP 原论文实验节脚注引用的仓库，但源码将自身描述为 RAO* 的第三方 reimplementation，不是 AAAI 2016 作者官方 artifact。因此结果标签固定为 `jarvisisfriday-raostar-reimplementation`。该 revision 没有仓库级许可证；本项目不复制或修改上游代码，只运行用户提供的干净 checkout。

```bash
git clone https://github.com/JarvisIsFriday/RAOStar /path/to/RAOStar
git -C /path/to/RAOStar checkout --detach 543f782d80ceb9555130e911c1fcf7074153d267
mkdir -p experiments/outputs/raostar-quad

.venv/bin/python -m experiments.scripts.run_raostar_quad \
  --checkout /path/to/RAOStar \
  --python .venv/bin/python \
  --accept-no-license \
  --repetitions 25 \
  --timeout 300 \
  --output experiments/outputs/raostar-quad/runs.jsonl
```

默认矩阵为 horizon `2/3/4` × risk budget `0/0.25/0.5`。小规模结构核验可增加 `--include-full-ilp --full-ilp-max-horizon 2`。runner 会核验 commit、clean worktree、必需文件、Python/NumPy 版本和结果协议，再从固定 commit 的临时 archive 启动上游 RAO*。

也可通过一键脚本运行：

```bash
RAOSTAR_CHECKOUT=/path/to/RAOStar \
RAOSTAR_PYTHON=.venv/bin/python \
RAOSTAR_ACCEPT_NO_LICENSE=1 \
tools/run_repro.sh
```

Quad 是上游原生但完全可观测的小场景。它适合验证共享模型回调、风险语义、策略协议和运行时间，不能称为 RAO* 论文 PSR 复现或一般 CC-POMDP SOTA 证据。上游 `values` 每步包含 action cost 与 state heuristic，fixed-horizon 叶节点还会调用 heuristic；实验忠实保留该目标，不能解释成纯 path cost。上游并列节点的对象 identity 顺序会导致 fresh process 间策略变化，因此正式结果至少使用多次独立进程，并保留 completion、admissibility warnings、policy completeness 和独立评估字段。完整协议见 [实验协议](docs/EXPERIMENT_PROTOCOL.md)。

## DARP 根策略求解

```bash
.venv/bin/darp \
  --domain /path/to/domain.rddl \
  --instance /path/to/instance.rddl \
  --duration /path/to/duration.json \
  --planner hilp \
  --output /path/to/policy.json
```

## 代码与论文对应

```text
RDDL + duration sidecar
  -> adapter.GroundedRDDLView / ExactRDDLKernel
  -> planning.preprocess       # Algorithm 1：根 history 与候选动作
  -> planning.expand           # Algorithm 2：belief smoothing、普通流和 safe 流
  -> planning.ilp_tree         # policy/flow/risk 线性约束
  -> planning.hilp             # Algorithm 3：solve p-ILP -> 展开 incumbent frontier
  -> ilp.GurobiSolver          # 数值 MILP 与 warm start
  -> planning.policy           # Fraction 复核选中策略的约束和 achieved utility
```

关键语义：

- 可达状态、非零 transition/observation row 和 reward 按首次访问生成并缓存，不预枚举完整状态空间。
- utility 的普通概率流与 chance constraint 的 safe-conditioned 流严格分离。
- full-ILP 完整展开有限 history tree，只用于很小 horizon 的内部 differential oracle；HILP 仅展开当前候选策略使用的 frontier。
- fixed、expected 和 deterministic chance duration 使用可表示输入的有理数质量进行严格边界判断。
- Gaussian 使用论文的均值与 `b(s)^2 σ²` 方差公式；`erfc` 是论文 `erf` CDF 的稳定等价式，继续条件仍是严格的 `τ(q) > ζ`。这是一种数值 CDF 实现，不是任意精度误差证明。
- Gurobi 的零 MIP gap 是数值证据，不是数学最优性证明；选中策略的可行性与 achieved utility 会用 `Fraction` 独立复核。

## 目录

```text
src/darp/            核心算法、模型和 ILP 后端
experiments/         仅 Quad 共享模型对照
docs/                算法映射与实验协议
tools/               安装与复现脚本
```

论文公式与代码映射见 [算法映射](docs/ALGORITHM_MAPPING.md)。
