# DARP 论文—代码映射

本文记录复核当前实现所需的符号、公式和算法边界。原始定义与证明见 [AAAI 论文页](https://ojs.aaai.org/index.php/AAAI/article/view/26743)；输入语法见 [RDDL 扩展](../README.md#rddl-扩展)，公开 API 用法见 [README](../README.md)。下文代码路径均相对于 `src/darp/`。

## 1. 模型与历史

模型为有限时域 POMDP $M=\langle S,A,\mathcal O,T,O,U,b_0,h\rangle$。DARP 实现论文实验使用的 CC-POMDP：执行中至少进入一次风险集合的概率不超过 $\Delta$，包含初态；expected-cost C-POMDP 不属于当前求解器范围。

| 符号 | 当前实现中的含义 |
| --- | --- |
| $S,A,\mathcal O$ | 状态、动作、观测集合 |
| $T(s,a,s'),O(o,s',a)$ | 转移概率 $P(s'\mid s,a)$、观测概率 $P(o\mid s',a)$ |
| $U(s,a),b_0,h$ | 单步 utility、初始 belief、规划时间阈值 |
| $q,qa,qo,q-1$ | 动作—观测历史、追加动作／观测的历史、前一步历史；空历史为 $0$ |
| $a_q,o_q,\pi(q)$ | 历史中的最后动作、最后观测，以及确定性策略在观测历史上选择的动作 |
| $\bar b_q,\tilde b_q$ | 普通流的预测 belief 与观测后的 posterior belief |
| $\rho(q),\tilde\rho(q)$ | 普通 history 概率质量、此前一直安全地到达该 history 的联合概率质量 |
| $R\subseteq S,r(b),\Delta$ | 风险状态集合、belief 落入风险集合的概率、总风险预算 |
| $u_q,r_q,x_q$ | 动作 history 的 utility 系数、首次失败概率系数、是否被策略选中的二元变量 |
| ILP 右端 $R$ | 剩余风险预算 $\Delta-r(b_0)$；论文与风险状态集合复用同一符号 |
| $D(s,a),G_q,\tau(q),\varsigma$ | 动作时长、累计时长、duration continuation 指标、停止阈值 |
| $b_i(s),\mu_q,\sigma_q^2$ | 完整 history 下第 $i$ 步动作开始前的平滑 belief、Gaussian 累计均值与方差 |
| $E,F,h_q$ | HILP 已展开动作节点、frontier 动作节点、frontier utility heuristic 系数 |

`adapter/loader.py` 继承 pyRDDLGym grammar 解析 DARP 扩展 RDDL，`adapter/problem.py` 构建 grounded 模型，`adapter/grounded.py` 校验支持范围；`adapter/kernel.py` 只枚举从根 belief 在有限 history 内实际触达的状态、转移和观测，并以稀疏 `float` 保存概率质量。每个触达的 CPF row 必须具有可有限枚举的 support；具体状态编码由 domain 决定，核心求解器不包含 Grid 或 Manhattan 特例。

标准 RDDL 缺失的 duration/risk 在 `adapter/problem.py` 中统一补为时长 1、无危险状态，未指定预算时取 0；显式风险谓词必须有显式预算。文件求解与原生 pyRDDLGym 环境评估共用该规则，不修改原 AST，也不改变规划算法。

## 2. Algorithm 1：预处理

论文 Algorithm 1 从空历史开始，枚举 action，并对仍满足 duration continuation 条件的 observation branch 继续调用 Algorithm 2。实现分工如下：

| 论文步骤 | 实现 |
| --- | --- |
| 建立根 belief 与可行动作 | `planning/preprocess.py` |
| 展开 history-action-observation | `planning/expand.py` |
| 完整有限 history tree | `planning/ilp_tree.py` |
| full-ILP 求解 | `planning/full_ilp.py` |

可行动作由当前 belief support 上的模型回调决定。terminal belief 与“非终止但无可行动作”的 dead end 分开处理。

RDDL 的 `termination` 优先于 duration 截止：到达终止状态的最后一步仍计入完整 reward、首次失败概率和物理时长，之后不再安排动作。相同 observation 下若混有终止和未终止状态，后续 belief 和 duration smoothing 以 `done=False` 为条件，仅传递未终止的普通／安全质量，保持其原始概率权重。初态风险仍按完整 $b_0$ 计算；初态全部终止时，含根动作的求解 API 报错“无需动作策略”。

## 3. Algorithm 2：belief、效用与风险

普通 belief 流先预测再按 observation 做 Bayes 更新：

$$
\bar b_q(s')=\sum_s T(s,a_q,s')\tilde b_{q-1}(s),
$$

$$
P(o_q\mid\bar b_q)=\sum_{s'}O(o_q,s',a_q)\bar b_q(s'),
\qquad
\tilde b_q(s')=\frac{O(o_q,s',a_q)\bar b_q(s')}{P(o_q\mid\bar b_q)}.
$$

history 的普通概率质量 $\rho(q)$ 决定效用系数：

$$
u_q=\rho(q)\sum_s\tilde b_{q-1}(s)U(s,a_q).
$$

CC-POMDP 另行传播“此前一直安全”的质量。以 $\bar b_q^{\mathrm{safe}}$ 区分由安全前缀预测出的 belief，action history $q$ 的首次失败贡献为

$$
r_q=\tilde\rho(q)\,r(\bar b_q^{\mathrm{safe}}),
\qquad
r(b)=\sum_{s\in R}b(s).
$$

unsafe successor 仍保留在普通流和 utility 中，只从后续 safe flow 中移除，因此失败质量只计一次。根 belief 已有风险从总预算中扣除，ILP 使用 $R=\Delta-r(b_0)$。`planning/expand.py` 实现双流、backward message 和 smoothed belief；`planning/policy.py` 汇总选中节点的风险与 achieved utility。

## 4. Duration continuation

树只在论文严格条件

$$
\tau(q)>\varsigma
$$

成立时继续扩展。fixed / expected 模型使用剩余时间 $\tau(q)=h-\mathbb E[G_q\mid q]$；deterministic-chance 模型保留 $(s,G_q)$ 的增广后验，以 $P(G_q<h\mid q)$ 判定，不能只用平均时长替代该分布。

独立 Gaussian duration $D(s,a)\sim\mathcal N(\mu_{s,a},\sigma^2_{s,a})$ 使用论文公式

$$
\mu_q=\sum_i\sum_s b_i(s)\mu_{s,a_i},
\qquad
\sigma_q^2=\sum_i\sum_s b_i(s)^2\sigma_{s,a_i}^2,
$$

$$
\tau(q)=P(G_q<h).
$$

代码计算

$$
\tfrac12\operatorname{erfc}\!\left(\frac{\mu_q-h}{\sqrt{2\sigma_q^2}}\right),
$$

正方差时，它与论文的 $\tfrac12[1+\operatorname{erf}((h-\mu_q)/(\sigma_q\sqrt2))]$ 代数等价，并减少尾部消减误差。方差按 $b_i(s)^2$ 加权，不能替换成 belief mixture 方差。所有 duration 数值与概率流使用 binary64；判定严格使用 `>`，包括零方差、对称点及 $\varsigma=0$ 的解析分支。

`model/duration.py` 实现 stopping test，`planning/expand.py` 用完整状态的 smoothed belief 累积 expected / Gaussian duration，或传播 deterministic-chance 增广后验。instance 的整数 horizon 仅是时间阈值，不额外截断 action depth；未触发 RDDL `termination` 时，由上述 stopping test 决定树的边界。

## 5. ILP

每个 action history 对应二元变量 $x_q$。确定性、observation-closed 的条件策略满足

$$
\sum_{a\in A}x_a=1,
\qquad
\sum_{a\in A}x_{qoa}=x_q
$$

（第二式仅针对需要继续决策的可达 observation branch）。优化问题为

$$
\max_x\sum_q u_qx_q,
\qquad
\text{s.t. }\sum_q r_qx_q\le R.
$$

`planning/ilp_tree.py` 生成 root、flow、observation-closure 和风险行；`ilp/gurobi.py` 求解 binary64 系数的二元模型，使用 `MIPGap=1e-6`、默认 `FeasibilityTol=1e-6` 和默认线程设置。`OPTIMAL` 表示在数值容差内完成搜索；策略风险采用同一可行性容差。full-ILP 枚举完整有限树，用作很小 horizon 的结构 oracle。

## 6. Algorithm 3：HILP

`planning/hilp.py` 重复执行：

1. 构造并求解当前 partial ILP；
2. 从 incumbent 读取被策略选择的 frontier；
3. 只展开这些 frontier；
4. warm-start 下一轮，直到没有可展开的选中 frontier 或达到显式资源上限。

一次 HILP 搜索在同一个 Gurobi model 上增量维护 p-ILP：保留已有 root、变量和 flow 行；frontier $q$ 展开时把目标系数从 $h_q$ 更新为 $u_q$，加入 child variables、flow 行并扩展同一条全局风险行。上一轮 incumbent 作为下一轮 MIP start，不固定策略前缀。启用外部 heuristic 与 `terminal_heuristic` 时，frontier 可先计算 $h_q$ 和一步风险，待 incumbent 选中后才生成观测、duration 与后继；这些增量和延迟计算保持同一数学模型。

领域启发式通过 [`UtilityHeuristic`](../README.md#自定义-heuristic) 外部注入。核心负责论文规定的 history 概率加权，其中 $b_q$ 是动作开始前的普通 belief：

$$
h_q=\sum_s \rho(q)b_q(s)h(s,a_q).
$$

回调返回单个状态的 utility-to-go，cost-to-go 应取负；未提供回调时使用一步 utility。只有 heuristic 确实是最大化目标的可采纳上界时，才可设置 `upper_bound=True`，用于最优性认证。frontier 风险系数保持为一步首次失败概率 $r_q$，即后续总风险的下界。`frontier_width=None` 展开 incumbent 中全部 frontier；有限宽度仅限制每批展开数量。

`terminal_heuristic` 按 observation branch 评价 duration 边界：用该分支概率加权的动作开始前状态 heuristic 替换最后一步 utility；继续分支和模型终止分支保留真实 RDDL utility。同一 action 的停止／继续分支可以共存。

`complete` 要求 Gurobi 在容差内返回 `OPTIMAL`、当前策略的 frontier 已完成 refinement、策略 duration-complete 且风险可行、没有 solver time 截断，并且全局已无可展开 frontier 或外部 heuristic 提供可采纳上界。因 expansion round 上限留下未完成 refinement 的结果保持 incomplete。一个策略可以已有 achieved utility，但仍因未选分支缺少上界而无法认证搜索完成。

## 7. 规划、执行与实验边界

`executor.PolicyExecutor` 按观测执行已保存的 `ConditionalPolicy`，真实状态用于统计首次失败与动作物理时长。执行得到的原始 RDDL discounted reward、风险频率、物理时长和墙钟时间，与求解器 objective、平滑 belief 的 duration stopping 量分开记录；执行器不重复应用 terminal heuristic。

保存格式与执行统计见 [策略保存与回放](../README.md#策略保存与回放)，实验输入、计时和与论文的差异见 [实验协议](EXPERIMENT_PROTOCOL.md)。核心只依赖通用模型与外部 heuristic；领域编码和论文表格流程属于 `experiments/`。
