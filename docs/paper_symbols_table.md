# Appendix

本文保留论文主要符号，并在末尾对应当前实现。完整算法边界见 [论文—代码映射](ALGORITHM_MAPPING.md)；DARP 当前求解 CC-POMDP，下面的 expected-cost C-POMDP 公式仅作为论文符号背景。

## POMDP

| 顺序 | 符号 | 原文英文解释 | 中文解释 |
| ---: | --- | --- | --- |
| 1 | $M=\langle S,A,\mathcal{O},T,O,U,b_0,h\rangle$ | tuple defining a fixed-horizon POMDP | 固定时域 POMDP |
| 2 | $S$ | finite set of discrete states | 有限离散状态集合 |
| 3 | $A$ | finite set of actions | 有限动作集合 |
| 4 | $\mathcal{O}$ | finite set of observations | 有限观测集合 |
| 5 | $T:S\times A\times S\to[0,1]$ | probabilistic transition function between states | 状态转移概率函数 |
| 6 | $T(s,a,s')$ | $Pr(s'\mid a,s)$ | 在状态 $s$ 执行动作 $a$ 后转移到 $s'$ 的概率 |
| 7 | $O:\mathcal{O}\times S\times A\to[0,1]$ | probabilistic observation function | 观测概率函数 |
| 8 | $O(o,s,a)$ | $Pr(o\mid s,a)$ | 在状态 $s$、动作 $a$ 下得到观测 $o$ 的概率 |
| 9 | $U:S\times A\to\mathbb{R}$ | utility function | 奖赏函数 |
| 10 | $s,s'$ | states in $S$ | 状态集合中的状态 |
| 11 | $a$ | action in $A$ | 动作集合中的动作 |
| 12 | $o$ | observation in $\mathcal{O}$ | 观测集合中的观测 |
| 13 | $b_0:S\to[0,1]$ | initial belief state, a probability distribution over $S$ | 初始信念状态，即状态集合上的概率分布 |
| 14 | $h$ | planning horizon | 规划时域 / 时间范围 |
| 15 | $q=\langle(a_q^1,o_q^1),(a_q^2,o_q^2),\ldots\rangle$ | action-observation sequence, also called a history | 动作-观测序列，也称历史 |
| 16 | $i$ | execution step index | 执行步索引 |
| 17 | $a_q^i$ | action at step $i$ in history $q$ | 历史 $q$ 中第 $i$ 步的动作 |
| 18 | $o_q^i$ | observation at step $i$ in history $q$ | 历史 $q$ 中第 $i$ 步的观测 |
| 19 | $a_q$ | last action in history $q$ | 历史 $q$ 的最后一个动作 |
| 20 | $o_q$ | last observation in history $q$ | 历史 $q$ 的最后一个观测 |
| 21 | $\tilde{A}$ | set of all possible sequences that end with an action | 所有以动作结尾的历史序列集合 |
| 22 | $\tilde{\mathcal{O}}$ | set of all sequences that end with an observation, including the empty sequence | 所有以观测结尾的历史序列集合，包含空序列 |
| 23 | $\mathcal{T}(q)\triangleq\{0,1,2,\ldots\}$ | execution steps of $q$ | 历史 $q$ 的执行步集合 |
| 24 | $q=0$ | empty sequence | 空历史序列 |
| 25 | $\|q\|\triangleq\|\mathcal{T}(q)\backslash\{0\}\|$ | length of the history | 历史长度 |
| 26 | $q\le q'$ | $q$ precedes $q'$ | $q$ 是 $q'$ 的前缀 / 父历史 |
| 27 | $q-k$ | history $q$ minus the last $k$ action-observation pairs | 删除历史 $q$ 最后 $k$ 个动作-观测对 |
| 28 | $\pi(\cdot):\tilde{\mathcal{O}}\to A$ | deterministic history-dependent policy | 确定性的历史依赖策略，从以观测结尾的历史映射到一个动作 |
| 29 | $\tilde{\mathcal{O}}_{\pi}$ | policy tree nodes | 策略 $\pi$ 构成的树中的观测节点 |
| 30 | $\pi^{\star}=\arg\max_{\pi}\mathbb{E}\left[\sum\limits_{q\in\tilde{O}_{\pi}:\|q\|<h}U(S_q,\pi(q))\mid \pi\right]$ | optimal policy / conditional plan | 最优策略，在步数小于horizon的历史中使得累积奖赏期望最大的那个策略 |
| 31 | $S_q$ | random state at time $\|q\|$ obtained by following history $q$ | 沿历史 $q$ 执行后，在时间 $\|q\|$ 的随机状态 |
| 32 | $M'=M\parallel\langle P,C\rangle$ <br> $\mathbb{E}\left[\sum\limits_{q\in\tilde{O}_{\pi}:\|q\|<h}P(S_q,\pi^\star(q))\mid \pi^{\star}\right]\le C$ | constrained POMDP | 给定最优策略时，具有期望成本cost上界约束C的 C-POMDP 模型 |
| 33 | $P:S\times A\to\mathbb{R}$ | cost function | 成本函数 |
| 34 | $M''=M\parallel\langle R,\Delta\rangle$ <br> $er(q\mid\pi)\triangleq\Pr\left(\bigvee\limits_{q'\in\tilde{O}_{\pi}:q'\ge q,\|q'\|\le h} S_{q'}\in R\mid q,\pi\right)\le \Delta$ | chance-constrained POMDP | 在$\pi$策略下的q历史的执行风险er就是，<br> 在horizon范围内存在某一时刻q'进入危险状态$S_{q'}$ 的 <br> **概率** 小于风险预算 $\Delta$ 的 CC-POMDP 模型 |
| 35 | $R\subset S$ | subset that represents risky states | 风险状态集合 |

## Durative Actions

### Fixed duration

| 顺序 | 符号 | 原文英文解释 | 中文解释 |
| ---: | --- | --- | --- |
| 1 | $D(s,a)$ | duration function | 动作持续时间函数 |
| 2 | $c_a\in\mathbb{R}_+$ | execution time of action $a$ under fixed duration | 固定持续时间模型中动作 $a$ 的执行时间 |
| 3 | $L^\pi \subseteq\tilde{A}$ | set of leaf nodes of policy $\pi$ | 策略 $\pi$ 的动作叶节点集合；是否继续由 duration stopping 条件决定 |

### Stochastic duration with percentile risk criteria

| 顺序 | 符号 | 原文英文解释 | 中文解释 |
| ---: | --- | --- | --- |
| 1 | $\tau(q)$ | duration-related probability / durative constraint function | 与持续时间相关的概率或约束函数 |
| 2 | $\tau(q')\triangleq \Pr\left(\mathbb{E}\left[\sum\limits_{q\in\tilde{O}^{\pi}:q<q'}D(S_q,\pi(q))\mid q'\right]<h\right)\le \varsigma$ | Stochastic duration with percentile risk criteria | 用累计 duration 的分布评价尚未达到时间阈值 $h$ 的概率；$\tau(q')\le\varsigma$ 时停止，严格大于阈值时继续 |
| 3 | $\tau(q)=(\tau^1(q),\tau^2(q),\ldots)$ | multiple criteria encoded in formulation | 多个持续性 / 资源类约束指标 |
| 4 | $\varsigma=(\varsigma^1,\varsigma^2,\ldots)$ | multiple thresholds | 多个约束阈值 |

### Chance-constrained duration

| 顺序 | 符号 | 原文英文解释 | 中文解释 |
| ---: | --- | --- | --- |
| 1 | $D(s,a) \in \mathbb{R}_+$ | deterministic duration | 确定的动作持续时间 |
| 2 | $\tau(q')\triangleq\Pr\left(\sum_{q\in\tilde{O}:q<q'}D(S_q,\pi(q))<h\mid q'\right)\le \varsigma$ | goal of chance-constrained deterministic duration | 给定状态与动作后 duration 确定，但隐状态路径仍随机；用累计时长小于 $h$ 的条件概率判定停止，不能用平均时长替代其分布 |
| 3 | $\max\limits_{\pi}\mathbb{E}\left[\sum\limits_{q\in\tilde{O}_{\pi}:\tau(q)>\varsigma}U(S_q,\pi(q))\mid \pi\right]$ <br> $\text{subject to }\mathbb{E}\left[\sum_{q\in\tilde{O}^{\pi}:\tau(q)>\varsigma}P(S_q,\pi(q))\mid \pi\right]\le C$ | replace $\|q\|<h$ with $\tau(q)>\varsigma$ in [POMDP-line30](#POMDP), durative C-POMDP $[M', \varsigma]$ | 可持续动作的 C-POMDP 的优化目标及约束 |
| 4 | $\max\limits_{\pi}\mathbb{E}\left[\sum\limits_{q\in\tilde{O}_{\pi}:\tau(q)>\varsigma}U(S_q,\pi(q))\mid \pi\right]$ <br> $\text{subject to }\Pr\left(\bigvee\limits_{q\in\tilde{O}^{\pi}:\tau(q-1)>\varsigma}S_q\in R\mid \pi\right)\le \Delta$ | replace $\|q\|<h$ with $\tau(q)>\varsigma$ in [POMDP-line31](#POMDP), duractive CC-POMDP $[M'', \varsigma]$ | 可持续动作的 CC-POMDP 的优化目标及约束 |

## Integer Linear Programming Formulation

| 顺序 | 符号 | 原文英文解释 | 中文解释 |
| ---: | --- | --- | --- |
| 1 | $qa$ | concatenation $q\Vert\langle a\rangle$ | 在历史 $q$ 后拼接动作 $a$ |
| 2 | $qo$ | concatenation $q\Vert\langle o\rangle$ | 在历史 $q$ 后拼接观测 $o$ |
| 3 | $x\in\{0,1\}^{*}$ | binary decision vector representing a deterministic policy | 表示确定性策略的二元决策向量 |
| 4 | $x_q$ | indicates whether the last action in $q$ is selected as part of the policy | 表示历史 $q$ 的最后动作是否被策略选中 |
| 5 | $x_q=1$ | last action in $q$ is selected | 历史 $q$ 的最后动作被选中 |
| 6 | $x_q=0$ | last action in $q$ is not selected | 历史 $q$ 的最后动作未被选中 |
| 7 | $\sum\limits_{a\in A}x_a=1$, <br> $\sum\limits_{a\in A}x_{qoa}=x_q,\quad\forall q\in\tilde{A},\ \forall o\in \mathcal{O}\mid \tau(qo)>\varsigma$ | first constraint enforces one action to be selected at the root of the And-Or tree, <br> second enforces exactly one child action at observation nodes | 对任意历史而言，只要时间没用完，就应该再选一个动作 |
| 8 | $ILP[\varsigma,u_q,r_q,R] \quad$: <br> <br>$\quad \max\limits_{x_q\in\{0,1\}}\sum\limits_{q\in\tilde{A}:\tau(q-1)>\varsigma}u_qx_q,\quad$ <br><br> $\text{subject to}\quad \sum\limits_{q\in\tilde{A}:\tau(q-1)>\varsigma}r_qx_q\le R$ , <br> <br> $\quad\quad\quad\quad\quad\quad\quad \sum\limits_{a\in A}x_a=1$ , <br> <br> $\sum\limits_{a\in A}x_{qoa}=x_q,\ \forall q\in\tilde{A},\forall o\in \mathcal{O},\ \text{s.t. }\tau(qo)>\varsigma$ | integer linear program with input parameters | 以 $\varsigma,u_q,r_q,R$ 为参数的整数线性规划。<br> 在所有还需要继续决策的动作历史节点q里，选择一部分节点，使得总奖赏最大。<br> 其约束是，本选中的节点带来的总风险或总成本不能超过预算R。 <br> 第二个约束是，在策略树的根节点，必须且只能选一个初始动作。 <br> 第三个约束是，如果历史动作节点q被选中了，那么对于后续每一个可能观测o，都必须选择下一个动作a，<br> 也就是说，POMDP的策略不是一条单一路径，而是一棵条件策略树。 |
| 9 | $u_q$ | utility constant for history/action node $q$ | 历史 / 动作节点 $q$ 对应的奖赏 |
| 10 | $r_q$ | penalty or risk constant for $q$ | 历史 / 动作节点 $q$ 对应的惩罚或风险值 |
| 11 | $R$ in ILP | bound in ILP risk/cost constraint | ILP 中风险或成本约束的上界；**注意与风险状态集合 $R$ 复用同一符号** |

### Utility and Penalty for C-POMDP

Algorithm 1 从观测历史 $N=\{0\}$ 开始：取出 $q$，为可行动作构造 $qa$，调用 Algorithm 2 计算系数与观测后继 $qao$；仍满足 $\tau(qao)>\varsigma$ 的观测历史进入后续处理。代码从这些观测直接生成下一层动作 frontier，不额外存储一个完整 $N$ 集合。根 belief 是模型的 $b_0$，根 history 概率为 $1$。

| 顺序 | 符号 | 原文英文解释 | 中文解释 |
| ---: | --- | --- | --- |
| 1 | $G$ | history tree | 动作—观测历史树；代码由 `interface.root` 及其后继节点表示 |
| 2 | $N$ | observation histories awaiting expansion | 待处理的观测历史集合，不是全部尚未生成的未来后代 |
| 3 | $F$ in Algorithm 1 | action histories collected by preprocessing | 预处理收集的动作历史；勿与 Algorithm 3 的未细化 frontier 混用 |
| 4 | $u_q \triangleq \rho(q)\cdot \sum_{s\in S}\tilde{b}_{q-1}(s)U(s,a_q)$ | utility is the product of the probability of sequence q occurring, denoted by $\rho(q)$,<br> and the expected uitility of the last action $a_q$ in the history sequece $q$ | 奖赏，是历史序列q出现的概率，与最后一个动作$a_q$的期望奖赏 <br>（处于s时的后验概率，乘以，处于s且采取$a_q$动作的奖赏，的累积和）的乘积 |
| 5 | $r_q \triangleq \rho(q)\cdot \sum_{s\in S}\tilde{b}_{q-1}(s)P(s,a_q)$ | cost coefficient for C-POMDP | expected-cost 模型的加权成本；CC-POMDP 使用下文的首次失败系数 |
| 6 | $\tau(q)$ | duration continuation function | 历史 $q$ 的 duration continuation 指标，不是物理累计时长本身 |
| 7 | $\rho(q)\triangleq \Pr\left(\bigwedge\limits_{i\in\mathcal{T}(q)}o_q^i\ \middle\|\ b_0,\bigwedge\limits_{j\in\mathcal{T}(q)}a_q^j\right)$ <br><br> $=\prod\limits_{i\in\mathcal{T}(q)}\Pr\left(o_q^i \,\middle\|\, b_0,\bigwedge\limits_{\substack{j\in\mathcal{T}(q)\\ j<i}}(a_q^j,o_q^j),a_q^i\right)$ <br><br> $=\prod\limits_{i\in\mathcal{T}(q)}\Pr(o_q^i\mid \bar{b}_q^i)$ | probability of sequence $q$ occurring | $\rho(q)$ 是在初始信念 $b_0$下，按照历史 $q$ 中的动作执行并观测到对应观测序列的概率。<br> 它可以分解为每一步观测概率的乘积。|
| 8 | $\bar{b}_q^i(s)\triangleq \sum_{s'\in S} T(s',a_q^i,s)\cdot \tilde{b}_q^{i-1}(s')$ | prior belief stat after action $a_q^i$ in $q$ | $\bar{b}_q^i$ 是历史q中动作$a_q^i$之后的先验信念状态 |
| 9 | $\tilde{b}_q^i(s)\triangleq \frac{O(o_q^i,s,a_q^i)\cdot \bar{b}_q^i(s)}{\Pr(o_q^i\mid \bar{b}_q^i)},\quad \forall s\in S$ | posterior belief | 后验信念 |
| 10 | $\Pr(o_q^i\mid \bar{b}_q^i)\triangleq \sum_{s\in S}\bar{b}_q^i(s)\cdot O(o_q^i,s,a_q^i)$ | probability of observation under belief $\bar{b}_q^i$ | 在信念 $\bar{b}_q^i$ 下观测到 $o_q^i$ 的概率，所有s状态的先验信念乘以该状态下采取动作 $a_q^i$ 后获得观测 $o^q_i$ 的概率累积和（期望） |

### Utility and Risk for CC-POMDP

| 顺序 | 符号 | 原文英文解释 | 中文解释 |
| ---: | --- | --- | --- |
| 1 | $r(b) \triangleq \sum_{s\in R}{b(s)}$ | probability of being in a risky state for belief $b$ | 给定信念b的风险概率r(b)就是，所有处于风险状态R中的状态s的信念b(s)的累积加和 |
| 2 | $\bar{b}_q(s)\triangleq \frac{ \sum_{s'\in S\setminus R}T(s',a_q,s)\tilde{b}_{q-1}(s') }{1-r(\tilde{b}_{q-1}) }$ | safe prior belief in CC-POMDP recursion | CC-POMDP 风险递推中的安全先验信念。分母是上一时刻信念下不在风险状态的概率，用于归一化。分子是，在上一时刻非风险的各种状态s'下，基于其后验信念b(s')执行动作 $a_q$ 后到达状态 s 的条件概率分布 |
| 3 | $\tilde{b}_q(s)\triangleq\frac{O(o_q,s,a_q)\cdot \bar{b}_q(s)}{\eta}$ | posterior belief in CC-POMDP risk recursion | 在安全前缀条件下用当前观测更新的 posterior；它仍可包含当前风险状态，下一次预测才对来源状态作 $s'\notin R$ 筛选。$\eta$ 是归一化参数 |

### Lemma 3.3

CC-POMDP 也可以被等价地写成 ILP。它的关键是把原来复杂的“整条执行过程中进入 risky states 的概率”转化成 ILP 里的线性形式：$\sum\limits_{q}{r_q x_q} \le R $。<br>
也就是说，只要提前算好每个动作历史节点 $q$ 的风险贡献 $r_q$，那么选不选这个节点就由二元变量 $x_q$ 决定，最终总风险就是线性的。CC-POMDP 等价于 ILP，只要 ILP 的参数按下面方式设置。

| 顺序 | 符号 | 原文英文解释 | 中文解释 |
| ---: | --- | --- | --- |
| 1 | $R\triangleq \Delta-r(b_0)$ |  | ILP 中真正还能使用的，剩余风险预算 $R$，就是总风险预算 $\Delta$ (允许进入 risky states 的最大概率) 减去，<br> 初始 belief $b_0$ 本身可能已经有一部分概率在 risky states 里，这部分风险就是 $r(b_0)$ |
| 2 | $r_q\triangleq \tilde{\rho}(q)\cdot r(\bar{b}_q),\quad q\in\tilde{A}$ | first-failure risk coefficient | 安全前缀与对应 history 的联合概率 $\tilde\rho(q)$，乘以本次动作后进入风险状态的条件概率 $r(\bar b_q)$；因此仅计算第一次失败，不重复计入已经失败的路径 |
| 3 | $u_q\triangleq \rho^\star(q)\cdot \sum\limits_{s\in S}\tilde{b}_{q-1}^*(s)U(s,a_q),\quad q\in\tilde{A}$ | ordinary-flow utility coefficient | 动作历史 $q$ 对总期望效用的贡献。$\rho^\star$ 与 $\tilde b^\star$ 由普通 belief 的 Eq. (9)、(10) 给出，不是安全条件流；效用仍包括已经发生失败的路径。$\tilde\rho(q)$ 仅用于首次失败风险 $r_q$。 |

### Stochastic Duration Model

Algorithm 2 每扩展一条历史 qao，都要计算这条历史的 belief、发生概率和 duration 指标 τ(qao)。其中 stochastic duration model 用概率分布计算“总时长不足”的概率， <br>
而 chance-constrained duration model 用增广状态空间计算“累计时长不足”的概率。后续 HILP 就根据 τ(qao)>ς 决定是否继续扩展该分支，从而控制搜索树规模。

## Heuristic Forward Search

| 顺序 | 符号 | 原文英文解释 | 中文解释 |
| ---: | --- | --- | --- |
| 1 | $E$ | expanded action nodes | 已扩展动作节点集合 |
| 2 | $F$ | frontier action nodes | 尚未作为展开节点编码的动作历史；可已有缓存的 Expand 结果 |
| 3 | $h_q^u$ | admissible heuristic for utility | 奖赏的可采纳启发式上界 |
| 4 | $h_q^r$ | admissible heuristic for risk | 风险的可采纳启发式下界 |

当前核心通过 `UtilityHeuristic` 接收外部 $h(s,a)$，对于观测历史 $q$ 后的动作 $qa$，计算

$$h_{qa}^u=\sum_s\rho^*(q)\tilde b_q^*(s)h(s,a).$$

未提供回调时使用一步 utility。只有回调确实是最大化 utility 的可采纳上界时，才应设置 `upper_bound=True`。风险启发式 $h_{qa}^r$ 保持为一步首次失败概率 $r_{qa}$，是未来总风险的下界。`terminal_heuristic` 的 duration 边界规则见 [算法映射](ALGORITHM_MAPPING.md#6-algorithm-3hilp)。

## 当前代码中的存储与索引

`FrontierItem.node.history` 是动作历史 $qa$；它携带的 belief、`ordinary_mass` 与 `safe_mass` 属于动作开始前的观测历史 $q$。

| 代码 | 论文对应与含义 |
| --- | --- |
| `ordinary_mass[s]` | 普通联合质量 $\rho^*(q)\tilde b_q^*(s)$，并保留分支仍继续执行的概率权重 |
| `safe_mass[s]` | history $q$、当前状态 $s$ 与全部已访问状态（包括当前状态）均安全的联合质量 |
| `belief` / `kernel.normalize_mass` | 从普通联合质量导出的归一化 posterior，供动作可行性和 duration 计算使用 |
| `HistoryRecord.ilp_metrics` | $E$ 中的 $u_q,r_q$，或 $F$ 中的 $h_q^u,h_q^r$ |
| `HistoryRecord.policy_expansion` | 实际 observation/child 结构及 achieved utility 指标；lazy frontier 为 `None` |
| `expanded_e` / `frontier_f` | Algorithm 3 的 $E$ 动作历史 ID 集合 / $F$ 动作历史映射；树节点由 `interface.root` 所属树保存 |
| `pending_frontier` | 从待处理观测历史生成、等待下一次 ILP 编码的动作；实现不另存 $N$ |
| `ILPSolveResult.variable_values` | 策略二元向量 $\mathbf{x}$；`selected_variables` 保存值大于 0.5 的变量 |
| `risk_budget` / `remaining_risk_budget` | 总预算 $\Delta$（包含 `PolicyTreeILP.risk_budget`）/ 扣除初态风险后的 $R=\Delta-r(b_0)$ |
| `initial_risk` | 初态风险 $r(b_0)$；构造风险行时从 $\Delta$ 中扣除，提取策略时加回总风险 |
| `_encode_policy_tree_records` | 从 history 记录编码完整 ILP；HILP 用 `IncrementalPartialTreeILP` 编码每轮变化 |

这些 mass 是论文概率乘积的直接存储，不增加模型状态。例如 history 概率 $0.2$ 乘 posterior $0.6$ 得到质量 $0.12$；归一化恢复 belief，却会丢掉 history 权重，因此不能把 belief 直接代入效用或风险系数。`safe_mass` 还已剔除当前风险状态，不等于未经筛选的论文安全 posterior 与 history 概率的乘积。

HILP 每轮先求解并保存接受的结果快照，再判断是否继续 refinement。round 上限为 0 仍求解初始模型；最后一次展开后仍求解更新后的模型。超时回退使用上一次接受的树、解和计数，避免把未成功求解的新 frontier 混入已返回策略。
