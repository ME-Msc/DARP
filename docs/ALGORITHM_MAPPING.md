# DARP 论文—代码映射

本文只记录复核实现所需的公式、算法步骤和证据边界。原始定义与证明以 [AAAI 论文页](https://ojs.aaai.org/index.php/AAAI/article/view/26743) 为准。

## 1. 模型与历史

论文使用有限时域 POMDP

$$
M=\langle S,A,\mathcal O,T,O,U,b_0,h\rangle,
$$

其中 $T(s,a,s')=P(s'\mid s,a)$，$O(o,s',a)=P(o\mid s',a)$。历史 $q=\langle(a^1,o^1),\ldots,(a^k,o^k)\rangle$，确定性条件策略把每个可达的 observation history 映射到一个 action。

约束有两类：

- C-POMDP：期望累计 cost 不超过 $C$；
- CC-POMDP：执行中首次进入风险集合的概率不超过 $\Delta$。

RDDL 经 `adapter/grounded.py` 解析成模型回调；`adapter/kernel.py` 只枚举从根 belief 在有限 history 内实际触达的状态、转移和观测，并以稀疏 `float` 保存概率质量。每个触达的 CPF row 必须具有可有限枚举的 support；具体状态编码由 domain 决定，核心求解器不包含 Grid 或 Manhattan 特例。

## 2. Algorithm 1：预处理

论文 Algorithm 1 从空历史开始，枚举 action，并对仍满足 duration continuation 条件的 observation branch 继续调用 Algorithm 2。实现分工如下：

| 论文步骤 | 实现 |
| --- | --- |
| 建立根 belief 与可行动作 | `planning/preprocess.py` |
| 展开 history-action-observation | `planning/expand.py` |
| 完整有限 history tree | `planning/ilp_tree.py` |
| full-ILP 求解 | `planning/full_ilp.py` |

可行动作由当前 belief support 上的模型回调决定。terminal belief 与“非终止但无可行动作”的 dead end 分开处理。

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

CC-POMDP 另行传播“此前一直安全”的质量。对 action history $q$，首次失败贡献为

$$
r_q=\tilde\rho(q)\,r(\bar b_q),
\qquad
r(b)=\sum_{s\in R}b(s).
$$

因此 unsafe successor 仍保留在普通流和 utility 中，只从后续 safe flow 中移除；它的首次失败质量只计一次。根 belief 已有风险从总预算中扣除。`planning/expand.py` 实现上述双流、backward message 和 smoothed belief；`planning/policy.py` 汇总选中节点的浮点 constraint 与 achieved utility。

期望 cost 约束使用普通概率流：

$$
r_q=\rho(q)\sum_s\tilde b_{q-1}(s)P(s,a_q).
$$

state/action-conditioned failure 与 expected-cost callback 必须显式声明约束语义；未知约束字段或超出合法范围的数值会直接报错。

## 4. Duration continuation

树只在论文严格条件

$$
\tau(q)>\varsigma
$$

成立时继续扩展。fixed、expected 和 deterministic-chance duration 使用 binary64 浮点累计。

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

它与论文的 $\tfrac12[1+\operatorname{erf}((h-\mu_q)/(\sigma_q\sqrt2))]$ 代数等价，并减少尾部消减误差。判定仍是严格 `>`，没有固定 ULP 上移或 `>=`；所有 duration 数值与概率流一样使用 binary64。

Duration 参数必须从 RDDL 之外的 JSON sidecar 读入，求解器不会隐式假设单位时长。RDDL 的整数 horizon 只作为 sidecar evaluator 的时间阈值，不会额外截断 duration tree；只有论文的 duration stopping test 决定 action depth。HILP 若因 expansion round 或 solver time 上限停止，结果保持 incomplete。

## 5. ILP

每个 action history对应二元变量 $x_q$。确定性、observation-closed 的条件策略满足

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

`planning/ilp_tree.py` 生成 root、flow、observation-closure 和风险/成本行；`ilp/gurobi.py` 直接求解 binary64 系数的二元模型。与论文参考代码相同，Gurobi 使用 `MIPGap=1e-4` 和默认线程设置；返回 `OPTIMAL` 表示在该数值容差内完成搜索。实现不再进行 zero-gap 求解、有理数 incumbent 复核或 no-good 重求解，策略风险由浮点传播以统一容差判断。

## 6. Algorithm 3：HILP

`planning/hilp.py` 重复执行：

1. 构造并求解当前 partial ILP；
2. 从 incumbent 读取被策略选择的 frontier；
3. 只展开这些 frontier；
4. warm-start 下一轮，直到没有可展开的选中 frontier 或达到显式资源上限。

一次 HILP 搜索在同一个 Gurobi model 上增量维护这些 p-ILP：已有 root、变量和 flow 行保持不变；frontier $q$ 展开时把目标系数从 $h_q$ 更新为 $u_q$，再加入 child variables、flow 行并扩展同一条全局风险行。上一轮 incumbent 作为下一轮 MIP start；该增量更新不固定策略前缀，也不改变论文的数学问题。

领域启发式通过 `UtilityHeuristic` 外部注入。回调只计算单个状态的 utility-to-go，核心负责论文规定的 history 概率加权：

$$
h_q=\sum_s \rho(q)b_q(s)h(s,a_q).
$$

cost-to-go 回调必须返回负值，因为 DARP 最大化 utility。未提供回调时只使用一步 utility 作为 fallback；核心不再内置 reachable-Bellman 或 Manhattan。`frontier_width=None` 展开 incumbent 中全部 frontier，与论文复现实验一致；有限宽度仅是显式的 batching 选项。`terminal_heuristic` 单独控制 duration 边界的评价，避免把实验的 terminal value 混进 RDDL reward。当前 action-level terminal value 要求同一 action 的所有 observation branch 同时停止；混合停止/继续会 fail fast，callback 也必须为模型 terminal state 返回正确终值。

只有算法不再包含被当前策略选中的可展开 frontier、Gurobi 返回容差内 `OPTIMAL`、浮点风险满足预算且没有触发资源上限时，结果才标记 `complete`。full-ILP 枚举完整有限树，仅作为很小 horizon 的结构 oracle。
