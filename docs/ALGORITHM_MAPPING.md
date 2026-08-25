# DARP 论文—代码映射

本文只记录复核实现所需的公式、算法步骤和证据边界。原始定义与证明以 [AAAI 论文页](https://ojs.aaai.org/index.php/AAAI/article/view/26743) 为准。

## 1. 模型与历史

论文使用有限时域 POMDP

$$
M=\langle S,A,\mathcal O,T,O,U,b_0,h\rangle,
$$

其中 $T(s,a,s')=P(s'\mid s,a)$，$O(o,s',a)=P(o\mid s',a)$。历史
$q=\langle(a^1,o^1),\ldots,(a^k,o^k)\rangle$，确定性条件策略把每个可达的 observation history 映射到一个 action。

约束有两类：

- C-POMDP：期望累计 cost 不超过 $C$；
- CC-POMDP：执行中首次进入风险集合的概率不超过 $\Delta$。

RDDL 经 `adapter/grounded.py` 解析成有限模型回调；`adapter/exact.py` 只枚举从根 belief 实际可达的状态、转移和观测，并以 `Fraction` 保存概率质量。

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

因此 unsafe successor 仍保留在普通流和 utility 中，只从后续 safe flow 中移除；它的首次失败质量只计一次。根 belief 已有风险从总预算中扣除。`planning/expand.py` 实现上述双流、backward message 和 smoothed belief；`planning/policy.py` 对选中策略重新传播精确质量，独立核对 constraint 与 achieved utility。

期望 cost 约束使用普通概率流：

$$
r_q=\rho(q)\sum_s\tilde b_{q-1}(s)P(s,a_q).
$$

state/action-conditioned failure 与 expected-cost callback 都必须声明精确语义；未知约束字段和无法验证的 exactness 会 fail closed。

## 4. Duration continuation

树只在论文严格条件

$$
\tau(q)>\varsigma
$$

成立时继续扩展。fixed、expected 和 deterministic-chance duration 对可表示输入使用有理数累计。

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

它与论文的 $\tfrac12[1+\operatorname{erf}((h-\mu_q)/(\sigma_q\sqrt2))]$ 代数等价，并减少尾部消减误差。判定仍是严格 `>`，没有固定 ULP 上移或 `>=`。这是稳定的 binary64 CDF 实现，不是有向舍入的数学区间证书。

RDDL 的整数 horizon 不会隐式截断 duration tree；只有论文的 duration stopping test 决定 action depth。HILP 若因 expansion round 或 solver time 上限停止，结果保持 incomplete。

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

`planning/ilp_tree.py` 生成 root、flow、observation-closure 和风险/成本行；`ilp/gurobi.py` 求解二元模型并复核 incumbent 的精确系数。Gurobi 的 `MIPGap=0` 与 `MIPGapAbs=0` 只表示浮点模型的 numerical zero gap。`Fraction` 复核证明选中策略在已编码模型中的精确可行性和实际效用，不证明全局数学最优。

## 6. Algorithm 3：HILP

`planning/hilp.py` 重复执行：

1. 构造并求解当前 partial ILP；
2. 从 incumbent 读取被策略选择的 frontier；
3. 只展开这些 frontier；
4. warm-start 下一轮，直到没有可展开的选中 frontier 或达到显式资源上限。

默认 `reachable-bellman` 在精确可达 kernel 上提供 optimistic utility tail；`one-step-greedy` 只是近似排序分数。只要仍有未展开的候选 subtree、启发式可采纳性无法验证或资源上限截断，结果就不能标记 complete。full-ILP 枚举完整有限树，仅作为很小 horizon 的结构 oracle。
