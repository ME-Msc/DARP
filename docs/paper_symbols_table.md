# Appendix

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
| 3 | $L^\pi \subseteq\tilde{A}$ | set of leaf nodes of policy $\pi$ | 策略 $\pi$ 的叶节点集合，$L^\pi$ 内的点构成的策略分支耗时不超过horizon |

### Stochastic duration with percentile risk criteria

| 顺序 | 符号 | 原文英文解释 | 中文解释 |
| ---: | --- | --- | --- |
| 1 | $\tau(q)$ | duration-related probability / durative constraint function | 与持续时间相关的概率或约束函数 |
| 2 | $\tau(q')\triangleq \Pr\left(\mathbb{E}\left[\sum\limits_{q\in\tilde{O}^{\pi}:q<q'}D(S_q,\pi(q))\mid q'\right]<h\right)\le \varsigma$ | Stochastic duration with percentile risk criteria | 动作持续时间随机时，在q'历史路径下，总持续时间的期望小于horizon的概率小于等于阈值 $\varsigma$。<br> 即，约束是，没走够时间的概率不能太大。在控制“由于时长不确定，任务可能来不及完成”的风险。<br> 以车载无人机协同场景为例，无人机和车分别执行任务后，无人机要花horizon的时长与车会合， 我们在控制无人机提前到达的概率不能超过 $\varsigma$|
| 3 | $\tau(q)=(\tau^1(q),\tau^2(q),\ldots)$ | multiple criteria encoded in formulation | 多个持续性 / 资源类约束指标 |
| 4 | $\varsigma=(\varsigma^1,\varsigma^2,\ldots)$ | multiple thresholds | 多个约束阈值 |

### Chance-constrained duration

| 顺序 | 符号 | 原文英文解释 | 中文解释 |
| ---: | --- | --- | --- |
| 1 | $D(s,a) \in \mathbb{R}_+$ | deterministic duration | 确定的动作持续时间 |
| 2 | $\tau(q')\triangleq\Pr\left(\sum_{q\in\tilde{O}:q<q'}D(S_q,\pi(q))<h\mid q'\right)\le \varsigma$ | goal of Chance-constrained deteministic duration | 持续时间违反时间约束（提前到达）的概率要小于 $\varsigma$。没有期望是因为持续时间是固定的，没有概率可言 |
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

```pseudo
Algorithm 1: Preprocess[M', ς]

Input:
    C-POMDP model M' = M || <P, C> 
                     = <S, A, 𝒪 , T, O, U, b_0, h, P, C>
    Percentile threshold ς

Output:
    ILP constants (u_q, r_q) for q ∈ Ã such that τ(q - 1) > ς
    Constraint bound R
    Durative function τ(·)

Initialize:
    G ← ∅
    N ← {0}
    F ← ∅
    b̃₀ ← 0
    b₀ ← 0
    ρ̃(q) ← 0
    R ← C

do:
    q ← Pick an arbitrary element from N
    N ← N \ {q}

    for a ∈ A do:
        Obtain:
            u_qa, r_qa, b_qa,
            (b̃_qao, τ(qao), ρ(qao)) for all o ∈ O

        by calling:
            Expand[qa, b_q, (b̃_q^i) for i ∈ T(q), ρ(q), M']

        for o ∈ O do:
            if τ(qao) > ς then:
                N ← N ∪ {qao}
                F ← F ∪ {qa}
while N ≠ ∅

return:
    (u_q, r_q) for q ∈ F
    R
    τ(·)
```

| 顺序 | 符号 | 原文英文解释 | 中文解释 |
| ---: | --- | --- | --- |
| 1 | $G$ | maybe the root | 应该是根节点 |
| 2 | $N$ | set of nodes to be expanded / new nodes | 待扩展节点集合 |
| 3 | $F$ | frontier nodes / search frontier | 搜索边界节点集合 |
| 4 | $u_q \triangleq \rho(q)\cdot \sum_{s\in S}\tilde{b}_{q-1}(s)U(s,a_q)$ | utility is the product of the probability of sequence q occurring, denoted by $\rho(q)$,<br> and the expected uitility of the last action $a_q$ in the history sequece $q$ | 奖赏，是历史序列q出现的概率，与最后一个动作$a_q$的期望奖赏 <br>（处于s时的后验概率，乘以，处于s且采取$a_q$动作的奖赏，的累积和）的乘积 |
| **5?** | $r_q \triangleq \rho(q)\cdot \sum_{s\in S}\tilde{b}_{q-1}(s)P(s,a_q)$ | risk for every history $q\in \tilde{A}$ such that $\tau(q-1) > \varsigma$ |  |
| **6?** | $\tau(q)$ | duration of history $q$ | 历史q的持续时间 |
| 7 | $\rho(q)\triangleq \Pr\left(\bigwedge\limits_{i\in\mathcal{T}(q)}o_q^i\ \middle\|\ b_0,\bigwedge\limits_{j\in\mathcal{T}(q)}a_q^j\right)$ <br><br> $=\prod\limits_{i\in\mathcal{T}(q)}\Pr\left(o_q^i \,\middle\|\, b_0,\bigwedge\limits_{\substack{j\in\mathcal{T}(q)\\ j<i}}(a_q^j,o_q^j),a_q^i\right)$ <br><br> $=\prod\limits_{i\in\mathcal{T}(q)}\Pr(o_q^i\mid \bar{b}_q^i)$ | probability of sequence $q$ occurring | $\rho(q)$ 是在初始信念 $b_0$下，按照历史 $q$ 中的动作执行并观测到对应观测序列的概率。<br> 它可以分解为每一步观测概率的乘积。|
| 8 | $\bar{b}_q^i(s)\triangleq \sum_{s'\in S} T(s',a_q^i,s)\cdot \tilde{b}_q^{i-1}(s')$ | prior belief stat after action $a_q^i$ in $q$ | $\bar{b}_q^i$ 是历史q中动作$a_q^i$之后的先验信念状态 |
| 9 | $\tilde{b}_q^i(s)\triangleq \frac{O(o_q^i,s,a_q^i)\cdot \bar{b}_q^i(s)}{\Pr(o_q^i\mid \bar{b}_q^i)},\quad \forall s\in S$ | posterior belief | 后验信念 |
| 10 | $\Pr(o_q^i\mid \bar{b}_q^i)\triangleq \sum_{s\in S}\bar{b}_q^i(s)\cdot O(o_q^i,s,a_q^i)$ | probability of observation under belief $\bar{b}_q^i$ | 在信念 $\bar{b}_q^i$ 下观测到 $o_q^i$ 的概率，所有s状态的先验信念乘以该状态下采取动作 $a_q^i$ 后获得观测 $o^q_i$ 的概率累积和（期望） |

### Utility and Risk for CC-POMDP

| 顺序 | 符号 | 原文英文解释 | 中文解释 |
| ---: | --- | --- | --- |
| 1？ | $r(b) \triangleq \sum_{s\in R}{b(s)}$ | probability of being in a risky state for belief $b$ | 给定信念b的风险概率r(b)就是，所有处于风险状态R中的状态s的信念b(s)的累积加和 |
| 2 | $\bar{b}_q(s)\triangleq \frac{ \sum_{s'\in S\setminus R}T(s',a_q,s)\tilde{b}_{q-1}(s') }{1-r(\tilde{b}_{q-1}) }$ | safe prior belief in CC-POMDP recursion | CC-POMDP 风险递推中的安全先验信念。分母是上一时刻信念下不在风险状态的概率，用于归一化。分子是，在上一时刻非风险的各种状态s'下，基于其后验信念b(s')执行动作 $a_q$ 后到达状态 s 的条件概率分布 |
| 3 | $\tilde{b}_q(s)\triangleq\frac{O(o_q,s,a_q)\cdot \bar{b}_q(s)}{\eta}$ | safe posterior belief in CC-POMDP recursion | CC-POMDP 风险递推中的安全后验信念。在“前面都安全”的前提下，再结合当前观测 $o_q$，更新得到新的安全信念。$\eta$是归一化参数 |

### Lemma 3.3

CC-POMDP 也可以被等价地写成 ILP。它的关键是把原来复杂的“整条执行过程中进入 risky states 的概率”转化成 ILP 里的线性形式：$\sum\limits_{q}{r_q x_q} \le R $。<br>
也就是说，只要提前算好每个动作历史节点 $q$ 的风险贡献 $r_q$，那么选不选这个节点就由二元变量 $x_q$ 决定，最终总风险就是线性的。CC-POMDP 等价于 ILP，只要 ILP 的参数按下面方式设置。

| 顺序 | 符号 | 原文英文解释 | 中文解释 |
| ---: | --- | --- | --- |
| 1 | $R\triangleq \Delta-r(b_0)$ |  | ILP 中真正还能使用的，剩余风险预算 $R$，就是总风险预算 $\Delta$ (允许进入 risky states 的最大概率) 减去，<br> 初始 belief $b_0$ 本身可能已经有一部分概率在 risky states 里，这部分风险就是 $r(b_0)$ |
| 2 | $r_q\triangleq \tilde{\rho}(q)\cdot r(\bar{b}_q),\quad q\in\tilde{A}$ | | 动作历史节点 q 对总执行风险的贡献，等于，安全到达历史 q 的概率$\tilde{\rho}(q)$（在之前没有进入 risky states 的条件下走到 q 的概率），乘以，在节点 q 的 safe prior belief 下进入 risky states 的概率 $r(\bar{b}_q)$，就是“安全走到 q，然后在 q 这里发生风险”的概率贡献 |
| 3 | $u_q\triangleq \rho^\star(q)\cdot \sum\limits_{s\in S}\tilde{b}_{q-1}^*(s)U(s,a_q),\quad q\in\tilde{A}$ |  | 动作历史节点 q 对总期望效用的贡献，等于，历史 q 按普通 POMDP belief update 发生的概率 $\rho^*(q)$，乘以，执行 q 的最后动作之前在状态 s 上的普通后验 belief，与 在状态 s 下执行动作 $a_q$ 的效用 $U(s,a_q)$ 乘积的累加和 |

### Stochastic Duration Model

Algorithm 2 每扩展一条历史 qao，都要计算这条历史的 belief、发生概率和 duration 指标 τ(qao)。其中 stochastic duration model 用概率分布计算“总时长不足”的概率， <br>
而 chance-constrained duration model 用增广状态空间计算“累计时长不足”的概率。后续 HILP 就根据 τ(qao)>ς 决定是否继续扩展该分支，从而控制搜索树规模。

## Heuristic Forward Search

| 顺序 | 符号 | 原文英文解释 | 中文解释 |
| ---: | --- | --- | --- |
| 1 | $E$ | expanded action nodes | 已扩展动作节点集合 |
| 2 | $F$ | frontier nodes | 未展开的节点边界 |
| 3 | $h_q^u$ | admissible heuristic for utility | 奖赏的可采纳启发式上界 |
| 4 | $h_q^r$ | admissible heuristic for risk | 风险的可采纳启发式下界 |
