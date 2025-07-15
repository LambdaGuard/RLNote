# 强化学习笔记

## Policy Gradient
RL的目标是最大化整条轨迹的期望回报，即最大化
$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]
$$
其中$\tau=(s_0, a_0, s_1, a_1, \ldots, s_T, a_T)$表示从初始状态$s_0$开始，按照策略$\pi_\theta$采取动作$a_t$，直到终止状态$s_T$的轨迹，$R(\tau)$表示整条轨迹的回报。又
$$
J(\theta) = \int_{\tau} R(\tau) p_\theta(\tau) \text{d}\tau
$$
其中
$$
p_\theta(\tau) = p(s_0) \prod_{t=0}^{\infty} p(a_t | s_t, \theta) p(s_{t+1} | s_t, a_t)
$$
利用log-derivative trick计算$J(\theta)$的梯度
$$
\begin{aligned}
\nabla_\theta J(\theta) = &\int_{\tau} R(\tau) \nabla_\theta p_\theta(\tau) \text{d}\tau\\
= &\int_{\tau} R(\tau) p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} \text{d}\tau\\
= &\int_{\tau} R(\tau) p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) \text{d}\tau\\
= & \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau) \nabla_\theta \log p_\theta(\tau)]
\end{aligned}
$$
其中
$$
\log p_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{\infty} \log p(a_t | s_t, \theta) + \sum_{t=0}^{\infty} \log p(s_{t+1} | s_t, a_t)
$$
因此
$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t | s_t)
$$
即
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t | s_t) \right]
$$
对于不同的策略梯度算法，可以对$R(\tau)$进行不同的处理：
1. $R(\tau)=\sum_{t=0}^{\infty} r_t$. 全轨迹回报，GDOMDP；
2. $R(\tau)=\sum_{t'=t}^{\infty} G_{t'}$. $a_t$动作之后的回报，REINFORCE，降低方差；
3. $R(\tau)=\sum_{t'=t}^{\infty} G_{t'}-b(s_t)$. $R(\tau)$减去基线，REINFORCE with baseline，这里的baseline是状态值函数$V(s_t)$，可以用$V(s_t)$的估计值$b(s_t)=\mathbb{E}[G_t|s_t]$来代替。
4. $Q^\pi(s_t, a_t)$. $R(\tau)$为动作$a_t$的动作价值函数，Q-Actor-Critic方法。
5. $A^\pi(s_t, a_t)$. $R(\tau)$为优势函数，Advantage Actor-Critic方法。
6. $r_t+V^\pi(s_{t+1})-V^\pi(s_t)$. $R(\tau)$为TD误差，Actor-Critic方法。

### RLOO
REINFORCE Leave One Out (RLOO)方法是对REINFORCE的改进，旨在降低方差。区别在于，RLOO采用所有其他样本的平均回报作为基线，而非仅含有当前样本的回报。具体来说，RLOO维护一个轨迹batch，并引入离群基线：
$$
b(s_t) = \frac{1}{N-1} \sum_{i \neq t} G_i
$$
该基线不含当前轨迹的信息，因此依然保持无偏性。由于能够利用其他样本的信息，RLOO的基线更为精准，方差通常低于REINFORCE。

### Proximal Policy Optimization (PPO)
PPO在原生PG的基础上引入了一个剪切的surrogate objective来控制每次更新的幅度，将原先直接加权的$A_t$换成了带剪切的比率形式：
$$
J(\theta) = \mathbb{E}_{t} \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$
PPO通过裁剪方式限制了策略更新的幅度，避免了过大的更新导致的策略崩溃。这里的$r_t(\theta)$是当前策略与旧策略的比率，即
$$
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$
PPO在保持PG算法梯度灵活性的同时，通过限制更新幅度来提高训练的稳定性和收敛速度。

### Group Relative Policy Optimization (GRPO)
GRPO（Group Relative Policy Optimization）是一种针对大模型 RLHF 的无 Critic、基于 REINFORCE 的 PPO 变体。它的核心在于用“组内比较”来估计优势，从而省略价值网络、节省显存，并保持较低方差。

对于每一条prompt，模型会按照当前策略生成$N$条回答，并分别计算它们的reward$\{r_1,...,r_N\}$。并计算组内平均$\mu=\frac1N \sum_i r_i$和标准差$\sigma=\sqrt{\frac1N \sum_i (r_i-\mu)^2}$。然后定义优势：
$$
A_i \;= \frac{r_i - \mu}{\sigma + \epsilon}
$$
GRPO不训练Criti网络，只使用优势函数$A_i$来更新策略：
$$
\nabla_\theta J(\theta) \approx \mathbb{E} \left[ A_i \nabla_\theta \log \pi_\theta(a_i | s_i) \right]
$$
这种方法省略了价值网络，大大降低了大模型微调时的现存需求，同时通过组内比较来估计优势，保持了较低的方差。另外，由于无需估计长期价值，GRPO在处理长文本生成任务时更为高效，特别适合CoT等长文本生成任务。

然而，由于对每个prompt都需要生成多个回答并计算组内统计，GRPO具有较高的推理成本。同时，对每一个prompt单独计算reward baseline会带来以下两方面的过拟合风险：
1. Reward Hacking（奖励劫持）

    每个 prompt 独立设 baseline 时，模型在该 prompt 下生成的多条候选答案，其优势被紧密绑定到这个 prompt 的 reward 分布上。优化时，模型往往会寻找 reward 模型的“漏洞”——即并非真正提升质量，而是利用奖励模型的偏差或边缘情况来获得更高分数，典型表现为 repetitive phrases、无关填充或对抗式用词，以“劫持”奖励信号。

2. Prompt-level的过拟合

    在一个 batch 内，每个 prompt 只参与本组响应的 baseline 计算，导致模型对训练集中相对简单或频繁出现的 prompt 学习过度：
    - 高度集中：模型专注于提升对这些 prompt 的表现，忽视跨 prompt 的泛化能力；
    - 输出多样性下降：为了最大化相对优势，响应趋向千篇一律，减少了 token‐level 上的多样性；
    - OOD 性能恶化：在新的或分布外的 prompt 上往往表现不佳，因为学习到的是针对训练 prompt 的“投机性”策略。
  
GRPO虽然能在短期内提高该 prompt 上的 reward，但却加剧了对奖励模型的依赖与“投机性”优化，显著损害模型的多样性和泛化能力。

为了阻止reward hacking，GRPO采用了一个PPO风格的办法：约束新策略不要偏离已有的策略太远，采用KL散度来量化。一个典型的GRPO对齐目标会写成：
$$
\max_\theta \mathbb{E}_{i\in\text{Group}, a_i\sim \pi_theta} \left[ \min(r_iA_i, \text{clip}(r_i, 1-\epsilon, 1+\epsilon)A_i) \right] - \beta \mathbb{E}_s\left[D_{KL}(\pi_\theta(\cdot|s)||\pi_{ref}(\cdot|s))\right]
$$



### 一些思考题
1. 为什么策略梯度要使用对数而非直接对$\pi_\theta(a_t | s_t)$求导？
   - 直接对$\pi_\theta(a_t | s_t)$求导，会得到$\nabla_\theta\pi$，难以通过采样估计。
   - 使用对数的好处是可以将$\pi_\theta(a_t | s_t)$的乘积转化为和，便于计算。同时也能分离出概率$\pi_\theta(a_t | s_t)$，使期望可以对应于蒙特卡罗采样。
2. 为什么REINFORCE方差很大？试讨论不同的基线（常数基线、状态基线、动作—状态基线）如何从数学上降低方差？当基线估计不准确时，会对训练产生什么影响？
   - REINFORCE的方差大是因为每个动作的回报都依赖于后续所有动作的回报，导致回报的波动很大。而梯度估计又是多个时间步的回报的加权平均，因此方差会被放大。同时，随着episode变长，环境和策略的随机性被成倍放大，即使存在折扣因子，也只能部分抑制远期奖励的影响。
   - 常数基线不会降低方差，但可以提高学习的稳定性，因为它不会改变梯度的方向，只是平移中心化了梯度。实现没有难度，可以保持无偏性。
   - 状态基线（$V(s_t)$）可以降低方差，因为它是对每个状态的平均回报的估计，可以减少回报的波动。实现需要训练一个状态值函数来估计每个状态的平均回报，可以保持无偏性(即AC)。
   - 动作—状态基线（$Q(s_t, a_t)$）进一步降低方差，因为它考虑了特定动作在特定状态下的价值，能够更精确地反映动作的实际贡献。但是实现较复杂，需要训练一个动作价值函数来估计每个状态-动作对的价值，并且破坏了无偏性。如果要这样用，则应该采用Advanatge Actor-Critic方法，即学习优势函数$A^\pi(s, a)$，其中$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$，$V^\pi(s)$可以保证无偏。
   - 关注基线部分的期望：
$$
\begin{aligned}
\mathbb{E}_{a\sim \pi}[\nabla_\theta \log \pi_\theta(a | s) b(s,a)] & = \int_{\mathcal{A}} \nabla_\theta \log \pi_\theta(a | s) b(s,a) \pi_\theta(a | s) \text{d}a\\
& = \int_{\mathcal{A}} \nabla_\theta \pi_\theta(a | s) b(s,a) \text{d}a\\
& = \nabla_\theta \int_{\mathcal{A}} \pi_\theta(a | s) b(s,a) \text{d}a\\
& = \nabla_\theta \mathbb{E}_{a\sim \pi} [b(s,a)]
\end{aligned}
$$
   - 当基线$b$是常数或仅依赖于状态，$\nabla_\theta \mathbb{E}_{a\sim \pi} [b(s,a)]=0$，此时即是无偏。当基线$b$的计算依赖于动作，此时基线关于参数$\theta$的梯度不为0，无偏性被破坏。
   - 当基线估计不准确时，可能会导致梯度估计偏离真实值，从而影响策略更新的方向和幅度，可能导致收敛速度变慢或陷入局部最优。

3. 为什么说 Actor–Critic 在样本利用率上优于蒙特卡洛方法？请结合 TD 残差与 bootstrapping 的概念说明两者在数据使用和更新频率上的差别。
   - REINFORCE方法属于蒙特卡罗采样，每次更新都需要等待整个episode结束才能计算回报。更新只能在每个episode结束后一次性完成，轨迹数据用完即丢弃。一条轨迹仅贡献一次梯度更新。
   - Actor-Critic在与环境交互的每一步都能计算一次TD残差，并立即用它更新策略和价值函数。每一步都能利用当前状态和动作的反馈进行更新，样本利用率更高。一条轨迹能贡献$T$次梯度更新。AC通过bootstrapping利用当前状态的价值估计来更新策略和价值函数，而不是等待整个轨迹结束。这样可以更快地适应环境变化，提高学习效率。