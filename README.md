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

---

### RLOO
REINFORCE Leave One Out (RLOO)方法是对REINFORCE的改进，旨在降低方差。区别在于，RLOO采用所有其他样本的平均回报作为基线，而非仅含有当前样本的回报。具体来说，RLOO维护一个轨迹batch，并引入离群基线：
$$
b(s_t) = \frac{1}{N-1} \sum_{i \neq t} G_i
$$
该基线不含当前轨迹的信息，因此依然保持无偏性。由于能够利用其他样本的信息，RLOO的基线更为精准，方差通常低于REINFORCE。

---

### Proximal Policy Optimization (PPO)
PPO在原生PG的基础上引入了一个剪切的surrogate objective来控制每次更新的幅度，将原先直接加权的$A_t$换成了带剪切的比率形式：
$$
J(\theta) = \mathbb{E}_{t} \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$
PPO通过裁剪方式限制了策略更新的幅度，避免了过大的更新导致的策略崩溃。这里的$r_t(\theta)$是当前策略与旧策略的比率，即
$$
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$
PPO在保持PG算法梯度灵活性的同时，通过限制更新幅度来提高训练的稳定性和收敛速度。

---

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

### 异步性问题
大语言模型在RLHF的时候，生成（rollout）和训练（update）的协同是一个关键问题。
#### 同步On-Policy
在一张GPU上，先用当前策略生成样本，然后马上用这些样本来更新策略。这样做的好处是简单直接，容易实现，但缺点是样本利用率低，因为每次更新都需要等待新的样本生成。
#### 分离式On-Policy
将生成放在GPU1，训练放在GPU2。GPU1生成样本，GPU2使用这些样本来更新策略。GPU1每生成完一批样本，就将它们传给GPU2进行更新。这样可以提高样本利用率，但仍然存在延迟问题，因为GPU2需要等待GPU1生成完样本才能进行更新。
#### 高效Off-Policy
在GPU1上生成样本，GPU2上训练策略。GPU1生成样本后，将它们存储在一个缓冲区中，GPU2可以从这个缓冲区中随机抽取样本进行更新。这样可以提高样本利用率，并且GPU2可以在等待新样本的同时进行训练。但是，这种方法需要处理样本的相关性问题，因为生成的样本可能与当前策略不一致，需要额外关注收敛稳定性和学习信号的一致性。
#### 实践中的常用做法
用像 Ray RLlib 或自研的分布式进程管理框架，将推理服务（VLLM、TensorRT）和训练服务拆到不同节点。生成端不断往数据队列里塞样本，训练端不断消费并更新模型。关键挑战在于：
- 异步带来的延迟偏差（policy lag）会影响策略梯度的准确性；
- 要设计合适的重放缓冲区策略（如优先级、最新策略裁剪），以兼顾样本多样性和新旧策略偏差；
- 同时要用KL 约束、学习率调整等手段，保持训练稳定。

---

### Generalized Advantage Estimation (GAE)
广义优势估计 (GAE) 是计算策略梯度算法优势的另一种方法，它能够更好地平衡偏差-方差权衡。传统的单步优势估计可能会引入过多的偏差，而使用完整轨迹往往会产生较高的方差。GAE 的工作原理是结合两种思路——多步预测和加权移动平均（或两者兼而有之）。

定义n步长优势估计：
$$
\hat{A}_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l r_{t+l} + \gamma^n V(s_{t+n}) - V(s_t)
$$
较小的n会导致低方差，但偏差较大。定义时序差分残差
$$
\delta_t^V= r_t + \gamma V(s_{t+1}) - V(s_t)
$$
GAE通过引入一个衰减因子$\lambda$来在所有不同的n步长优势估计之间进行平滑过渡：
$$
\begin{aligned}
\hat{A}_t^{\text{GAE}} &= (1-\lambda) \sum_{l=1}^{\infty} \lambda^{l-1} \hat{A}_t^{(l)}\\
& = (1-\lambda) \sum_{l=1}^{\infty} \lambda^{l-1} \left( \sum_{k=0}^{l-1} \gamma^k r_{t+k} + \gamma^l V(s_{t+l}) - V(s_t) \right)\\
& = (1-\lambda) \sum_{l=1}^{\infty} \lambda^{l-1} \left( \sum_{k=0}^{l-1} \gamma^k \delta_{t+k}^V \right)\\
& = (1-\lambda) \sum_{k=0}^\infty \gamma^k\delta_{t+k}\sum_{l=k+1}^{\infty} \lambda^{l-k-1}\\
& = (1-\lambda) \sum_{k=0}^\infty \frac{\lambda^k}{1-\lambda}\gamma^k\delta_{t+k}^V \\
& = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V
\end{aligned}
$$

---

### Trust Region Policy Optimization (TRPO)
Trust Region Policy Optimization (TRPO) 是 Schulman 等人于 2015 年提出的一种稳定且高效的策略梯度方法，通过在“信赖域”（trust region）内限制每次策略更新的幅度，来避免策略发生“跳跃式”退化。
TRPO 的核心思想是通过约束策略更新的 KL 散度来限制每次更新的幅度。具体来说，TRPO 通过以下优化目标来更新策略：
$$
\begin{aligned}
&\max_\theta\quad 
\mathbb{E}_{s,a\sim\pi_{\theta_{\rm old}}}
\Bigl[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\rm old}}(a\mid s)}\,A^{\pi_{\rm old}}(s,a)\Bigr] \\[6pt]
&\text{subject to}\quad 
\mathbb{E}_{s\sim d^{\pi_{\rm old}}}\bigl[\,D_{\rm KL}\bigl(\pi_{\theta_{\rm old}}(\cdot\mid s)\,\|\,\pi_\theta(\cdot\mid s)\bigr)\bigr]
\;\le\;\delta,
\end{aligned}
$$
其优点在于每次更新都有严格的KL散度保证，性能稳定。理论上接近自然梯度，收敛快而样本效率高。与PPO相比，TRPO直接优化二次子问题，约束更精确，在复杂环境下能取得更优秀的性能。 但是其实现较为复杂，计算开销大，尤其在大模型上需要更多的计算资源和内存。

---

### Deterministic Policy Gradient (DPG)
Deterministic Policy Gradient (DPG) 是 Silver et al. 在 2014 年提出的一种基于策略梯度的强化学习方法，专门用于连续动作空间，核心思想是将随机策略换成“确定性”策略，从而大幅降低策略梯度方差，并且能够天然地做离线（off‐policy）学习。
DPG 的核心思想是将策略函数 $\pi_\theta(s)$ 直接映射到动作空间，而不是通过概率分布采样动作。DPG的Actor网络直接输出一个确定性动作$a=\mu_\theta(s)$，目标函数为
$$
J(\theta) = \mathbb{E}_{s\sim \rho^\beta} \left[ Q_{\phi}(s, \mu_\theta(s)) \right]
$$
其中$\rho^\beta$是行为策略的状态分布，$Q_{\phi}(s, a)$是动作价值函数。DPG的梯度更新为
$$
\nabla_\theta J(\theta) = \mathbb{E}_{s\sim \rho^\beta} \left[ \nabla_a Q_{\phi}(s, a) \bigg|_{a=\mu_\theta(s)} \nabla_\theta \mu_\theta(s) \right]
$$
其中$\nabla_a Q_{\phi}(s, a)$由Critic网络得到，$\nabla_\theta \mu_\theta(s)$是Actor网络的Jacobian。
DDPG（Deep Deterministic Policy Gradient）是 DPG 的深度学习扩展，结合了深度神经网络和经验回放（experience replay）机制。DDPG使用Actor-Critic架构，其中Actor网络输出确定性动作，Critic网络估计动作价值函数。DDPG的更新步骤包括：
1. 从经验回放缓冲区中采样一批$(s, a, r, s')$样本；
2. Target网络：维护慢更新的目标网络$Q_{\phi'}$和$\mu_{\theta'}$，通过软更新$\phi' \leftarrow \tau \phi + (1 - \tau) \phi'$和$\theta' \leftarrow \tau \theta + (1 - \tau) \theta'$来稳定训练；
3. 探索噪声：在线执行时在Actor输出的动作上添加噪声（如Ornstein-Uhlenbeck噪声）来鼓励探索；
4. 更新Critic网络：使用Bellman方程更新Critic网络的参数$\phi$。Critic更新若干步之后，再做一次Actor网络的更新。
5. 
| 特性     | 优点                                        | 缺点                                             |
| ------ | ----------------------------------------- | ---------------------------------------------- |
| 样本效率   | 离线学习（off‐policy）＋经验重用 → 样本利用率高            | 如果行为分布与目标策略差距大，可能引入偏差（off‐policy bias）         |
| 方差     | 确定性策略“消除”了随机策略的方差来源 → 梯度方差低               | Critic 拟合 Q(s,a) 难度大 → 易发散，需要大量调参与技巧           |
| 连续动作支持 | 直接输出连续动作，无需对动作分布求导                        | 难以处理离散动作；在高维动作空间中依赖良好的探索噪声                     |
| 算法复杂度  | 只需一个 Actor＋一个 Critic，结合 Target 网络和回放池即可实现 | 需要维护 Target 网络、回放池、探索噪声、超参（学习率、τ、Batch Size）众多 |

DDPG适用于连续控制任务，例如机器人控制、自动驾驶、高维机械臂、模拟物理控制等。同时，对于样本宝贵但是可以离线记录的任务（如游戏回放、模拟器数据），DDPG也能发挥优势。

---

### Soft Actor-Critic (SAC)

Soft Actor–Critic (SAC) 是一款基于最大化“回报＋策略熵”目标的 off-policy Actor–Critic 算法，兼顾了高样本效率和良好探索性，特别适合连续动作空间。普通 RL 只追求最大化期望回报，往往在策略收敛后会变得过于确定性（deterministic），缺少探索。最大熵框架 在目标里加入策略熵项
$$
J(\pi)=\sum_{t}\mathbb{E}_{(s_t,a_t)\sim\pi}\Bigl[r(s_t,a_t)+\alpha\,\mathcal{H}\bigl(\pi(\cdot\mid s_t)\bigr)\Bigr]
$$
在保证高回报的同时，鼓励策略保持一定的随机性，从而提高探索性。SAC中Critic包含两个价值网络$Q_{\phi_1}, Q_{\phi_2}$以避免过高估计偏差，目标值采用两个网络中的较小值：
$$
y = r + \gamma\Bigl[\min_{i=1,2}Q_{\bar\phi_i}(s',a') - \alpha\ln\pi_\theta(a'\mid s')\Bigr]
$$
其中 $a'\sim\pi_\theta(\cdot\mid s')$，$\bar\phi_i$ 为延迟更新的 target 网络参数。Actor参数化为高斯分布 $\pi_\theta(a\mid s)=\mathcal{N}(\mu_\theta(s),\sigma_\theta(s))$，对连续动作做 reparameterization（如$a=\tanh(\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon)$，$\epsilon\sim\mathcal{N}(0,I)$）。优化目标为最大化 Q 减去熵成本：
$$
J(\theta)=\mathbb{E}_{s\sim\mathcal{D},\,\epsilon}\Bigl[\alpha\ln\pi_\theta(a\mid s)-Q_{\phi}(s,a)\Bigr].
$$
SAC算法的优点在于高样本效率（off-policy）和高稳定性，并且可学习的温度参数可以自动调整探索的强度，在连续动作空间中具有广泛应用。缺点是复杂度高，超参数多，并且在高维动作下需要小心设计以调节噪声。



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

4. 在GAE中，当 $\lambda$ 从 0 变到 1 时，算法行为如何变化？在什么场景下倾向于选择较大或较小的 $\lambda$？
   - 当 $\lambda$ 从 0 变到 1 时，GAE的优势估计从单步TD残差（低偏差、高方差）逐渐过渡到多步回报（高偏差、低方差）。具体来说：
     - $\lambda=0$ 时，GAE等价于单步TD残差，优势估计仅依赖于当前状态和下一个状态的回报，方差最小，但偏差较大。
     - $\lambda=1$ 时，GAE等价于完整轨迹的回报，优势估计依赖于整个轨迹的回报，偏差最小，但方差较大。
     - 在 $\lambda$ 介于 0 和 1 之间时，GAE通过加权平均多步回报来平衡偏差和方差。
   - 选择大$\lambda$的情况：
     - 环境确定性高、奖励信号稳定。模拟器噪声小或奖励延迟长，需要更准确的长期回报估计；价值网络易于收敛且误差小，可放心依赖较少的 bootstrapping。
     - 追求策略无偏性。想尽量减少对价值网络的依赖，获得更“真实”的期望回报；适用于 offline RL、批量评估，或对最终性能精度要求高的场景。
     - 样本充足、计算可承受。可以接受更高的方差带来的抖动，只要长期收敛性更好；有充足的样本或大 batch 来平均掉噪声。
   - 选择小$\lambda$的情况：
     - 环境噪声大、奖励稀疏或不稳定。需要更快地适应环境变化。
     - 策略更新频繁。在高频率小步长更新（如异步并行、大模型微调等）下，用一步TD能保持更稳定的学习信号。
     - 样本稀缺、计算资源有限。需要更快地收敛，减少每次更新的方差；样本量小或 batch size 小，无法承受高方差带来的抖动。适合实时、在线学习、机器人控制、小游戏或短会和任务。
   - 实践经验：
     - PPO/A2C 通常使用 $\lambda \in [0.9, 0.98]$.
     - 对于长期依赖（CoT，长对话）任务，$\lambda$ 可以设置得更大到0.99.
     - 对于高速在线控制或非常嘈杂的环境，可以降到0.8或更低。