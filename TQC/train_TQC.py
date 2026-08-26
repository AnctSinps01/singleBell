import torch
import torch.optim as optim
import numpy as np

from environ import NPendulumEnv
from frame_stack import FrameStacker
from settings import Settings
from TQC.buffer import ReplayBuffer
from TQC.actor import TQCActor
from TQC.critics import TQCCritic


class TQCAgent:
    def __init__(self, history_length, n_poles, lr=3e-4, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        
        state_dim = history_length * (1 + n_poles)
        action_dim = 1
        
        # 论文参数设置
        self.n_nets = 3
        self.n_quantiles = 25
        # 截断（Truncation）：丢弃预测最高的一部分分位数，防止过估计
        self.top_quantiles_to_drop = 5 
        
        # 建立网络
        self.actor = TQCActor(state_dim, action_dim)
        self.critic = TQCCritic(state_dim, action_dim, self.n_nets, self.n_quantiles)
        self.critic_target = TQCCritic(state_dim, action_dim, self.n_nets, self.n_quantiles)
        
        # 初始化 Target 网络权重
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 自动调整温度系数 Alpha
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
    def select_action(self, state, evaluate=False):
        state = state.float()
        with torch.no_grad():
            if evaluate:
                mu, _ = self.actor(state)
                return torch.tanh(mu).item()
            else:
                action, _ = self.actor.sample(state)
                return action.item()

    def quantile_huber_loss(self, quantiles, target):
        """计算分位数 Huber 损失 (Quantile Huber Loss)"""
        # quantiles: (batch, n_nets, n_quantiles)
        # target: (batch, total_quantiles - drop)
        
        # 扩展维度以便进行广播计算 pairwise 差距
        # target 扩展为 (batch, 1, 1, target_quantiles)
        target = target.unsqueeze(1).unsqueeze(1)
        # quantiles 扩展为 (batch, n_nets, n_quantiles, 1)
        quantiles = quantiles.unsqueeze(-1)
        
        pairwise_delta = target - quantiles
        
        # Huber 损失计算
        kappa = 1.0
        abs_u = torch.abs(pairwise_delta)
        huber_loss = torch.where(abs_u <= kappa, 0.5 * abs_u.pow(2), kappa * (abs_u - 0.5 * kappa))
        
        # 分位数权重 \tau
        tau = (torch.arange(self.n_quantiles, dtype=torch.float32) + 0.5) / self.n_quantiles
        tau = tau.view(1, 1, self.n_quantiles, 1)
        
        # 分位数回归损失
        loss = (torch.abs(tau - (pairwise_delta.detach() < 0).float()) * huber_loss).mean()
        return loss

    def update(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        alpha = self.log_alpha.exp()
        
        # ---------------- 1. 更新 Critic ----------------
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            # 获取下一个状态的分位数 (Batch, 3, 25)
            next_quantiles = self.critic_target(next_state, next_action)
            
            # 将所有网络的分位数合并并排序 (Batch, 75)
            next_quantiles, _ = torch.sort(next_quantiles.view(batch_size, -1), dim=1)
            
            # **TQC 核心操作：截断 (Truncation)**
            # 丢弃最高的那几个分位数，抑制过估计
            target_quantiles = next_quantiles[:, : self.n_nets * self.n_quantiles - self.top_quantiles_to_drop]
            
            # 结合奖励与熵 (Batch, 70)
            target = reward + (1 - done) * self.gamma * (target_quantiles - alpha * next_log_prob)

        # 当前 Critic 预测 (Batch, 3, 25)
        current_quantiles = self.critic(state, action)
        critic_loss = self.quantile_huber_loss(current_quantiles, target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------------- 2. 更新 Actor ----------------
        new_action, log_prob = self.actor.sample(state)
        # 获取当前新动作的分位数评估
        actor_quantiles = self.critic(state, new_action)
        # TQC 建议 actor 优化时使用各网络均值来指导
        actor_q = actor_quantiles.mean(dim=2).mean(dim=1, keepdim=True)
        
        actor_loss = (alpha * log_prob - actor_q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ---------------- 3. 更新 Alpha (温度系数) ----------------
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # ---------------- 4. 软更新 Target 网络 ----------------
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


# ==========================================
# 5. 训练主循环 (Off-Policy 方式)
# ==========================================
def evaluate_agent(agent, env, history_length, n_poles, eval_steps=500):
    """测试评估环境性能"""
    test_angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    total_eval_reward = 0.0
    eval_stacker = FrameStacker(history_length=history_length)

    for base_ang in test_angles:
        ang_vector = [] # 或者 [base_ang] * n_poles 根据你的需求
        obs = env.reset(ang=ang_vector)

        for _ in range(history_length):
            obs, _ = env.step(0.0)
            eval_stacker.push(obs)

        state_tensor = eval_stacker.get_stacked_state()
        ep_reward = 0.0

        for _ in range(eval_steps):
            # evaluate=True 会取均值而不采样，动作更稳定
            action = agent.select_action(state_tensor, evaluate=True)
            next_obs, reward = env.step(action)
            state_tensor = eval_stacker.push(next_obs)
            ep_reward += reward

        total_eval_reward += ep_reward

    return total_eval_reward / (len(test_angles) * eval_steps)


def train_tqc():
    SET = Settings()
    n_poles = SET.POLES
    history_length = SET.HISTORY
    
    # 初始化环境
    env = NPendulumEnv(n=n_poles)
    stacker = FrameStacker(history_length=history_length)
    
    # 初始化 Agent (对应论文：N=3, M=25)
    agent = TQCAgent(history_length, n_poles, lr=3e-4, gamma=0.99, tau=0.005)
    
    # 论文设定经验回放池 1e6
    replay_buffer = ReplayBuffer(capacity=1000000)
    
    max_steps = 2_000_000
    batch_size = 256
    start_steps = 10000 # 初始随机探索步数
    eval_interval = 5000 # 评估间隔步数
    
    obs = env.reset()
    for _ in range(history_length):
        obs, _ = env.step(0.0)
        stacker.push(obs)
    state = stacker.get_stacked_state().squeeze(0).numpy() # (History * Features)
    
    best_eval_reward = -float('inf')
    episode_reward = 0
    episode_length = 0
    
    for t in range(max_steps):
        # 1. 探索与收集数据
        if t < start_steps:
            action = np.random.uniform(-1, 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = agent.select_action(state_tensor, evaluate=False)
            
        next_obs, reward = env.step(action)
        stacker.push(next_obs)
        next_state = stacker.get_stacked_state().squeeze(0).numpy()
        
        # 在连续控制中通常不存在自然 done（除非你想设置跌倒 done）
        # 这里统一当做 0 处理 (由于是转移控制，理论上系统一直运行)
        done = 0 
        
        replay_buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        episode_length += 1
        
        # 为了与 PPO 代码一致，这里强制每 1000 步重置一次环境（可自定义）
        if episode_length >= 1000:
            obs = env.reset()
            for _ in range(history_length):
                obs, _ = env.step(0.0)
                stacker.push(obs)
            state = stacker.get_stacked_state().squeeze(0).numpy()
            episode_reward = 0
            episode_length = 0
            
        # 2. 网络更新 (每次环境步执行一次更新，论文中 Gradient steps = 1)
        if t >= start_steps:
            agent.update(replay_buffer, batch_size=batch_size)
            
        # 3. 评估与保存
        if t % eval_interval == 0 and t >= start_steps:
            print("\n--- Running Fixed Angle Evaluation ---")
            current_eval_reward = evaluate_agent(agent, env, history_length, n_poles)
            print(f"Step {t} | Eval Score: {current_eval_reward:.4f} | Best So Far: {best_eval_reward:.4f}")
            
            if current_eval_reward > best_eval_reward:
                best_eval_reward = current_eval_reward
                torch.save(agent.actor.state_dict(), f"tqc_actor_{n_poles}.pth")
                torch.save(agent.critic.state_dict(), f"tqc_critic_{n_poles}.pth")
                print(f">>> New Best Model Saved with Score: {best_eval_reward:.4f} <<<\n")

if __name__ == "__main__":
    train_tqc()