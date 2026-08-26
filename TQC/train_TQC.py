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
    def __init__(
        self,
        history_length,
        n_poles,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        n_nets=3,
        n_quantiles=25,
        top_quantiles_to_drop=5,
        device=None,
    ):
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.state_dim = history_length * (1 + n_poles)
        self.action_dim = 1
        self.n_nets = n_nets
        self.n_quantiles = n_quantiles
        self.top_quantiles_to_drop = top_quantiles_to_drop
        self.total_quantiles = n_nets * n_quantiles

        if not 0 <= top_quantiles_to_drop < self.total_quantiles:
            raise ValueError(
                "top_quantiles_to_drop must be in "
                f"[0, {self.total_quantiles})"
            )

        self.actor = TQCActor(self.state_dim, self.action_dim).to(self.device)
        self.critic = TQCCritic(
            self.state_dim,
            self.action_dim,
            self.n_nets,
            self.n_quantiles,
        ).to(self.device)
        self.critic_target = TQCCritic(
            self.state_dim,
            self.action_dim,
            self.n_nets,
            self.n_quantiles,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.target_entropy = -float(self.action_dim)
        self.log_alpha = torch.zeros(
            1, device=self.device, requires_grad=True
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, evaluate=False):
        state = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        )
        if state.ndim in (1, 2):
            state = state.unsqueeze(0)

        with torch.no_grad():
            action = self.actor.act(state, deterministic=evaluate)
        return action.squeeze(0).cpu().numpy()

    def quantile_huber_loss(self, quantiles, target):
        """计算分位数 Huber 损失 (Quantile Huber Loss)"""
        # quantiles: (batch, n_nets, n_quantiles)
        # target: (batch, total_quantiles - drop)
        if quantiles.ndim != 3:
            raise ValueError("quantiles must have shape (B, N, M)")
        if target.ndim != 2:
            raise ValueError("target must have shape (B, target_quantiles)")
        if quantiles.size(0) != target.size(0):
            raise ValueError("quantiles and target batch sizes must match")
        
        # 扩展维度以便进行广播计算 pairwise 差距
        # target 扩展为 (batch, 1, 1, target_quantiles)
        target = target.unsqueeze(1).unsqueeze(1)
        # quantiles 扩展为 (batch, n_nets, n_quantiles, 1)
        quantiles = quantiles.unsqueeze(-1)
        
        pairwise_delta = target - quantiles
        
        # Huber 损失计算
        kappa = 1.0
        abs_u = torch.abs(pairwise_delta)
        huber_loss = torch.where(
            abs_u <= kappa,
            0.5 * abs_u.pow(2),
            kappa * (abs_u - 0.5 * kappa),
        )
        
        # 分位数权重 \tau
        tau = (
            torch.arange(
                self.n_quantiles,
                dtype=quantiles.dtype,
                device=quantiles.device,
            )
            + 0.5
        ) / self.n_quantiles
        tau = tau.view(1, 1, self.n_quantiles, 1)
        
        # 分位数回归损失
        quantile_weight = torch.abs(
            tau - (pairwise_delta.detach() < 0).to(quantiles.dtype)
        )
        loss = (quantile_weight * huber_loss).mean()
        return loss

    def update(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:
            raise ValueError(
                f"update requires {batch_size} transitions, "
                f"but the replay buffer contains {len(replay_buffer)}"
            )

        state, action, reward, next_state, done = replay_buffer.sample(
            batch_size, device=self.device
        )

        # ---------------- 1. 更新 Critic ----------------
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            next_quantiles = self.critic_target(next_state, next_action)
            sorted_quantiles = torch.sort(
                next_quantiles.flatten(start_dim=1), dim=1
            ).values
            kept_quantiles = self.total_quantiles - self.top_quantiles_to_drop
            target_quantiles = sorted_quantiles[:, :kept_quantiles]
            target = reward + (1.0 - done) * self.gamma * (
                target_quantiles - self.alpha.detach() * next_log_prob
            )

        # 当前 Critic 预测 (Batch, 3, 25)
        current_quantiles = self.critic(state, action)
        critic_loss = self.quantile_huber_loss(current_quantiles, target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------------- 2. 更新 Actor ----------------
        self.critic.requires_grad_(False)
        new_action, log_prob = self.actor.sample(state)
        actor_quantiles = self.critic(state, new_action)
        actor_q = actor_quantiles.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        actor_loss = (self.alpha.detach() * log_prob - actor_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        # ---------------- 3. 更新 Alpha (温度系数) ----------------
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # ---------------- 4. 软更新 Target 网络 ----------------
        with torch.no_grad():
            for target_param, param in zip(
                self.critic_target.parameters(), self.critic.parameters()
            ):
                target_param.lerp_(param, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.detach().item(),
        }


# ==========================================
# 5. 训练主循环 (Off-Policy 方式)
# ==========================================
def evaluate_agent(agent, env, history_length, n_poles, eval_steps=500):
    """测试评估环境性能"""
    test_angles = np.linspace(-np.pi, np.pi, 24, endpoint=False)
    total_eval_reward = 0.0
    eval_stacker = FrameStacker(history_length=history_length)

    for base_ang in test_angles:
        ang_vector = [base_ang] * n_poles
        obs = env.reset(ang=ang_vector)
        state_tensor = eval_stacker.reset(obs)
        ep_reward = 0.0

        for _ in range(eval_steps):
            # evaluate=True 会取均值而不采样，动作更稳定
            action = agent.select_action(state_tensor, evaluate=True).item()
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
    eval_env = NPendulumEnv(n=n_poles)
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
    state = stacker.reset(obs).squeeze(0).numpy()
    
    best_eval_reward = -float('inf')
    episode_reward = 0
    episode_length = 0
    
    for t in range(max_steps):
        # 1. 探索与收集数据
        if t < start_steps:
            action = np.random.uniform(-1, 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = agent.select_action(state_tensor, evaluate=False).item()
            
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
            state = stacker.reset(obs).squeeze(0).numpy()
            episode_reward = 0
            episode_length = 0
            
        # 2. 网络更新 (每次环境步执行一次更新，论文中 Gradient steps = 1)
        if t >= start_steps:
            agent.update(replay_buffer, batch_size=batch_size)
            
        # 3. 评估与保存
        if t % eval_interval == 0 and t >= start_steps:
            print("\n--- Running Fixed Angle Evaluation ---")
            current_eval_reward = evaluate_agent(
                agent,
                eval_env,
                history_length,
                n_poles,
            )
            print(
                f"Step {t} | Eval Score: {current_eval_reward:.4f} "
                f"| Best So Far: {best_eval_reward:.4f}"
            )
            
            if current_eval_reward > best_eval_reward:
                best_eval_reward = current_eval_reward
                torch.save(agent.actor.state_dict(), f"tqc_actor_{n_poles}.pth")
                torch.save(agent.critic.state_dict(), f"tqc_critic_{n_poles}.pth")
                print(f">>> New Best Model Saved with Score: {best_eval_reward:.4f} <<<\n")

if __name__ == "__main__":
    train_tqc()
