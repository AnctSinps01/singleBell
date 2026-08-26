import argparse
from itertools import product
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

    def checkpoint_state(
        self,
        step,
        best_eval_reward,
        best_success_rate=-1.0,
        best_worst_reward=-float("inf"),
    ):
        return {
            "version": 1,
            "step": int(step),
            "best_eval_reward": float(best_eval_reward),
            "best_success_rate": float(best_success_rate),
            "best_worst_reward": float(best_worst_reward),
            "agent_config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "n_nets": self.n_nets,
                "n_quantiles": self.n_quantiles,
                "top_quantiles_to_drop": self.top_quantiles_to_drop,
                "gamma": self.gamma,
                "tau": self.tau,
            },
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def save_checkpoint(
        self,
        path,
        step,
        best_eval_reward,
        best_success_rate=-1.0,
        best_worst_reward=-float("inf"),
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.checkpoint_state(
                step,
                best_eval_reward,
                best_success_rate,
                best_worst_reward,
            ),
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False
        )
        config = checkpoint.get("agent_config", {})
        expected = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "n_nets": self.n_nets,
            "n_quantiles": self.n_quantiles,
        }
        mismatches = {
            key: (config.get(key), value)
            for key, value in expected.items()
            if config.get(key, value) != value
        }
        if mismatches:
            raise ValueError(f"checkpoint configuration mismatch: {mismatches}")

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        return {
            "step": int(checkpoint.get("step", 0)),
            "best_eval_reward": float(
                checkpoint.get("best_eval_reward", -float("inf"))
            ),
            "best_success_rate": float(
                checkpoint.get("best_success_rate", -1.0)
            ),
            "best_worst_reward": float(
                checkpoint.get("best_worst_reward", -float("inf"))
            ),
        }

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
def build_evaluation_cases(
    n_poles,
    diagonal_cases=8,
    random_cases=8,
    seed=0,
):
    """构造覆盖平衡点、耦合姿态和独立姿态的固定评估集。"""
    cases = []

    for angles in product((0.0, np.pi), repeat=n_poles):
        cases.append(("equilibrium", np.asarray(angles, dtype=np.float64)))

    diagonal_angles = np.linspace(
        -np.pi, np.pi, diagonal_cases, endpoint=False
    )
    for angle in diagonal_angles:
        cases.append(
            ("diagonal", np.full(n_poles, angle, dtype=np.float64))
        )

    rng = np.random.default_rng(seed)
    for angles in rng.uniform(-np.pi, np.pi, (random_cases, n_poles)):
        cases.append(("random", angles))

    return cases


def evaluate_agent(
    agent,
    env,
    history_length,
    n_poles,
    eval_steps=500,
    success_window=100,
):
    """在固定初态集上评估回零性能，返回奖励和控制质量指标。"""
    cases = build_evaluation_cases(n_poles)
    eval_stacker = FrameStacker(history_length=history_length)
    episodes = []

    for suite, angles in cases:
        obs = env.reset(ang=angles.tolist())
        state_tensor = eval_stacker.reset(obs)
        rewards = []
        actions = []
        stable = []

        for _ in range(eval_steps):
            action = agent.select_action(state_tensor, evaluate=True).item()
            next_obs, reward = env.step(action)
            state_tensor = eval_stacker.push(next_obs)
            rewards.append(reward)
            actions.append(abs(action))
            stable.append(
                abs(env.q[0]) <= 0.2
                and np.max(np.abs(env.q[1:])) <= np.deg2rad(10.0)
                and abs(env.dq[0]) <= 0.5
                and np.max(np.abs(env.dq[1:])) <= 0.5
            )

        window = min(success_window, eval_steps)
        episodes.append(
            {
                "suite": suite,
                "mean_reward": float(np.mean(rewards)),
                "success": bool(np.all(stable[-window:])),
                "final_angle_error": float(
                    np.max(np.abs(env.q[1:]))
                ),
                "final_cart_error": float(abs(env.q[0])),
                "mean_action": float(np.mean(actions)),
            }
        )

    suite_success = {}
    for suite in sorted({episode["suite"] for episode in episodes}):
        suite_episodes = [
            episode for episode in episodes if episode["suite"] == suite
        ]
        suite_success[suite] = float(
            np.mean([episode["success"] for episode in suite_episodes])
        )

    return {
        "mean_reward": float(
            np.mean([episode["mean_reward"] for episode in episodes])
        ),
        "worst_reward": float(
            np.min([episode["mean_reward"] for episode in episodes])
        ),
        "success_rate": float(
            np.mean([episode["success"] for episode in episodes])
        ),
        "mean_final_angle_error": float(
            np.mean(
                [episode["final_angle_error"] for episode in episodes]
            )
        ),
        "mean_final_cart_error": float(
            np.mean([episode["final_cart_error"] for episode in episodes])
        ),
        "mean_action": float(
            np.mean([episode["mean_action"] for episode in episodes])
        ),
        "suite_success": suite_success,
        "episodes": len(episodes),
    }


def format_evaluation(result):
    suite_text = ", ".join(
        f"{name}={rate:.0%}"
        for name, rate in result["suite_success"].items()
    )
    return (
        f"reward={result['mean_reward']:.4f} | "
        f"worst={result['worst_reward']:.4f} | "
        f"success={result['success_rate']:.1%} | "
        f"angle={np.rad2deg(result['mean_final_angle_error']):.2f} deg | "
        f"cart={result['mean_final_cart_error']:.3f} | "
        f"|action|={result['mean_action']:.3f} | "
        f"suites: {suite_text}"
    )


def train_tqc(resume_path=None, max_steps=2_000_000):
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

    batch_size = 256
    start_steps = 10000 # 初始随机探索步数
    eval_interval = 5000 # 评估间隔步数

    model_dir = Path(__file__).resolve().parent
    actor_path = model_dir / f"tqc_actor_{n_poles}.pth"
    checkpoint_path = model_dir / f"tqc_checkpoint_{n_poles}.pth"
    obs = env.reset()
    state = stacker.reset(obs).squeeze(0).numpy()

    best_eval_reward = -float('inf')
    best_success_rate = -1.0
    best_worst_reward = -float("inf")
    initial_step = 0
    if resume_path is not None:
        resume_state = agent.load_checkpoint(resume_path)
        initial_step = resume_state["step"]
        best_eval_reward = resume_state["best_eval_reward"]
        best_success_rate = resume_state["best_success_rate"]
        best_worst_reward = resume_state["best_worst_reward"]
        print(f"Resumed TQC checkpoint at step {initial_step}.")

    episode_reward = 0
    episode_length = 0

    for t in range(initial_step, max_steps):
        # 1. 探索与收集数据
        if len(replay_buffer) < start_steps:
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
        if len(replay_buffer) >= max(start_steps, batch_size):
            agent.update(replay_buffer, batch_size=batch_size)
            
        # 3. 评估与保存
        if t % eval_interval == 0 and t >= start_steps:
            print("\n--- Running Fixed Angle Evaluation ---")
            evaluation = evaluate_agent(
                agent,
                eval_env,
                history_length,
                n_poles,
            )
            current_eval_reward = evaluation["mean_reward"]
            print(
                f"Step {t} | {format_evaluation(evaluation)} | "
                f"best_success={best_success_rate:.1%} | "
                f"best_reward={best_eval_reward:.4f}"
            )

            current_rank = (
                evaluation["success_rate"],
                current_eval_reward,
                evaluation["worst_reward"],
            )
            best_rank = (
                best_success_rate,
                best_eval_reward,
                best_worst_reward,
            )
            if current_rank > best_rank:
                best_success_rate = evaluation["success_rate"]
                best_eval_reward = current_eval_reward
                best_worst_reward = evaluation["worst_reward"]
                torch.save(agent.actor.state_dict(), actor_path)
                print(
                    ">>> New best actor saved: "
                    f"success={best_success_rate:.1%}, "
                    f"reward={best_eval_reward:.4f} <<<\n"
                )

            agent.save_checkpoint(
                checkpoint_path,
                step=t + 1,
                best_eval_reward=best_eval_reward,
                best_success_rate=best_success_rate,
                best_worst_reward=best_worst_reward,
            )

def parse_args():
    parser = argparse.ArgumentParser(description="Train a TQC agent.")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=2_000_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_tqc(resume_path=args.resume, max_steps=args.max_steps)
