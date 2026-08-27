from itertools import product
from pathlib import Path
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.optim as optim
import numpy as np

from environ import NPendulumEnv
from settings import Settings
from TQC.buffer import ReplayBuffer
from TQC.actor import TQCActor
from TQC.critics import TQCCritic


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if (
        hasattr(torch, "xpu")
        and torch.xpu.is_available()
        and hasattr(torch.xpu, "get_rng_state_all")
    ):
        state["torch_xpu"] = torch.xpu.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint requires CUDA RNG state")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "torch_xpu" in state:
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("checkpoint requires XPU RNG state")
        torch.xpu.set_rng_state_all(state["torch_xpu"])


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary_path)
    temporary_path.replace(path)


def environment_state_dict(env):
    return {
        "n": env.n,
        "q": env.q.copy(),
        "dq": env.dq.copy(),
    }


def load_environment_state(env, state):
    if int(state["n"]) != env.n:
        raise ValueError(
            f"environment pole count mismatch: {state['n']} != {env.n}"
        )
    q = np.asarray(state["q"], dtype=np.float64)
    dq = np.asarray(state["dq"], dtype=np.float64)
    if q.shape != env.q.shape or dq.shape != env.dq.shape:
        raise ValueError("checkpoint environment state has invalid shape")
    env.q[...] = q
    env.dq[...] = dq


def default_training_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


class TQCAgent:
    def __init__(
        self,
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
        self.device = (
            torch.device(device) if device else default_training_device()
        )
        self.state_dim = 2 + 2 * n_poles
        self.action_dim = 1
        self.n_nets = n_nets
        self.n_quantiles = n_quantiles
        self.top_quantiles_to_drop = top_quantiles_to_drop
        self.total_quantiles = n_nets * n_quantiles
        self.quantile_tau = (
            (torch.arange(n_quantiles, device=self.device) + 0.5)
            / n_quantiles
        ).view(1, 1, n_quantiles, 1)

        if not 0 <= top_quantiles_to_drop < self.total_quantiles:
            raise ValueError(
                "top_quantiles_to_drop must be in "
                f"[0, {self.total_quantiles})"
            )

        self.actor = TQCActor(self.state_dim, self.action_dim).to(self.device)
        if self.device.type == "cpu":
            self.action_actor = self.actor
        else:
            self.action_actor = TQCActor(
                self.state_dim, self.action_dim
            ).cpu()
            self.sync_action_actor()
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

    def sync_action_actor(self):
        if self.action_actor is self.actor:
            return
        self.action_actor.load_state_dict(
            {
                name: value.detach().cpu()
                for name, value in self.actor.state_dict().items()
            }
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def checkpoint_state(
        self,
        step,
        best_eval_reward,
        best_success_rate=-1.0,
        best_worst_reward=-float("inf"),
        training_state=None,
    ):
        return {
            "version": 3,
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
                "target_entropy": self.target_entropy,
            },
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "training_state": training_state,
        }

    def save_checkpoint(
        self,
        path,
        step,
        best_eval_reward,
        best_success_rate=-1.0,
        best_worst_reward=-float("inf"),
        training_state=None,
    ):
        if training_state is None:
            raise ValueError("complete checkpoint requires training_state")
        atomic_torch_save(
            self.checkpoint_state(
                step,
                best_eval_reward,
                best_success_rate,
                best_worst_reward,
                training_state,
            ),
            path,
        )

    def load_checkpoint(self, path):
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False
        )
        if checkpoint.get("version") not in (2, 3):
            raise ValueError(
                "checkpoint does not contain complete training state"
            )
        if checkpoint.get("training_state") is None:
            raise ValueError("checkpoint is missing training_state")
        config = checkpoint.get("agent_config", {})
        expected = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "n_nets": self.n_nets,
            "n_quantiles": self.n_quantiles,
            "top_quantiles_to_drop": self.top_quantiles_to_drop,
            "gamma": self.gamma,
            "tau": self.tau,
            "target_entropy": self.target_entropy,
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
        self.sync_action_actor()
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
            "training_state": checkpoint["training_state"],
        }

    def select_action(self, state, evaluate=False):
        state = torch.as_tensor(state, dtype=torch.float32, device="cpu")
        if state.ndim == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            action = self.action_actor.act(state, deterministic=evaluate)
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
        tau = self.quantile_tau.to(dtype=quantiles.dtype)
        
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
        
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ---------------- 2. 更新 Actor ----------------
        self.critic.requires_grad_(False)
        new_action, log_prob = self.actor.sample(state)
        actor_quantiles = self.critic(state, new_action)
        actor_q = actor_quantiles.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        actor_loss = (self.alpha.detach() * log_prob - actor_q).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.sync_action_actor()
        self.critic.requires_grad_(True)

        # ---------------- 3. 更新 Alpha (温度系数) ----------------
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
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
    n_poles,
    eval_steps=500,
    success_window=100,
):
    """在固定初态集上评估回零性能，返回奖励和控制质量指标。"""
    cases = build_evaluation_cases(n_poles)
    episodes = []

    for suite, angles in cases:
        obs = env.reset(ang=angles.tolist())
        state_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        rewards = []
        actions = []
        stable = []

        for _ in range(eval_steps):
            action = agent.select_action(state_tensor, evaluate=True).item()
            next_obs, reward = env.step(action)
            state_tensor = torch.as_tensor(
                next_obs, dtype=torch.float32
            ).unsqueeze(0)
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


def train_tqc(
    resume_path=None,
    max_steps=2_000_000,
    batch_size=1024,
    update_interval=4,
    log_interval=1000,
):
    if batch_size <= 0 or update_interval <= 0 or log_interval <= 0:
        raise ValueError(
            "batch_size, update_interval, and log_interval must be positive"
        )
    SET = Settings()
    n_poles = SET.POLES
    
    # 初始化环境
    env = NPendulumEnv(n=n_poles)
    eval_env = NPendulumEnv(n=n_poles)
    
    # 初始化 Agent (对应论文：N=3, M=25)
    agent = TQCAgent(n_poles, lr=3e-4, gamma=0.99, tau=0.005)
    print(
        f"Training device: {agent.device}; "
        "action collection device: cpu (synchronized every actor update)"
    )
    
    # 论文设定经验回放池 1e6
    replay_buffer = ReplayBuffer(capacity=1000000)

    start_steps = 10000 # 初始随机探索步数
    eval_interval = 5000 # 评估间隔步数

    model_dir = Path(__file__).resolve().parent
    actor_path = model_dir / f"tqc_actor_{n_poles}.pth"
    checkpoint_path = model_dir / f"tqc_checkpoint_{n_poles}.pth"
    state = env.reset()

    best_eval_reward = -float('inf')
    best_success_rate = -1.0
    best_worst_reward = -float("inf")
    initial_step = 0
    if resume_path is not None:
        resume_state = agent.load_checkpoint(resume_path)
        training_state = resume_state["training_state"]
        saved_config = training_state["training_config"]
        current_config = {
            "batch_size": batch_size,
            "update_interval": update_interval,
            "start_steps": start_steps,
            "eval_interval": eval_interval,
        }
        if saved_config != current_config:
            raise ValueError(
                "training configuration mismatch: "
                f"checkpoint={saved_config}, current={current_config}"
            )
        replay_buffer.load_state_dict(training_state["replay_buffer"])
        load_environment_state(env, training_state["environment"])
        state = np.asarray(training_state["state"], dtype=np.float64).copy()
        if state.shape != env._get_obs().shape:
            raise ValueError("checkpoint observation has invalid shape")
        initial_step = resume_state["step"]
        best_eval_reward = resume_state["best_eval_reward"]
        best_success_rate = resume_state["best_success_rate"]
        best_worst_reward = resume_state["best_worst_reward"]
        episode_reward = float(training_state["episode_reward"])
        episode_length = int(training_state["episode_length"])
        update_count = int(training_state["log_update_count"])
        restore_rng_state(training_state["rng_state"])
        print(f"Resumed TQC checkpoint at step {initial_step}.")
    else:
        episode_reward = 0.0
        episode_length = 0
        update_count = 0
    log_started_at = time.perf_counter()
    log_started_step = initial_step

    for t in range(initial_step, max_steps):
        # 1. 探索与收集数据
        if len(replay_buffer) < start_steps:
            action = np.random.uniform(-1, 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = agent.select_action(state_tensor, evaluate=False).item()
            
        next_obs, reward = env.step(action)
        next_state = next_obs
        
        # 在连续控制中通常不存在自然 done（除非你想设置跌倒 done）
        # 这里统一当做 0 处理 (由于是转移控制，理论上系统一直运行)
        done = 0 
        
        replay_buffer.push(state, action, reward, next_state, done)
        
        state = next_state
        episode_reward += reward
        episode_length += 1
        
        # 为了与 PPO 代码一致，这里强制每 1000 步重置一次环境（可自定义）
        if episode_length >= 1000:
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            
        # 2. 合并多个环境步的样本到一次大批量更新，减少 XPU 调度开销。
        if (
            len(replay_buffer) >= max(start_steps, batch_size)
            and (t + 1) % update_interval == 0
        ):
            agent.update(replay_buffer, batch_size=batch_size)
            update_count += 1

        if (t + 1) % log_interval == 0:
            now = time.perf_counter()
            elapsed = now - log_started_at
            completed_steps = t + 1 - log_started_step
            print(
                f"Step {t + 1}/{max_steps} | "
                f"{completed_steps / elapsed:.1f} steps/s | "
                f"updates={update_count} | buffer={len(replay_buffer)}"
            )
            log_started_at = now
            log_started_step = t + 1
            update_count = 0
            
        # 3. 评估与保存
        if t % eval_interval == 0 and t >= start_steps:
            print("\n--- Running Fixed Angle Evaluation ---")
            evaluation = evaluate_agent(
                agent,
                eval_env,
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
                atomic_torch_save(agent.actor.state_dict(), actor_path)
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
                training_state={
                    "training_config": {
                        "batch_size": batch_size,
                        "update_interval": update_interval,
                        "start_steps": start_steps,
                        "eval_interval": eval_interval,
                    },
                    "replay_buffer": replay_buffer.state_dict(),
                    "environment": environment_state_dict(env),
                    "state": np.asarray(state, dtype=np.float64).copy(),
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "log_update_count": update_count,
                    "rng_state": capture_rng_state(),
                },
            )


if __name__ == "__main__":
    train_tqc(
        # "TQC/tqc_checkpoint_2.pth",
        batch_size=256,
        update_interval=1
    )
