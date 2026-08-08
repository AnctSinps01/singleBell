import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from actor import ActorNetwork
from critic import CriticNetwork
from frame_stack import FrameStacker
from environ import NPendulumEnv
from settings import Settings


class PPOAgent:

    def __init__(
        self,
        history_length=4,
        n_poles=2,
        lr=5e-5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.1,
    ):
        self.actor = ActorNetwork(history_length, n_poles)
        self.critic = CriticNetwork(history_length, n_poles)

        try:
            a_checkpoint = torch.load(
                f"actor_{n_poles}.pth", map_location="cpu"
            )
            c_checkpoint = torch.load(
                f"critic_{n_poles}.pth", map_location="cpu"
            )

            self.actor.load_state_dict(a_checkpoint)
            self.critic.load_state_dict(c_checkpoint)

            print("Successfully Load Model.")

        except FileNotFoundError:
            print("File Not Found. Train from random paras.")
        except RuntimeError as e:
            print(f"Failed Loading Model: {e}")
        except Exception as e:
            # 捕获其他所有未预见的异常
            print(f"Unknown Error: {e}")

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = 0.01  # 熵正则化系数，防止过早收敛

    def select_action(self, state):
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action_and_log_prob(state)
            value = self.critic(state)
        return action.item(), log_prob.item(), value.item()

    def update(self, rollouts, ppo_epochs=10, batch_size=64):
        # 展开 rollout 数据
        b_states = torch.cat(rollouts["states"])
        b_actions = torch.tensor(
            rollouts["actions"], dtype=torch.float32
        ).unsqueeze(1)
        b_logprobs = torch.tensor(
            rollouts["logprobs"], dtype=torch.float32
        ).unsqueeze(1)
        b_rewards = torch.tensor(rollouts["rewards"], dtype=torch.float32)
        b_values = torch.tensor(rollouts["values"], dtype=torch.float32)

        # 计算 GAE (不考虑终止状态，视为无限连续任务)
        with torch.no_grad():
            b_next_value = self.critic(rollouts["next_state"]).squeeze(-1)
            advantages = torch.zeros_like(b_rewards)
            lastgaelam = 0
            for t in reversed(range(len(b_rewards))):
                if t == len(b_rewards) - 1:
                    nextvalues = b_next_value
                else:
                    nextvalues = b_values[t + 1]

                delta = b_rewards[t] + self.gamma * nextvalues - b_values[t]
                advantages[t] = lastgaelam = (
                    delta + self.gamma * self.gae_lambda * lastgaelam
                )

            returns = advantages + b_values

        advantages = advantages.unsqueeze(1)
        returns = returns.unsqueeze(1)

        # 优势标准化 (Advantage Normalization)
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        # PPO 迭代更新
        dataset_size = len(b_states)
        indices = np.arange(dataset_size)

        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                # 获取 mini-batch
                mb_states = b_states[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_logprobs = b_logprobs[mb_idx]

                # 重新计算 log_prob 和 value
                _, new_logprobs, entropy = self.actor.get_action_and_log_prob(
                    mb_states, mb_actions
                )
                new_values = self.critic(mb_states)

                # --- Actor Loss (Policy) ---
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                pg_loss1 = mb_advantages * ratio
                pg_loss2 = mb_advantages * torch.clamp(
                    ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef
                )
                actor_loss = -torch.min(pg_loss1, pg_loss2).mean()

                # --- Critic Loss (Value) ---
                critic_loss = nn.MSELoss()(new_values, mb_returns)

                # --- 总 Loss ---
                loss = actor_loss - self.ent_coef * entropy.mean()

                # 优化 Actor
                self.actor_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optimizer.step()

                # 优化 Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=0.5
                )
                self.critic_optimizer.step()


def evaluate_agent(agent, env, history_length, n_poles, eval_steps):
    """在 12 个固定角度下评估当前 Agent 的性能

    测试角度: 0°, 30°, 60°, ..., 330° (转换为弧度)
    """
    # 1. 生成 12 个均匀分布的角度 (弧度制)
    test_angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    total_eval_reward = 0.0

    # 临时创建一个 stacker 避免污染训练的 stacker 数据
    eval_stacker = FrameStacker(history_length=history_length)

    for base_ang in test_angles:
        # env.reset(ang=...) 要求 len(ang) == n_poles
        # 将 N 个杆都设为相同的初始角度
        if 0:
            ang_vector = [base_ang] * n_poles
        else:
            ang_vector = []

        # 重置环境到指定角度
        obs = env.reset(ang=ang_vector)

        # 预热 stacker
        for _ in range(history_length):
            obs, _ = env.step(0.0)
            eval_stacker.push(obs)

        state_tensor = eval_stacker.get_stacked_state()
        ep_reward = 0.0

        # 在当前角度下运行指定步数
        for _ in range(eval_steps):
            action, _, _ = agent.select_action(state_tensor)
            next_obs, reward = env.step(action)
            state_tensor = eval_stacker.push(next_obs)
            ep_reward += reward

        total_eval_reward += ep_reward

    # 计算 12 个角度下的平均每步奖励 (或平均 Episode 奖励)
    avg_eval_reward = total_eval_reward / (len(test_angles) * eval_steps)
    return avg_eval_reward


def train_ppo():
    SET = Settings()
    n_poles = SET.POLES
    history_length = SET.HISTORY

    env = NPendulumEnv(n=n_poles)
    stacker = FrameStacker(history_length=history_length)
    agent = PPOAgent(history_length=history_length, n_poles=n_poles)

    total_iterations = 10000  # 总更新次数 (Iteration)
    steps_per_update = 2000  # 每次更新收集的步数

    global_step = 0

    best_eval_reward = evaluate_agent(
        agent, env, history_length, n_poles, steps_per_update)
    print(f"--- Init Score: {best_eval_reward:.4f} ---")


    for iteration in range(1, total_iterations + 1):
        rollouts = {
            "states": [],
            "actions": [],
            "logprobs": [],
            "rewards": [],
            "values": [],
        }

        iteration_reward = 0

        obs = env.reset()
        for _ in range(history_length):
            obs, _ = env.step(0.0)
            stacker.push(obs)
        state_tensor = stacker.get_stacked_state()

        # 步数收集逻辑
        for _ in range(steps_per_update):
            # 1. 代理选择动作
            action, log_prob, value = agent.select_action(state_tensor)

            # 2. 与环境交互
            next_obs, reward = env.step(action)
            next_state_tensor = stacker.push(next_obs)

            # 3. 记录数据
            rollouts["states"].append(state_tensor)
            rollouts["actions"].append(action)
            rollouts["logprobs"].append(log_prob)
            rollouts["rewards"].append(reward)
            rollouts["values"].append(value)

            # 状态推进
            state_tensor = next_state_tensor
            iteration_reward += reward
            global_step += 1

        # 收集完 2048 步后，记录最后一步的状态用于 GAE 计算
        rollouts["next_state"] = next_state_tensor

        # 执行网络更新
        agent.update(rollouts, ppo_epochs=10, batch_size=64)

        # 打印日志 (平均每步奖励)
        if iteration % 1 == 0:
            avg_reward = iteration_reward / steps_per_update
            print(
                f"Iteration: {iteration} | Global Steps: {global_step} | Avg Reward/Step: {avg_reward:.4f}"
            )

        # 保存模型
        if iteration % 100 == 0:
            print("\n--- Running Fixed Angle Evaluation ---")
            current_eval_reward = evaluate_agent(
                agent, env, history_length, n_poles, steps_per_update)
            print(
                f"Iteration {iteration} \
                Eval Score: {current_eval_reward:.4f} \
                | Best So Far: {best_eval_reward:.4f}"
            )

            # 新权重表现更好才更新保存
            if current_eval_reward > best_eval_reward:
                best_eval_reward = current_eval_reward
                torch.save(agent.actor.state_dict(), f"actor_{n_poles}.pth")
                torch.save(agent.critic.state_dict(), f"critic_{n_poles}.pth")
                print(
                    f">>> New Best Model Saved with Score: {best_eval_reward:.4f} <<<\n"
                )
            else:
                print(">>> New Model did not improve. Skip saving. <<<\n")

        # if iteration % 100 == 0:
        #     torch.save(agent.actor.state_dict(), f"actor_{n_poles}.pth")
        #     torch.save(agent.critic.state_dict(), f"critic_{n_poles}.pth")
        #     print("Model Saved.")

    

if __name__ == "__main__":
    train_ppo()