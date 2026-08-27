from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from PPO.actor import ActorNetwork
from TQC.actor import TQCActor
from frame_stack import FrameStacker
from environ import NPendulumEnv


# ==================== 推理配置：直接修改这里 ====================
ALGORITHM = "tqc"       # "ppo" 或 "tqc"
ACTOR_PATH = None        # None: 自动选择对应算法的权重；也可填写 Path("...")
N_POLES = 2
HISTORY_LENGTH = 6       # 必须与训练该权重时的 HISTORY 一致
DETERMINISTIC = True     # True: 均值动作；False: 随机采样动作
MAX_STEPS = 2000
# ==============================================================


class Policy:
    def __init__(self, actor, algorithm, device):
        self.actor = actor
        self.algorithm = algorithm
        self.device = device

    def select_action(self, state, deterministic=True):
        state = state.to(self.device)
        with torch.no_grad():
            if self.algorithm == "ppo":
                if deterministic:
                    mu, _ = self.actor(state)
                    action = torch.tanh(mu)
                else:
                    action, _, _ = self.actor.get_action_and_log_prob(state)
            else:
                action = self.actor.act(state, deterministic=deterministic)
        return action.cpu().item()


def load_actor(
    algorithm,
    actor_path,
    history_length=4,
    n_poles=2,
    device="cpu",
):
    """加载 PPO 或 TQC actor，不依赖对应 critic。"""
    algorithm = algorithm.lower()
    if algorithm == "ppo":
        actor = ActorNetwork(
            history_length=history_length,
            n_poles=n_poles,
        )
    elif algorithm == "tqc":
        input_dim = history_length * (1 + n_poles)
        actor = TQCActor(input_dim=input_dim, action_dim=1)
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}")

    checkpoint = torch.load(actor_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "actor" in checkpoint:
        checkpoint = checkpoint["actor"]
    actor.load_state_dict(checkpoint)
    actor.to(device)
    actor.eval()
    return Policy(actor, algorithm, device)


def compute_positions(q, l_poles):
    """
    根据状态向量 q = [x, θ1, θ2, ...] 计算各节点的 (x, y) 坐标。
    节点 0 = 小车，节点 i = 第 i 根杆的末端 (i=1..n)。
    """
    x_cart = q[0]
    theta = q[1:]
    n = len(theta)

    # 杆的长度
    l = l_poles  # shape (n,)

    # 累积坐标
    x_nodes = [x_cart]
    y_nodes = [0.0]  # 小车高度 y=0

    cum_x = x_cart
    cum_y = 0.0
    for i in range(n):
        cum_x += l[i] * np.sin(theta[i])
        cum_y += l[i] * np.cos(theta[i])
        x_nodes.append(cum_x)
        y_nodes.append(cum_y)

    return np.array(x_nodes), np.array(y_nodes)


def render_setup(ax, algorithm, n_poles, l_poles, x_threshold):
    """初始化 matplotlib 绘图元素，返回需要动态更新的对象引用"""
    # 轨道
    ax.axhline(y=0, color='gray', linewidth=2, linestyle='-', alpha=0.5)

    # 轨道边界
    ax.axvline(x=-x_threshold, color='red', linewidth=1, linestyle='--', alpha=0.4)
    ax.axvline(x=x_threshold, color='red', linewidth=1, linestyle='--', alpha=0.4)

    # 小车 (矩形)
    cart_width = 0.3
    cart_height = 0.15
    cart = Rectangle((-cart_width / 2, -cart_height / 2), cart_width, cart_height,
                     fc='steelblue', ec='black', linewidth=2)

    # 杆的线条
    pole_lines = []
    for i in range(n_poles):
        (line,) = ax.plot([], [], lw=3, color=plt.cm.viridis(i / max(n_poles, 1)), marker='o', markersize=4)
        pole_lines.append(line)

    # 节点圆点
    (joints_scatter,) = ax.plot([], [], 'o', color='darkorange', markersize=6, zorder=5)

    # 信息文字
    info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', family='monospace')

    ax.add_patch(cart)

    # 固定显示范围
    ax.set_xlim(-x_threshold - 0.5, x_threshold + 0.5)
    total_length = np.sum(l_poles) + 0.3
    ax.set_ylim(-total_length - 0.2, total_length + 0.2)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{n_poles}-Pendulum {algorithm.upper()} Inference')

    # 用于动态更新的引用
    elements = {
        'cart': cart,
        'pole_lines': pole_lines,
        'joints': joints_scatter,
        'info_text': info_text,
        'ax': ax,
    }
    return elements


def run_inference(
    actor_path,
    algorithm="ppo",
    n_poles=2,
    history_length=4,
    deterministic=True,
    max_steps=2000,
):
    """
    主推理循环：固定运行 xxxx 步；
    deterministic=True 时使用均值动作（无探索），否则从分布采样。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. 只加载 actor
    policy = load_actor(
        algorithm,
        actor_path,
        history_length,
        n_poles,
        device,
    )

    # 2. 初始化环境与帧堆叠器
    env = NPendulumEnv(n=n_poles)
    stacker = FrameStacker(history_length=history_length)

    # 3. 设置 matplotlib
    plt.ion()  # 交互模式
    fig, ax = plt.subplots(figsize=(8, 6))
    elements = render_setup(
        ax,
        algorithm,
        n_poles,
        env.l_poles,
        env.x_threshold,
    )

    # 4. 状态变量
    obs = env.reset()
    state_tensor = stacker.reset(obs)
    episode_reward = 0.0
    step_count = 0
    print(f"Starting inference for fixed {max_steps} steps... Close the plot window to exit early.")
    print(f"Deterministic mode: {deterministic}")

    while plt.fignum_exists(fig.number) and step_count < max_steps:
        action = policy.select_action(state_tensor, deterministic)

        # --- 环境步进 (直接忽略 returned done 标志) ---
        next_obs, reward = env.step(action)
        next_state_tensor = stacker.push(next_obs)

        episode_reward += reward
        step_count += 1

        # --- 渲染 ---
        x_nodes, y_nodes = compute_positions(env.q, env.l_poles)

        # 更新小车位置
        cart = elements['cart']
        cart.set_x(x_nodes[0] - 0.15)  # 矩形左下角 x

        # 更新杆
        for i, line in enumerate(elements['pole_lines']):
            # 第 i 根杆从节点 i 延伸到节点 i+1
            line.set_data([x_nodes[i], x_nodes[i + 1]], [y_nodes[i], y_nodes[i + 1]])

        # 更新节点
        elements['joints'].set_data(x_nodes, y_nodes)

        # 更新文字
        info_str = (f"Step: {step_count}/{max_steps}\n"
                    f"Reward: {episode_reward:.3f}\n"
                    f"Action: {action:+.3f}\n"
                    f"Cart x: {env.q[0]:+.3f}\n"
                    f"Max θ:  {np.max(np.abs(env.q[1:])):.3f} rad")
        elements['info_text'].set_text(info_str)

        fig.canvas.draw()
        fig.canvas.flush_events()
        # plt.pause(0.001)  # 小幅暂停让图形更新

        state_tensor = next_state_tensor

    plt.ioff()
    plt.close()
    print(f"Inference finished. Total Steps: {step_count}, Cumulative Reward: {episode_reward:.3f}")


def default_actor_path(algorithm, n_poles):
    if algorithm == "ppo":
        return PROJECT_ROOT / "PPO" / f"actor_{n_poles}.pth"
    return PROJECT_ROOT / "TQC" / f"tqc_actor_{n_poles}.pth"


def validate_config():
    algorithm = ALGORITHM.lower()
    if algorithm not in {"ppo", "tqc"}:
        raise ValueError("ALGORITHM 必须是 'ppo' 或 'tqc'")
    if N_POLES < 1:
        raise ValueError("N_POLES 必须大于等于 1")
    if HISTORY_LENGTH < 1:
        raise ValueError("HISTORY_LENGTH 必须大于等于 1")
    if MAX_STEPS < 1:
        raise ValueError("MAX_STEPS 必须大于等于 1")

    actor_path = (
        default_actor_path(algorithm, N_POLES)
        if ACTOR_PATH is None
        else Path(ACTOR_PATH).expanduser()
    )
    if not actor_path.is_absolute():
        actor_path = PROJECT_ROOT / actor_path
    if not actor_path.is_file():
        raise FileNotFoundError(f"找不到 actor 权重: {actor_path}")
    return algorithm, actor_path


if __name__ == "__main__":
    algorithm, actor_path = validate_config()
    print(
        f"Algorithm: {algorithm.upper()} | Actor: {actor_path} | "
        f"Poles: {N_POLES} | History: {HISTORY_LENGTH}"
    )
    run_inference(
        actor_path=actor_path,
        algorithm=algorithm,
        n_poles=N_POLES,
        history_length=HISTORY_LENGTH,
        deterministic=DETERMINISTIC,
        max_steps=MAX_STEPS,
    )
