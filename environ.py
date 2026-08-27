import numpy as np


class NPendulumEnv:
    def __init__(self, n: int=2):
        self.n = n
        self.g = 10.0
        self.dt = 0.01
        self.max_force = 50.0
        self.x_threshold = 2.0
        
        self.m_cart = 1.0
        self.m_poles = np.array([1.0] * n)
        self.l_poles = np.array([1.0] * n)
        
        # 预计算
        self.M_sum = np.array([np.sum(self.m_poles[i:]) for i in range(self.n)])
        
        # 创建 n x n 的索引网格以计算 M_max(i, j)
        i_idx, j_idx = np.indices((self.n, self.n))
        max_idx = np.maximum(i_idx, j_idx)
        self.M_max_mat = self.M_sum[max_idx]  # n x n 矩阵：对应 max(i,j) 及其以上的质量和
        
        # n x n 矩阵：L_i * L_j
        self.L_mat = self.l_poles[:, None] * self.l_poles[None, :]
        
        self.q = np.zeros(1 + self.n)
        self.dq = np.zeros(1 + self.n)
        
    def reset(self, ang: list=[]):
        self.q.fill(0.0)
        self.dq.fill(0.0)

        if ang:
            if len(self.q) - 1 != len(ang):
                raise ValueError("Wrong Dim With Angle Reset !")
            self.q[1:] = np.array(ang)
        else:
            if 0:
                diff = 0.05
            else:
                diff = np.pi
            self.q[1:] = np.random.uniform(-diff, diff, self.n)
        return self._get_obs()
        
    def _get_ddq(self, q: np.ndarray, dq: np.ndarray, force: float):
        """
        核心物理计算引擎：给定当前状态 (q, dq) 和外力，返回加速度 ddq。
        【完全向量化，无 for 循环】
        """
        q_angles = q[1:]
        dq_angles = dq[1:]
        
        H = np.zeros((self.n + 1, self.n + 1))
        C = np.zeros(self.n + 1)
        
        # 1. 填充惯性矩阵 H
        H[0, 0] = self.m_cart + self.M_sum[0]
        
        # 小车与摆的耦合 (1 x n 向量)
        H_cart_pole = self.M_sum * self.l_poles * np.cos(q_angles)
        H[0, 1:] = H_cart_pole
        H[1:, 0] = H_cart_pole
        
        # 摆与摆之间的耦合 (n x n 矩阵)
        theta_diff_mat = q_angles[:, None] - q_angles[None, :] # n x n 的角度差矩阵
        H[1:, 1:] = self.M_max_mat * self.L_mat * np.cos(theta_diff_mat)
        
        # 2. 填充外力与离心力、重力向量 C
        # 小车受力 = 推力 + 摆的离心力投影求和
        C[0] = force + np.sum(self.M_sum * self.l_poles * (dq_angles ** 2) * np.sin(q_angles))
        
        # 摆的受力 = 重力项 - 离心/科里奥利力交互项
        gravity_term = self.M_sum * self.g * self.l_poles * np.sin(q_angles)
        
        # 利用矩阵乘法/广播一次性计算交互项
        # 注意: 当 i==j 时, sin(0)=0，所以包含对角线也不会影响结果，直接全矩阵相加即可
        interaction_mat = self.M_max_mat * self.L_mat * (dq_angles[None, :] ** 2) * np.sin(theta_diff_mat)
        interaction_term = np.sum(interaction_mat, axis=1) # 对列求和
        
        C[1:] = gravity_term - interaction_term
        
        # 3. 求解加速度 ddq
        ddq = np.linalg.solve(H, C)
        return ddq

    def step(self, action_ratio: float):
        force = action_ratio * self.max_force
        
        # === 【精度优化】：使用四阶龙格-库塔法 (RK4) 替代欧拉法 ===
        # 状态表示可以打包为一个大向量: state = [q, dq]
        # RK4 需要在内部进行 4 次微小步长的导数估算
        
        q0 = self.q.copy()
        dq0 = self.dq.copy()
        
        # k1
        ddq1 = self._get_ddq(q0, dq0, force)
        k1_q = dq0
        k1_dq = ddq1
        
        # k2
        q2 = q0 + 0.5 * self.dt * k1_q
        dq2 = dq0 + 0.5 * self.dt * k1_dq
        ddq2 = self._get_ddq(q2, dq2, force)
        k2_q = dq2
        k2_dq = ddq2
        
        # k3
        q3 = q0 + 0.5 * self.dt * k2_q
        dq3 = dq0 + 0.5 * self.dt * k2_dq
        ddq3 = self._get_ddq(q3, dq3, force)
        k3_q = dq3
        k3_dq = ddq3
        
        # k4
        q4 = q0 + self.dt * k3_q
        dq4 = dq0 + self.dt * k3_dq
        ddq4 = self._get_ddq(q4, dq4, force)
        k4_q = dq4
        k4_dq = ddq4
        
        # 最终更新
        self.q += (self.dt / 6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
        self.dq += (self.dt / 6.0) * (k1_dq + 2*k2_dq + 2*k3_dq + k4_dq)
        # ===========================================================
        
        # 角度标准化至 [-pi, pi]
        self.q[1:] = (self.q[1:] + np.pi) % (2 * np.pi) - np.pi
        
        obs = self._get_obs()
        
        # 奖励设计
        angle_penalty = np.sum(np.abs(self.q))
        angular_velocity_weights = np.array([0.015, 0.009, 0.005])[
            np.minimum(np.arange(self.n), 2)
        ]
        angular_velocity_penalty = np.sum(
            angular_velocity_weights * np.abs(self.dq[1:])
        )
        reward = np.exp(-0.3 * angle_penalty - angular_velocity_penalty)
        
        # out_of_bounds = bool(abs(self.q[0]) > self.x_threshold)
        # fallen = bool(np.any(np.abs(self.q[1:]) > self.theta_threshold))
        # done = out_of_bounds or fallen
        
        return obs, reward
        
    def _get_obs(self) -> np.ndarray:
        # Full instantaneous state: cart/pole positions and velocities.
        return np.concatenate((self.q, self.dq)).copy()
