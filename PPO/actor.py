import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorNetwork(nn.Module):
    def __init__(self, n_poles=2, action_dim=1):
        super(ActorNetwork, self).__init__()
        input_dim = 1 + 2 * n_poles
        
        # 共享特征提取
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
        )
        
        # 输出未经过 tanh 的均值 (logits)
        self.mu_head = nn.Linear(32, action_dim)
        
        # PPO 连续控制的标准做法：方差通常设为独立于状态的可学习参数
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        # ================= 显式初始化权重 =================
        self._init_weights()

    def _init_weights(self):
        # 1. 隐藏层：正交初始化，针对 LeakyReLU 增益
        gain = nn.init.calculate_gain('leaky_relu')
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.constant_(m.bias, 0.0)
                
        # 2. 动作均值输出层：极小增益(0.01)，保证初始均值接近 0，充分探索
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.constant_(self.mu_head.bias, 0.0)

    def forward(self, state):
        x = state.view(state.size(0), -1)

        x_concat = torch.cat([x, -x], dim=0)
        features = self.net(x_concat)
        raw_mu = self.mu_head(features)

        mu_pos, mu_neg = torch.chunk(raw_mu, 2, dim=0)
        mu = 0.5 * (mu_pos - mu_neg)

        std = self.log_std.exp().expand_as(mu)
        self.log_std.data.clamp_(-5.0, 2.0)
        return mu, std

    def get_action_and_log_prob(self, state, action=None):
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        
        if action is None:
            u = dist.sample()
            a = torch.tanh(u)
        else:
            a_clipped = torch.clamp(action, -0.999999, 0.999999)
            u = torch.atanh(a_clipped)
            a = a_clipped
            
        log_prob_u = dist.log_prob(u)
        log_prob_a = log_prob_u - torch.log(1.0 - a.pow(2) + 1e-6)
        log_prob = log_prob_a.sum(dim=-1, keepdim=True)
        
        return a, log_prob, dist.entropy().sum(dim=-1, keepdim=True)
