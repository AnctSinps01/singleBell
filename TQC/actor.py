import torch
import torch.nn as nn


class TQCActor(nn.Module):
    def __init__(self, input_dim, action_dim=1):
        super(TQCActor, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim

        # 对应论文：隐藏层 400 -> 300
        self.net = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(300, action_dim)
        self.log_std_layer = nn.Linear(300, action_dim)
        
        self.log_std_max = 2.0
        self.log_std_min = -20.0

    def forward(self, state):
        # Accept a batch of instantaneous full-state observations.
        x = state.reshape(state.size(0), -1)
        if x.size(1) != self.input_dim:
            raise ValueError(
                f"expected state dimension {self.input_dim}, "
                f"got {x.size(1)}"
            )
        net_out = self.net(x)

        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        
        # 重参数化采样 (rsample 允许梯度回传)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # 计算 log_prob，并修正由于 tanh 引起的概率变化
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

    def act(self, state, deterministic=True):
        if deterministic:
            mu, _ = self.forward(state)
            return torch.tanh(mu)
        action, _ = self.sample(state)
        return action
