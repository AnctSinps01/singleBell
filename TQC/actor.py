import torch
import torch.nn as nn


class TQCActor(nn.Module):
    def __init__(self, input_dim, action_dim=1):
        super(TQCActor, self).__init__()
        # 对应论文：隐藏层 400 -> 300
        self.net = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(300, action_dim)
        self.log_std_layer = nn.Linear(300, action_dim)
        
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state):
        # Flatten state: (Batch, History, Features) -> (Batch, Input_dim)
        x = state.view(state.size(0), -1)
        net_out = self.net(x)
        
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        
        # 重参数化采样 (rsample 允许梯度回传)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t) # 将动作限制在 [-1, 1]
        action = y_t
        
        # 计算 log_prob，并修正由于 tanh 引起的概率变化
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob