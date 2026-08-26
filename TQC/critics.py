import torch
import torch.nn as nn


class TQCCritic(nn.Module):
    def __init__(self, state_dim, action_dim=1, n_nets=3, n_quantiles=25):
        super(TQCCritic, self).__init__()
        self.n_nets = n_nets
        self.n_quantiles = n_quantiles
        
        # 对应论文：3个隐藏层，每层 512
        self.nets = nn.ModuleList()
        for _ in range(n_nets):
            net = nn.Sequential(
                nn.Linear(state_dim + action_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, n_quantiles) # 输出的是 M 个分位数
            )
            self.nets.append(net)

    def forward(self, state, action):
        x = state.view(state.size(0), -1)
        sa = torch.cat([x, action], dim=1)
        
        # 收集每个 critic 网络的输出: 最终维度 -> (Batch, n_nets, n_quantiles)
        quantiles = torch.stack([net(sa) for net in self.nets], dim=1)
        return quantiles