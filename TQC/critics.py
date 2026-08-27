import torch
import torch.nn as nn


class TQCCritic(nn.Module):
    def __init__(self, state_dim, action_dim=1, n_nets=3, n_quantiles=25):
        super(TQCCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
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
                nn.Linear(512, n_quantiles),
            )
            self.nets.append(net)

    def forward(self, state, action):
        x = state.reshape(state.size(0), -1)
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        action = action.reshape(action.size(0), -1)

        if x.size(1) != self.state_dim:
            raise ValueError(
                f"expected state dimension {self.state_dim}, "
                f"got {x.size(1)}"
            )
        if action.size(1) != self.action_dim:
            raise ValueError(
                f"expected action dimension {self.action_dim}, "
                f"got {action.size(1)}"
            )
        if x.size(0) != action.size(0):
            raise ValueError("state and action batch sizes must match")

        sa = torch.cat([x, action], dim=1)
        paired_sa = torch.cat([sa, -sa], dim=0)

        # 对每个分位数做偶对称投影，严格保证 Q(-s, -a) = Q(s, a)。
        raw_quantiles = torch.stack(
            [net(paired_sa) for net in self.nets], dim=1
        )
        quantiles_pos, quantiles_neg = torch.chunk(
            raw_quantiles, 2, dim=0
        )
        quantiles = 0.5 * (quantiles_pos + quantiles_neg)
        return quantiles
