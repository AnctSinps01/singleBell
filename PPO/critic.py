import torch
import torch.nn as nn

class CriticNetwork(nn.Module):
    def __init__(self, history_length=4, n_poles=2):
        super(CriticNetwork, self).__init__()
        state_dim = 1 + n_poles
        input_dim = history_length * state_dim
        
        # 特征提取层
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
        )
        # 价值输出层
        self.value_head = nn.Linear(32, 1)

        # ================= 显式初始化权重 =================
        self._init_weights()

    def _init_weights(self):
        # 1. 隐藏层：正交初始化
        gain = nn.init.calculate_gain('leaky_relu')
        for m in self.feature_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.constant_(m.bias, 0.0)
                
        # 2. V(s) 价值输出层：增益设为 1.0
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, state_history):
        x = state_history.view(state_history.size(0), -1)

        x_concat = torch.cat([x, -x], dim=0)
        features = self.feature_net(x_concat)
        raw_v = self.value_head(features)

        v_pos, v_neg = torch.chunk(raw_v, 2, dim=0)
        v = 0.5 * (v_pos + v_neg)
        
        return v