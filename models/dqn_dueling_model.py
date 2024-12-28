import torch
import torch.nn as nn
import numpy as np  # Thêm dòng này

class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQNNetwork, self).__init__()
        
        # Thay Conv2D bằng Linear layers vì trạng thái là một vector 1D
        self.fc = nn.Sequential(
            nn.Linear(np.prod(input_shape), 512),
            nn.ReLU()
        )

        # Nhánh Value và Advantage
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Chuyển đổi trạng thái thành vector
        x = self.fc(x)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Kết hợp Value và Advantage để tính Q-value
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
