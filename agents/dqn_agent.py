import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from models.dqn_model import DQNModel
# from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from utils.replay_buffer import PrioritizedReplayBuffer

import torch.optim as optim

class DQNNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        # Hàm này tính kích thước output của conv layers
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class DQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Thêm biến use_double_dqn
        self.use_double_dqn = True  # Có thể set True hoặc False tùy ý
        
        # Các hyperparameters khác
        self.gamma = 0.992
        self.batch_size = 128
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.996
        
        self.memory = PrioritizedReplayBuffer(100000)
        
        self.model = DQNNetwork(state_dim, action_dim).to(device)
        self.target_model = DQNNetwork(state_dim, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.steps = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return q_values.argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
            
        # Thêm gradient clipping để tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Thêm target network update frequency
        if self.steps % 1000 == 0:  # Cập nhật target network mỗi 1000 steps
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Sample với priorities
        samples = self.memory.sample(self.batch_size)
        if samples is None:
            return None
            
        states, actions, rewards, next_states, dones, indices, weights = samples
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Double DQN
        if self.use_double_dqn:
            with torch.no_grad():
                next_actions = self.model(next_states).argmax(1)
                next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        else:
            with torch.no_grad():
                next_q_values = self.target_model(next_states).max(1)[0]
        
        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Get current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute TD errors for updating priorities
        td_errors = torch.abs(target_q_values - current_q_values).detach().cpu().numpy()
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors)
        
        # Sửa cách tính loss để tránh broadcasting error
        losses = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
        weighted_loss = (weights * losses).mean()
        
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return weighted_loss.item()

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': float(self.epsilon),
            'steps': int(self.steps),
            'epsilon_min': float(self.epsilon_min),
            'epsilon_decay': float(self.epsilon_decay)
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Xử lý tương thích ngược với checkpoint cũ
        if 'target_model_state_dict' in checkpoint:
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        else:
            # Nếu không có target_model trong checkpoint, copy từ model chính
            self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        
        # Xử lý các thông số epsilon có thể không có trong checkpoint cũ
        self.epsilon_min = checkpoint.get('epsilon_min', 0.1)  # default 0.1
        self.epsilon_decay = checkpoint.get('epsilon_decay', 0.992)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def switch_device(self, new_device):
        """Chuyển model sang device mới"""
        self.device = new_device
        self.model = self.model.to(new_device)
        self.target_model = self.target_model.to(new_device)
        
        # Chuyển optimizer state
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(new_device)
