import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from models.dqn_dueling_model import DuelingDQNNetwork  # Sửa import để sử dụng đúng model
from utils.replay_buffer import ReplayBuffer  # Dùng ReplayBuffer thay vì PrioritizedReplayBuffer
import torch.optim as optim

class DuelingDQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.gamma = 0.99
        self.batch_size = 128
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.use_double_dqn = True

        self.memory = ReplayBuffer(100000)  # Sử dụng ReplayBuffer
        self.model = DuelingDQNNetwork(state_dim, action_dim).to(device)
        self.target_model = DuelingDQNNetwork(state_dim, action_dim).to(device)
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

        if self.steps % 1000 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # ReplayBuffer trả về 5 giá trị
        samples = self.memory.sample(self.batch_size)
        if samples is None:
            return None

        states, actions, rewards, next_states, dones = samples

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        if self.use_double_dqn:
            with torch.no_grad():
                next_actions = self.model(next_states).argmax(1)
                next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        else:
            with torch.no_grad():
                next_q_values = self.target_model(next_states).max(1)[0]

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        losses = F.smooth_l1_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return losses.item()


    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save(self, path):
        torch.save({'model_state_dict': self.model.state_dict(),
                    'target_model_state_dict': self.target_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epsilon': self.epsilon}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint.get('target_model_state_dict', self.model.state_dict()))
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
