import numpy as np
from collections import deque
import random
import torch

# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)
    
#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
    
#     def sample(self, batch_size):
#         state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
#         return (np.array(state), np.array(action), np.array(reward), 
#                 np.array(next_state), np.array(done))
    
#     def __len__(self):
#         return len(self.buffer)

class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
            
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, alpha=0.7, beta=0.4):
        if len(self.buffer) < batch_size:
            return None
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done = zip(*samples)
        
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done), 
                indices, np.array(weights))
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # small constant to prevent 0 priority
    
    def __len__(self):
        return len(self.buffer)
