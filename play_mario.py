from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from utils.wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from agents.dqn_agent import DQNAgent
import torch
import numpy as np
from pathlib import Path
import sys
import time

def calculate_reward(info, reward, prev_info):
    total_reward = reward
    current_x = info.get('x_pos', 0)
    prev_x = prev_info.get('x_pos', current_x)
    position_delta = current_x - prev_x
    
    if position_delta > 0:
        total_reward += position_delta * 0.5
    else:
        total_reward -= 0.2
        
    if info.get('flag_get', False):
        total_reward += 500
        
    if info.get('life', 2) < prev_info.get('life', 2):
        total_reward -= 50
        
    return total_reward

def play_mario(model_path, num_episodes=1):
    try:
        # Khởi tạo environment
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = SkipFrame(env, skip=4)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=84)

        # Khởi tạo agent
        state_dim = (1, 84, 84)
        action_dim = env.action_space.n
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = DQNAgent(state_dim, action_dim, device)
        
        # Load model
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path)
        agent.load(model_path)
        print(f"Model info:")
        print(f"Steps trained: {checkpoint.get('steps', 'N/A')}")
        print(f"Best reward: {checkpoint.get('best_reward', 'N/A')}")
        print(f"Epsilon: {checkpoint.get('epsilon', 'N/A')}")
        
        # Sử dụng epsilon từ checkpoint
        agent.epsilon = checkpoint.get('epsilon', 0.01)
        prev_info = {}
        
        print("\nAvailable actions:", SIMPLE_MOVEMENT)
        print("\nPress Ctrl+C to exit")
        
        for episode in range(num_episodes):
            state = env.reset()
            state = np.expand_dims(state, axis=0)
            done = False
            total_reward = 0
            steps = 0
            
            print(f"\nEpisode {episode + 1}")
            
            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                # Tính toán reward như trong training
                reward = calculate_reward(info, reward, prev_info)
                prev_info = info.copy()
                
                next_state = np.expand_dims(next_state, axis=0)
                state = next_state
                total_reward += reward
                steps += 1
                
                env.render()
                time.sleep(0.01)
                
                if steps % 10 == 0:
                    print(f"\rSteps: {steps} | Reward: {total_reward:.2f} | "
                          f"Position: {info.get('x_pos', 0)} | "
                          f"Lives: {info.get('life', 'N/A')}", end="")
            
            print(f"\nEpisode finished after {steps} steps with reward {total_reward:.2f}")
            time.sleep(1)  # Pause between episodes
            
    except KeyboardInterrupt:
        print("\nStopping game...")
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        env.close()
        print("\nEnvironment closed")

if __name__ == "__main__":
    model_path = "checkpoints/dqn/best_model.pth"
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
    else:
        play_mario(model_path, num_episodes=1)  # Reduced number of episodes
  