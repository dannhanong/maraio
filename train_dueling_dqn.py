from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from utils.wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from agents.dueling_dqn_agent import DuelingDQNAgent
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from pathlib import Path

# Khởi tạo environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)

# Khởi tạo agent và paths
state_dim = (1, 84, 84)
action_dim = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
agent = DuelingDQNAgent(state_dim, action_dim, device)

# Tạo thư mục với timestamp
checkpoint_dir = Path('checkpoints/dueling_dqn')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Training parameters
episodes = 1000
save_interval = 10
eval_interval = 10
starting_episode = 0
best_reward = float('-inf')

# Load best model và xác định episode bắt đầu
checkpoint_path = checkpoint_dir / 'best_model.pth'
if checkpoint_path.exists():
    print(f"Loading best model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    agent.load(checkpoint_path)
    print(f"Loaded model with epsilon: {agent.epsilon:.3f}")
    best_reward = checkpoint.get('best_reward', float('-inf'))

    # Tìm checkpoint episode gần nhất
    checkpoints = list(checkpoint_dir.glob('checkpoint_episode_*.pth'))
    if checkpoints:
        latest_episode = max([int(x.stem.split('_')[-1]) for x in checkpoints])
        starting_episode = latest_episode + 1

    print(f"Continuing training from step: {agent.steps}")
    print(f"Continuing from episode: {starting_episode}")
    print(f"Best reward so far: {best_reward:.2f}")
else:
    print("No checkpoint found. Starting fresh training.")

# Khởi tạo writer với tên có timestamp
current_time = time.strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/mario_dueling_dqn_{current_time}')

# Tracking metrics
episode_data = {
    'rewards': [],
    'positions': [],
    'losses': []
}

print(f"Training on device: {device}")
print(f"Training from episode {starting_episode} to {starting_episode + episodes}")
print("-" * 50)

def calculate_reward(info, reward, prev_info):
    total_reward = reward
    current_x = info.get('x_pos', 0)
    prev_x = prev_info.get('x_pos', current_x)
    position_delta = current_x - prev_x

    # Reward cho việc di chuyển
    if position_delta > 0:
        total_reward += position_delta * 0.1  # Giảm reward cho việc di chuyển
    else:
        total_reward -= 1.0  # Tăng penalty cho việc đứng yên/lùi

    # Reward cho việc sống sót
    total_reward += 0.1

    # Penalties
    if info.get('life', 2) < prev_info.get('life', 2):
        total_reward -= 100  # Tăng penalty cho việc chết

    # Rewards
    if info.get('flag_get', False):
        total_reward += 500

    return total_reward

def save_checkpoint(episode, reward, max_position):
    """Lưu checkpoint với thông tin training"""
    global best_reward

    checkpoint = {
        'model_state_dict': agent.model.state_dict(),
        'target_model_state_dict': agent.target_model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': float(agent.epsilon),
        'epsilon_min': float(agent.epsilon_min),
        'epsilon_decay': float(agent.epsilon_decay),
        'steps': int(agent.steps),
        'episode': int(episode),
        'best_reward': float(best_reward),
        'max_position': int(max_position)
    }

    # Lưu checkpoint định kỳ
    torch.save(checkpoint, checkpoint_dir / f'checkpoint_episode_{episode}.pth')

    # Lưu best model nếu đạt reward cao hơn
    if reward > best_reward:
        best_reward = reward
        torch.save(checkpoint, checkpoint_dir / 'best_model.pth')

# Training loop
try:
    for episode in range(starting_episode, starting_episode + episodes):
        state = env.reset()
        state = np.expand_dims(state, axis=0)
        done = False
        total_reward = 0
        steps = 0
        episode_loss = []
        max_position = 0  # Track max position reached

        prev_info = {}

        while not done:
            # Get action
            action = agent.act(state)

            # Take step
            next_state, reward, done, info = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)

            # Track position
            current_position = info.get('x_pos', 0)
            max_position = max(max_position, current_position)

            # Modify reward to encourage forward progress
            reward = calculate_reward(info, reward, prev_info)
            prev_info = info.copy()

            # Clip reward cho training
            clipped_reward = np.clip(reward, -15, 15)

            # Store transition
            agent.memory.push(state, action, clipped_reward, next_state, done)

            # Train agent
            loss = agent.train()
            if loss is not None:
                episode_loss.append(loss)

            state = next_state
            total_reward += reward  # Use unclipped reward for tracking
            steps += 1

        # Calculate metrics
        avg_loss = np.mean(episode_loss) if episode_loss else 0

        # Logging
        writer.add_scalar('Training/Episode Reward', total_reward, episode)
        writer.add_scalar('Training/Episode Length', steps, episode)
        writer.add_scalar('Training/Average Loss', avg_loss, episode)
        writer.add_scalar('Training/Epsilon', agent.epsilon, episode)
        writer.add_scalar('Training/Max Position', max_position, episode)

        # Save periodic checkpoint
        if episode % save_interval == 0:
            save_checkpoint(episode, total_reward, max_position)

        print(f"\nEpisode {episode}/{starting_episode + episodes - 1}")
        print(f"Reward: {total_reward:.2f}")
        print(f"Max Position: {max_position}")
        print(f"Steps: {steps}")
        print(f"Epsilon: {agent.epsilon:.3f}")
        print(f"Average Loss: {avg_loss:.5f}")
        print(f"Best Reward So Far: {best_reward:.2f}")
        print("-" * 50)

        # Update epsilon
        agent.update_epsilon()

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    print("Saving final model...")
    agent.save(checkpoint_dir / 'interrupted_model.pth')
    print("Model saved!")

finally:
    env.close()
    writer.close()

    # Save final model
    print("\nSaving final model...")
    save_checkpoint(episode, total_reward, max_position)

    print("\nTraining summary:")
    print(f"Episodes completed: {episode - starting_episode + 1}")
    print(f"Best reward achieved: {best_reward:.2f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Total steps trained: {agent.steps}")
    print(f"Max position reached: {max_position}")
