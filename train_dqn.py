from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from utils.wrappers import SkipFrame, GrayScaleObservation, ResizeObservation
from agents.dqn_agent import DQNAgent
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
agent = DQNAgent(state_dim, action_dim, device)

# Đường dẫn Replay Buffer
replay_buffer_path = 'replay_buffer.pkl'

# Tải Replay Buffer nếu tồn tại, nếu không khởi tạo replay buffer mới
if Path(replay_buffer_path).exists():
    print(f"Loading Replay Buffer from {replay_buffer_path}")
    try:
        agent.memory.load(replay_buffer_path)
    except Exception as e:
        print(f"Error loading Replay Buffer: {e}")
else:
    print("No existing Replay Buffer found. Starting with an empty buffer.")

# Tạo thư mục với timestamp
checkpoint_dir = Path('checkpoints/dqn')
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

    # Nạp best_reward từ checkpointss
    best_reward = checkpoint.get('best_reward', float('-inf'))
    print(f"Best reward loaded: {best_reward}")

    # Tiếp tục từ episode gần nhất
    checkpoints = list(checkpoint_dir.glob('checkpoint_episode_*.pth'))
    if checkpoints:
        latest_episode = max([int(x.stem.split('_')[-1]) for x in checkpoints])
        starting_episode = latest_episode + 1

    print(f"Continuing from episode: {starting_episode}")
else:
    print("No checkpoint found. Starting fresh training.")
    best_reward = float('-inf')

# Khởi tạo writer với tên có timestamp
current_time = time.strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'runs/mario_dqn_{current_time}')

# Tracking metrics
episode_data = {
    'rewards': [],
    'positions': [],
    'losses': []
}

print(f"Training on device: {device}")
print(f"Training from episode {starting_episode} to {starting_episode + episodes}")
print("-" * 50)

# Thêm tracking để theo dõi sự ổn định
class TrainingStats:
    def __init__(self, window_size=100):
        self.rewards = []
        self.positions = []
        self.window_size = window_size
        
    def add_episode(self, reward, position):
        self.rewards.append(reward)
        self.positions.append(position)
        
    def get_stats(self):
        if len(self.rewards) < self.window_size:
            return None
        
        recent_rewards = self.rewards[-self.window_size:]
        recent_positions = self.positions[-self.window_size:]
        
        return {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'mean_position': np.mean(recent_positions),
            'std_position': np.std(recent_positions)
        }

def evaluate_agent(env, agent, num_episodes=5):
    # Lưu giá trị epsilon hiện tại
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # Tắt exploration
    
    total_rewards = []
    
    for _ in range(num_episodes):
        state = env.reset()
        state = np.expand_dims(state, axis=0)
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            reward = calculate_reward(info, reward, {})
            episode_reward += reward
            state = np.expand_dims(next_state, axis=0)
            
        total_rewards.append(episode_reward)
    
    # Khôi phục giá trị epsilon ban đầu
    agent.epsilon = original_epsilon
    return np.mean(total_rewards), np.std(total_rewards)

def calculate_reward(info, reward, prev_info):
    total_reward = reward
    current_x = info.get('x_pos', 0)
    prev_x = prev_info.get('x_pos', current_x)
    position_delta = current_x - prev_x
    
    # Reward cho việc di chuyển
    if position_delta > 0:
        total_reward += position_delta * 0.1  # Giảm reward cho việc di chuyển
    else:
        total_reward -= 2.0  # Tăng penalty cho việc đứng yên/lùi
    
    # Reward cho việc sống sót
    total_reward += 0.1
    
    # Penalties
    if info.get('life', 2) < prev_info.get('life', 2):
        total_reward -= 100  # Tăng penalty cho việc chết
    
    # Rewards
    if info.get('flag_get', False):
        total_reward += 1000
    
    return total_reward

def save_checkpoint(episode, reward, max_position):
    """Lưu checkpoint với thông tin training"""
    global best_reward  # Biến toàn cục lưu best_reward
    
    checkpoint = {
        'model_state_dict': agent.model.state_dict(),
        'target_model_state_dict': agent.target_model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': float(agent.epsilon),  # Giá trị epsilon
        'epsilon_min': float(agent.epsilon_min),
        'epsilon_decay': float(agent.epsilon_decay),
        'steps': int(agent.steps),
        'episode': int(episode),
        'best_reward': float(best_reward),  # Thêm best_reward vào checkpoint
        'max_position': int(max_position)
    }
    
    # Lưu checkpoint định kỳ
    torch.save(checkpoint, checkpoint_dir / f'checkpoint_episode_{episode}.pth')
    
    # Lưu best model nếu đạt reward cao hơn
    if reward >= best_reward:
        best_reward = reward
        torch.save(checkpoint, checkpoint_dir / 'best_model.pth')

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    agent.device = device
    agent.model = agent.model.to(device)
    agent.target_model = agent.target_model.to(device)
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.target_model.load_state_dict(agent.model.state_dict())
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Chuyển optimizer state sang device mới
    for state in agent.optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
                
    agent.epsilon = checkpoint['epsilon']
    agent.steps = checkpoint['steps']
    best_reward = checkpoint.get('best_reward', float('-inf'))  # Nạp best_reward từ checkpoint
    return checkpoint

def cleanup_old_checkpoints(checkpoint_dir, keep_n=5):
    """Chỉ giữ lại n checkpoints gần nhất"""
    checkpoints = list(checkpoint_dir.glob('checkpoint_episode_*.pth'))
    if len(checkpoints) > keep_n:
        # Sắp xếp theo thời gian tạo
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        # Xóa các checkpoint cũ
        for checkpoint in checkpoints[:-keep_n]:
            checkpoint.unlink()

# Training loop
try:
    validation_rewards = []
    patience = 20  # Số episodes chờ cải thiện
    best_val_reward = float('-inf')
    no_improvement = 0
    
    for episode in range(starting_episode, starting_episode + episodes):
        start_time = time.time()
        state = env.reset()
        state = np.expand_dims(state, axis=0)
        done = False
        total_reward = 0
        steps = 0
        episode_loss = []
        max_position = 0  # Track max position reached
        
        stats_tracker = TrainingStats()
        
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
            
            # Render mỗi 100 episodes
            if episode % 100 == 0:
                env.render()
            
            stats_tracker.add_episode(total_reward, max_position)
        
        # Calculate metrics
        episode_time = time.time() - start_time
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
            agent.memory.save(replay_buffer_path)
        
        # Save best model
        if (total_reward > best_reward or
            (total_reward == best_reward and max_position > best_position) or
            (total_reward == best_reward and max_position == best_position and agent.epsilon <= best_epsilon)):
            best_reward = total_reward
            best_position = max_position
            best_epsilon = agent.epsilon
            save_checkpoint(episode, total_reward, max_position)
            print(f"\nNew best model saved with:")
            if total_reward > best_reward:
                print("New best model due to higher reward.")
            elif total_reward == best_reward and max_position > best_position:
                print("New best model due to better position.")
            elif total_reward == best_reward and max_position == best_position and agent.epsilon <= best_epsilon:
                print("New best model due to lower epsilon.")
        
        # Print progress
        print(f"\nEpisode {episode}/{starting_episode + episodes - 1}")
        print(f"Reward: {total_reward:.2f}")
        print(f"Max Position: {max_position}")
        print(f"Steps: {steps}")
        print(f"Epsilon: {agent.epsilon:.3f}")
        print(f"Average Loss: {avg_loss:.5f}")
        print(f"Time: {episode_time:.2f}s")
        print(f"Best Reward So Far: {best_reward:.2f}")
        print("-" * 50)
        
        # Early stopping với điều kiện mới
        if total_reward >= 6000 and max_position > 1200:  # Phải đạt cả reward và position
            print("Reached target reward AND position! Training completed.")
            break
        
        stats = stats_tracker.get_stats()
        
        if stats:
            writer.add_scalar('Stability/Reward_StdDev', stats['std_reward'], episode)
            writer.add_scalar('Stability/Position_StdDev', stats['std_position'], episode)
        
        if episode % eval_interval == 0:
            mean_reward, std_reward = evaluate_agent(env, agent)
            writer.add_scalar('Evaluation/Mean_Reward', mean_reward, episode)
            writer.add_scalar('Evaluation/Reward_StdDev', std_reward, episode)
        
        if episode % 10 == 0:
            print("\nDebug Information:")
            print(f"Memory Buffer Size: {len(agent.memory)}")
            
            # Tính toán statistics cho rewards và positions
            recent_rewards = episode_data['rewards'][-100:] if episode_data['rewards'] else [0]
            recent_positions = episode_data['positions'][-100:] if episode_data['positions'] else [0]
            
            print(f"Recent Rewards: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
            print(f"Recent Positions: {np.mean(recent_positions):.2f} ± {np.std(recent_positions):.2f}")
            print(f"Learning Rate: {agent.optimizer.param_groups[0]['lr']}")
        
        # Track episode data
        episode_data['rewards'].append(total_reward)
        episode_data['positions'].append(max_position)
        episode_data['losses'].append(np.mean(episode_loss))
        
        # Compute rolling statistics
        window_size = 100
        if len(episode_data['rewards']) >= window_size:
            recent_rewards = episode_data['rewards'][-window_size:]
            recent_positions = episode_data['positions'][-window_size:]
            
            reward_mean = np.mean(recent_rewards)
            reward_std = np.std(recent_rewards)
            position_mean = np.mean(recent_positions)
            
            print(f"\nRolling Statistics (last {window_size} episodes):")
            print(f"Average Reward: {reward_mean:.2f} ± {reward_std:.2f}")
            print(f"Average Position: {position_mean:.2f}")
            
            # Early stopping based on stability
            if reward_std > reward_mean * 0.5:  # Nếu std > 50% mean
                print("Warning: High variance in rewards!")
            
        # Adaptive learning rate
        if episode % 50 == 0:
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] *= 0.95
        
        # Validation mỗi 10 episodes
        if episode % 10 == 0:
            mean_reward, std_reward = evaluate_agent(env, agent)
            validation_rewards.append(mean_reward)
            
            print(f"\nEvaluation: Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
            
            # Early stopping - So sánh mean_reward thay vì tuple
            if mean_reward > best_val_reward:
                best_val_reward = mean_reward
                no_improvement = 0
                # Save best model
                save_checkpoint(episode, mean_reward, max_position)
            else:
                no_improvement += 1
                
            if no_improvement >= patience:
                print(f"No improvement for {patience} validations. Stopping training.")
                break
        
        # Update epsilon once per episode
        agent.update_epsilon()
        
        # Print episode information
        print(f"Episode {episode}/{episodes}")
        print(f"Epsilon: {agent.epsilon:.3f}")
    
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    print("Saving final model...")
    agent.save(checkpoint_dir / 'interrupted_model.pth')
    agent.memory.save(replay_buffer_path)
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
