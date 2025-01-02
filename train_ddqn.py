import datetime
from pathlib import Path

from agents.doubledqn_agent import Mario
from env import build_env
from utils.logger import MetricLogger

env = build_env()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)

mario = Mario(
    state_dim=(4, 21, 21),
    action_dim=env.action_space.n,
    save_dir=save_dir,
)

logger = MetricLogger(save_dir)

episodes = 40000

# Optional: Load previous buffer if it exists
buffer_path = Path("checkpoints/replay_buffer.h5")  # Adjust path as needed
if buffer_path.exists():
    mario.load_buffer(buffer_path)
    print(f"Loaded existing replay buffer from {buffer_path}")
else:
    print("No existing replay buffer found. Starting fresh.")

for e in range(episodes):
    state = env.reset()
    while True:
        action = mario.act(state)
        next_state, reward, done, info = env.step(action)
        mario.cache(state, next_state, action, reward, done)
        q, loss = mario.learn()
        logger.log_step(reward, loss, q)
        state = next_state
        if done or info["flag_get"]:
            break

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)