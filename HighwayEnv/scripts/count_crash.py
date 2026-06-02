import gym
from stable_baselines3 import DQN

# Load the pre-trained model
model = DQN.load("new1201_dqn/model")

# Initialize the environment
env = gym.make('merge-v0', render_mode='rgb_array')

# Set test to True to run this block
test = True
if test:
    crash_count = 0
    num_episodes = 10

    for _ in range(num_episodes):
        done = truncated = False
        obs = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()

            # Check for crashes
            if 'crash' in info and info['crash']:
                crash_count += 1

    env.close()

    print(f"Total crashes in {num_episodes} episodes: {crash_count}")
