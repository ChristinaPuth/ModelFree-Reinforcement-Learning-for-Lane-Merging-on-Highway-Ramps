import optuna
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env
import highway_env.envs.common.abstract
highway_env.envs.common.abstract.AbstractEnv._automatic_rendering = lambda self: None

from merge_env1 import MergeEnv1
from gymnasium.envs.registration import register
import os
import time
import csv
from tqdm import tqdm

# Register the custom environment
register(
    id="merge-v1",
    entry_point="merge_env1:MergeEnv1",
)

# Log file to save each trial result
LOG_CSV_PATH = "dqn_optuna/trial_results_fastcar.csv"
# LOG_CSV_PATH = "/home/st2080/Documents/EEC_256_Final/dqn_optuna/trial_results_bigcar.csv"



#  Objective function for hyperparameter optimization 
def optimize_dqn(trial):
    env = gym.make("merge-v1", render_mode="rgb_array")

    # Hyperparameter search space
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    buffer_size = trial.suggest_int("buffer_size", 10000, 50000)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.7, 0.99)
    train_freq = trial.suggest_int("train_freq", 1, 5)
    target_update_interval = trial.suggest_int("target_update_interval", 50, 500)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        target_update_interval=target_update_interval,
        verbose=0,
    )

    model.learn(total_timesteps=10000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=3, return_episode_rewards=False)
    env.close()

    # Log trial result
    with open(LOG_CSV_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            trial.number,
            mean_reward,
            learning_rate,
            buffer_size,
            batch_size,
            gamma,
            train_freq,
            target_update_interval
        ])

    return mean_reward

# Run the optimization 
if __name__ == "__main__":
    start_time = time.time()
    n_trials = 20
    os.makedirs("dqn_optuna", exist_ok=True)

    # Create CSV file with header
    with open(LOG_CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "reward", "learning_rate", "buffer_size", "batch_size", "gamma", "train_freq", "target_update_interval"])

    with tqdm(total=n_trials, desc="Optimization Progress") as pbar:
        def callback(study, trial):
            pbar.update(1)

        study = optuna.create_study(direction="maximize")
        study.optimize(optimize_dqn, n_trials=n_trials, callbacks=[callback])

    elapsed_time = time.time() - start_time
    print("Best hyperparameters:", study.best_params)
    print("learning_rate:", study.best_params["learning_rate"])
    print("buffer_size:", study.best_params["buffer_size"])
    print("batch_size:", study.best_params["batch_size"])
    print("gamma:", study.best_params["gamma"])
    print("train_freq:", study.best_params["train_freq"])
    print("target_update_interval:", study.best_params["target_update_interval"])
    print(f"Total optimization time: {elapsed_time:.2f} seconds")

  