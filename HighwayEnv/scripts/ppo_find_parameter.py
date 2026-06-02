import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from gymnasium.envs.registration import register
import highway_env
import highway_env.envs.common.abstract
highway_env.envs.common.abstract.AbstractEnv._automatic_rendering = lambda self: None
from merge_env1 import MergeEnv1
import os
import time
import csv
from tqdm import tqdm

# Register custom environment
register(
    id="merge-v1",
    entry_point="merge_env1:MergeEnv1",
)

LOG_CSV_PATH = "ppo_optuna/trial_results_fastcar.csv"

# Objective function for hyperparameter optimization 
def optimize_ppo(trial):
    env = gym.make("merge-v1", render_mode="rgb_array")

    # Hyperparameter search space
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    n_steps = trial.suggest_int("n_steps", 64, 2048)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.7, 0.99)
    n_epochs = trial.suggest_int("n_epochs", 5, 20)

    model = PPO(
        "MlpPolicy",
        env,
        
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        n_epochs=n_epochs,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        verbose=0,
    )

    model.learn(total_timesteps=10000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=3, return_episode_rewards=False)
    env.close()

    with open(LOG_CSV_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            trial.number,
            mean_reward,
            learning_rate,
            n_steps,
            batch_size,
            gamma,
            n_epochs
        ])

    return mean_reward

#  Run the optimization 
if __name__ == "__main__":
    start_time = time.time()
    n_trials = 20
    os.makedirs("ppo_optuna", exist_ok=True)

    with open(LOG_CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "reward", "learning_rate", "n_steps", "batch_size", "gamma", "n_epochs"])

    with tqdm(total=n_trials, desc="Optimization Progress") as pbar:
        def callback(study, trial):
            pbar.update(1)

        study = optuna.create_study(direction="maximize")
        study.optimize(optimize_ppo, n_trials=n_trials, callbacks=[callback])

    elapsed_time = time.time() - start_time
    print("Best hyperparameters:", study.best_params)
    print("learning_rate:", study.best_params["learning_rate"])
    print("n_steps:", study.best_params["n_steps"])
    print("batch_size:", study.best_params["batch_size"])
    print("gamma:", study.best_params["gamma"])
    print("n_epochs:", study.best_params["n_epochs"])
    print(f"Total optimization time: {elapsed_time:.2f} seconds")

