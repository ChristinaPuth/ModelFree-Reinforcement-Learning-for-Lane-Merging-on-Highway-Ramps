# import optuna
# from stable_baselines3 import A2C
# from stable_baselines3.common.evaluation import evaluate_policy
# import gymnasium as gym
# from gymnasium.wrappers import RecordVideo
# import highway_env
# import highway_env.envs.common.abstract
# highway_env.envs.common.abstract.AbstractEnv._automatic_rendering = lambda self: None

# from merge_env1 import MergeEnv1
# from gymnasium.envs.registration import register
# import os
# import time
# from tqdm import tqdm

# # 注册自定义环境
# register(
#     id="merge-v1",
#     entry_point="merge_env1:MergeEnv1",
# )

# # === 目标函数 ===
# def optimize_a2c(trial):
#     env = gym.make("merge-v1", render_mode="rgb_array")

#     # 超参数搜索空间
#     learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
#     n_steps = trial.suggest_int("n_steps", 5, 20)
#     gamma = trial.suggest_float("gamma", 0.90, 0.999)
#     ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.05)

#     model = A2C(
#         "MlpPolicy",
#         env,
#         learning_rate=learning_rate,
#         n_steps=n_steps,
#         gamma=gamma,
#         ent_coef=ent_coef,
#         verbose=0,
#     )

#     model.learn(total_timesteps=10000)
#     mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=3, return_episode_rewards=False)
#     env.close()
#     return mean_reward

# # === 执行优化 ===
# if __name__ == "__main__":
#     start_time = time.time()
#     n_trials = 20
    
#     # 使用 tqdm 包裹优化进度
#     with tqdm(total=n_trials, desc="progress") as pbar:
#         def callback(study, trial):
#             pbar.update(1)

#         study = optuna.create_study(direction="maximize")
#         study.optimize(optimize_a2c, n_trials=n_trials, callbacks=[callback])

#     elapsed_time = time.time() - start_time
#     print("best parameter：", study.best_params)
#     print("learning rate：", study.best_params["learning_rate"])
#     print("n_steps：", study.best_params["n_steps"])
#     print("gamma：", study.best_params["gamma"])
#     print("ent_coef：", study.best_params["ent_coef"])
#     print(f"time：{elapsed_time:.2f} s")

#     # 使用最优参数重新训练并保存
#     env = gym.make("merge-v1", render_mode="rgb_array")
#     best_params = study.best_params

#     model = A2C(
#         "MlpPolicy",
#         env,
#         learning_rate=best_params["learning_rate"],
#         n_steps=best_params["n_steps"],
#         gamma=best_params["gamma"],
#         ent_coef=best_params["ent_coef"],
#         verbose=1,
#     )
#     model.learn(total_timesteps=20000)
#     os.makedirs("a2c_optuna_model", exist_ok=True)
#     model.save("a2c_optuna_model/best_model")
#     env.close()



import optuna
from stable_baselines3 import A2C
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

LOG_CSV_PATH = "a2c_optuna/trial_results_fastcar.csv"

# Objective function for hyperparameter optimization 
def optimize_a2c(trial):
    env = gym.make("merge-v1", render_mode="rgb_array")

    # Hyperparameter search space
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    n_steps = trial.suggest_int("n_steps", 5, 20)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.05)

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=1.0,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
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
            gamma,
            ent_coef
        ])

    return mean_reward

# Run the optimization 
if __name__ == "__main__":
    start_time = time.time()
    n_trials = 20
    os.makedirs("a2c_optuna", exist_ok=True)

    with open(LOG_CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "reward", "learning_rate", "n_steps", "gamma", "ent_coef"])

    with tqdm(total=n_trials, desc="Optimization Progress") as pbar:
        def callback(study, trial):
            pbar.update(1)

        study = optuna.create_study(direction="maximize")
        study.optimize(optimize_a2c, n_trials=n_trials, callbacks=[callback])

    elapsed_time = time.time() - start_time
    print("Best hyperparameters:", study.best_params)
    print("learning_rate:", study.best_params["learning_rate"])
    print("n_steps:", study.best_params["n_steps"])
    print("gamma:", study.best_params["gamma"])
    print("ent_coef:", study.best_params["ent_coef"])
    print(f"Total optimization time: {elapsed_time:.2f} seconds")
