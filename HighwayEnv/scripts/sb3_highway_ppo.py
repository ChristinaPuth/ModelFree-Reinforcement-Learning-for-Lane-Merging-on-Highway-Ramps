


def train_ppo(total_timesteps=1e5, model_path="highway_ppo/Fined_para_fastCar_1e6_Model", log_dir="highway_ppo", n_cpu=6):
    import gymnasium as gym
    from stable_baselines3 import PPO
    from gymnasium.wrappers import RecordVideo
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    import highway_env  # noqa: F401
    import highway_env.envs.common.abstract
    highway_env.envs.common.abstract.AbstractEnv._automatic_rendering = lambda self: None

    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "."))  # 添加当前脚本目录到路径

    def make_merge_env():
        from merge_env1 import MergeEnv1
        from gymnasium.envs.registration import register
        try:
            register(id="merge-v1", entry_point="merge_env1:MergeEnv1")
        except Exception:
            pass
        return gym.make("merge-v1")


    batch_size = 256
    env = make_vec_env(make_merge_env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=151,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=0.00021783737033594376,
        gamma= 0.7867135213898531,
        verbose=2,
        tensorboard_log=f"{log_dir}/log"
    )
    model.learn(total_timesteps=int(total_timesteps))
    model.save(model_path)

    from merge_env1 import MergeEnv1
    from gymnasium.envs.registration import register
    try:
        register(id="merge-v1", entry_point="merge_env1:MergeEnv1")
    except Exception:
        pass
    model = PPO.load(model_path)
    env = gym.make("merge-v1", render_mode="rgb_array")

    for _ in range(5):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()

    env = RecordVideo(
        env, video_folder=f"{log_dir}/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.config["simulation_frequency"] = 15
    env.unwrapped.set_record_video_wrapper(env)

    for _ in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    env.close()
