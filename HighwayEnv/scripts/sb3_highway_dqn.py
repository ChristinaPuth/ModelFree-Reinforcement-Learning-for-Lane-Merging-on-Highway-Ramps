



def train_dqn(total_timesteps=1e5, model_path="highway_dqn/Fined_para_fastCar_1e6_Model", log_dir="highway_dqn"):
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo
    from stable_baselines3 import DQN
    import highway_env  # noqa: F401
    import highway_env.envs.common.abstract
    highway_env.envs.common.abstract.AbstractEnv._automatic_rendering = lambda self: None

    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "."))

    from merge_env1 import MergeEnv1
    from gymnasium.envs.registration import register

    try:
        register(id="merge-v1", entry_point="merge_env1:MergeEnv1")
    except Exception:
        pass


    env = gym.make("merge-v1", render_mode="rgb_array")
    obs, info = env.reset()

   
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=0.00017212185153428082,
        buffer_size=19217,
        learning_starts=200,
        batch_size=64,
        gamma=0.7584743879841858,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=168,
        verbose=1,
        tensorboard_log=f"{log_dir}/log",
    )

   
    model.learn(total_timesteps=int(total_timesteps))
    model.save(model_path)
    del model

    
    model = DQN.load(model_path, env=env)
    env = RecordVideo(
        env, video_folder=f"{log_dir}/Videos", episode_trigger=lambda e: True
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
