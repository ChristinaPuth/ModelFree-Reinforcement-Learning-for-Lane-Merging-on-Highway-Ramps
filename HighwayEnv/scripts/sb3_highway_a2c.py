


def train_a2c(total_timesteps=1e4, model_path="highway_a2c/Fined_para_fastCar_1e6_Model", log_dir="highway_a2c"):
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo
    from stable_baselines3 import A2C
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

  
    model = A2C(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=6.885998346583947e-05,
        n_steps=9,
        gamma= 0.8437218350588245,
        gae_lambda=1.0,
        ent_coef= 0.029656313979457935,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=f"{log_dir}/log",
        verbose=1,
    )


    model.learn(total_timesteps=int(total_timesteps))
    model.save(model_path)
    del model

    model = A2C.load(model_path, env=env)
    env = RecordVideo(
        env, video_folder=f"{log_dir}/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.config["simulation_frequency"] = 15
    env.unwrapped.set_record_video_wrapper(env)

    for _ in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    env.close()
