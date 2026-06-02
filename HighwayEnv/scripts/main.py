from sb3_highway_a2c import train_a2c
from sb3_highway_dqn import train_dqn
from sb3_highway_ppo import train_ppo
if __name__ == "__main__":
    print("Running A2C...")
    train_a2c(
    total_timesteps=1e6,
    log_dir="highway_a2c/Fined_para_fastCar_1e6"
)

    print("A2C Finished.")

    print("Running DQN...")
    train_dqn(
    total_timesteps=1e6,
 
    log_dir="highway_dqn/Fined_para_fastCar_1e6"
)
    print("dqn Finished.")

    print("Running PPO...")
    train_ppo(
    total_timesteps=1e6,

    log_dir="highway_ppo/Fined_para_fastCar_1e6"
)
    print("PPO Finished.")


