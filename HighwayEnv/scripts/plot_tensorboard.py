


from tbparse import SummaryReader
import matplotlib.pyplot as plt

logdirs = {
    "DQN": "/home/st2080/Documents/EEC_256_Final/highway_dqn/Fined_para_normalCar_1e5/log/DQN_1",
    "A2C": "/home/st2080/Documents/EEC_256_Final/highway_a2c/Fined_para_normalCar_1e5/log/A2C_1",
    "PPO": "/home/st2080/Documents/EEC_256_Final/highway_ppo/Fined_para_normalCar_1e5/log/PPO_1",
}


tag = "rollout/ep_rew_mean"
label = "Average Reward"

plt.figure(figsize=(8, 5))

for model_name, logdir in logdirs.items():
    reader = SummaryReader(logdir)
    df = reader.scalars
    tag_data = df[df["tag"] == tag]
    plt.plot(tag_data["step"], tag_data["value"], label=model_name)

plt.title("Post-Tuned Models on Normal Car Env (1e5 steps)")
plt.xlabel("Step")
plt.ylabel(label)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("/home/st2080/Documents/EEC_256_Final/HighwayEnv/results/Post-Tuned Models on Normal Car Env (1e5 steps).png")
# plt.show()
