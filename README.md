# Highway Merging Reinforcement Learning System 

This project's goal is to construct a reinforcement learning system for **highway merging**, using a custom-built environment, and compare the performance of three popular RL algorithms: **A2C**, **DQN**, and **PPO**.

Built on top of [highway-env](https://github.com/eleurent/highway-env), this project implements a new merging environment and supports model training, hyperparameter tuning, evaluation, and visualization. It is suitable for academic research and algorithm analysis in autonomous driving scenarios.

---

##  Project Structure

```text
EEC_256_FINAL/
│
├── a2c_optuna/               # A2C tuning results (Optuna CSVs and models)
├── dqn_optuna/               # DQN tuning results
├── ppo_optuna/               # PPO tuning results
│
├── highway_a2c/              # A2C saved model directory
├── highway_dqn/              # DQN saved model directory
├── highway_ppo/              # PPO saved model directory
│
├── HighwayEnv/               # Custom environment base (modified from highway-env)
│   ├── vehicle/
│   │   └── kinematics.py     # Custom vehicle behavior
│
├── results/                  # Visualization outputs and evaluation results
│
├── scripts/                  # Main execution scripts
│   ├── merge_env1.py              # Custom merging environment definition
│   ├── sb3_highway_dqn.py         # DQN training script
│   ├── sb3_highway_ppo.py         # PPO training script
│   ├── sb3_highway_a2c.py         # A2C training script
│   ├── dqn_fined_with_rewards.py  # Tuned DQN training with reward structure
│   ├── a2c_fined_parameter.py     # Tuned A2C training script
│   ├── ppo_find_parameter.py      # PPO tuning script using Optuna
│   ├── count_crash.py             # Crash statistics analysis
│   ├── plot_tensorboard.py        # TensorBoard reward curve visualization
│   └── main.py                    # Unified runner for all 3 models
│
├── README.md                 # This documentation file
├── requirements.txt          # Dependency list
├── .gitignore / LICENSE / CFF and metadata


```
## Custom Environment: `merge-v1`

This project features a custom highway merging environment named `merge-v1`, built as an extension of the [highway-env](https://github.com/eleurent/highway-env) framework. It simulates a realistic traffic merging scenario, designed to evaluate the performance of reinforcement learning agents in multi-agent traffic settings.

### Environment Overview

- **Main highway segment**: a → b → c → d (multi-lane)
- **Curved on-ramp merging path**: j → k → b → c (sine-shaped trajectory)
- **Mixed traffic support**: includes both Controlled Autonomous Vehicles (CAVs) and Human-Driven Vehicles (HDVs)
- **Obstacle-based merging constraint**: forces vehicles to complete merging before a set spatial deadline

The environment is registered as:
```python
register(id="merge-v1", entry_point="merge_env1:MergeEnv1")
```
##  Supported Algorithms

This project integrates the following reinforcement learning algorithms using the [Stable-Baselines3 (SB3)](https://github.com/DLR-RM/stable-baselines3) library:

| Algorithm | Description |
|----------|-------------|
| **A2C**   | A lightweight synchronous actor-critic method, suitable for small to medium-scale environments |
| **DQN**   | A value-based, off-policy algorithm designed for discrete action spaces |
| **PPO**   | A robust and stable on-policy algorithm, well-suited for complex decision-making tasks like highway merging |

All models are enhanced with **Optuna** for automatic hyperparameter tuning. The framework performs systematic searches over key parameters such as:
- Learning rate (`learning_rate`)
- Discount factor (`gamma`)
- Number of steps (`n_steps`)
- Batch size (`batch_size`)
- Number of epochs (`n_epochs`) for PPO

---

## Results and Visualization

- **Training metrics** are logged using **TensorBoard**
- **Reward trends**, exploration rate, training loss, and learning rate dynamics are recorded
- **tbparse** is used for post-processing and extracting scalars
- All reward curves are plotted and saved in the `results/` directory, supporting side-by-side comparisons across DQN, A2C, and PPO

Example output path:
```text
results/ppo_vs_dqn_vs_a2c_rewards.png
```
##  Quick Start (DQN Example)

This section provides a step-by-step guide for training and evaluating the DQN model in the custom highway merging environment.

### 1. Install Dependencies

Ensure all required Python libraries are installed:

```bash
pip install -r requirements.txt
```
#### Hyperparameter Optimization with Optuna
Automatically search for optimal DQN hyperparameters (e.g., learning rate, gamma, n_steps):
```bash
python3 scripts/dqn_find_parameter.py
```
run time: 1e4 steps should run about 50min each method
 #### Visualize Training Rewards
 Parse TensorBoard logs and generate average reward curves for model performance comparison:
```bash
python3 scripts/plot_tensorboard.py
```

#### Run All Three Models (DQN, A2C, PPO)
 Evaluate all supported reinforcement learning algorithms in parallel:
```bash
python3 scripts/main.py
```
runtime: 1e6 step  should run about 8hours totally(3 method) 
#### Important: Before executing the scripts, ensure all file paths and configuration parameters are correctly set based on your local environment structure and experiment setup.