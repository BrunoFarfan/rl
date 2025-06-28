from pathlib import Path

import gymnasium as gym
from plotting.csv_reader import load_lengths_from_csv_dir
from plotting.plotter import plot_learning_curves
from reward_wrapper import TraumaWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm

SAVE_DIR = 'data/test_dqn_normal'


def main(n_runs: int = 15, n_timesteps: int = 300_000, trauma_step: int | None = None):
    (Path('Proyecto') / SAVE_DIR).mkdir(parents=True, exist_ok=True)

    for run in tqdm(range(n_runs), desc='DQN runs'):
        env = gym.make('MountainCar-v0')
        if trauma_step is not None:
            env = TraumaWrapper(env, trauma_step=trauma_step)
        env = Monitor(env, f'Proyecto/{SAVE_DIR}/dqn_results_{run + 1}')

        model = DQN(
            'MlpPolicy',
            env,
            learning_rate=0.004,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=128,
            gamma=0.98,
            exploration_fraction=0.2,
            exploration_final_eps=0.07,
            train_freq=16,
            gradient_steps=8,
            target_update_interval=600,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
        )

        model.learn(total_timesteps=n_timesteps, log_interval=10)

        env.close()

    # Save the model
    model.save(f'Proyecto/{SAVE_DIR}/dqn_model')


def hyperparameter_search():
    import itertools

    import numpy as np

    # Define hyperparameter grid
    learning_rates = [1e-4, 5e-4, 1e-3]
    buffer_sizes = [10000, 50000]
    batch_sizes = [32, 64]
    gammas = [0.99, 1.0]
    exploration_fractions = [0.05, 0.1]
    exploration_final_epsilons = [0.01, 0.05]

    param_grid = list(
        itertools.product(
            learning_rates,
            buffer_sizes,
            batch_sizes,
            gammas,
            exploration_fractions,
            exploration_final_epsilons,
        )
    )

    best_score = float('inf')
    best_params = None

    for params in tqdm(param_grid, desc='Hyperparameter search'):
        lr, buf, batch, gamma, expl_frac, expl_final = params
        env = gym.make('MountainCar-v0')
        env = Monitor(env, None)  # No file output for speed
        model = DQN(
            'MlpPolicy',
            env,
            learning_rate=lr,
            buffer_size=buf,
            batch_size=batch,
            gamma=gamma,
            exploration_fraction=expl_frac,
            exploration_final_eps=expl_final,
            verbose=0,
        )
        # Short run for speed
        model.learn(total_timesteps=10_000, log_interval=10)
        # Evaluate
        episode_lengths = env.get_episode_lengths()
        print(np.mean(episode_lengths[-5:]))
        avg_length = np.mean(episode_lengths[-5:]) if episode_lengths else float('inf')
        if avg_length < best_score:
            best_score = avg_length
            best_params = params
        env.close()

    print('Best hyperparameters:')
    print(
        f'learning_rate={best_params[0]}, buffer_size={best_params[1]}, '
        f'batch_size={best_params[2]}, gamma={best_params[3]}, '
        f'exploration_fraction={best_params[4]}, '
        f'exploration_final_eps={best_params[5]}'
    )
    print(f'Average episode length (last 5): {best_score}')


def plot_results(
    min_length: int = 1500,
    vertical_line: float | None = None,
):
    lengths_1 = load_lengths_from_csv_dir(
        'Proyecto/data/test_dqn_normal/', column_length=min_length
    )
    lengths_2 = load_lengths_from_csv_dir('Proyecto/data/test_dqn_trauma', column_length=min_length)
    plot_learning_curves(
        results={'DQN sin trauma': lengths_1, 'DQN con trauma': lengths_2},
        title='DQN en MountainCar-v0',
        xlabel='Episodio',
        ylabel='Largo promedio de episodio',
        vertical_line_x=len(lengths_1[0]) * vertical_line if vertical_line else None,
    )


def show_agent(agent_path: str, n_steps: int = 1000):
    model = DQN.load(agent_path)
    env = gym.make('MountainCar-v0', render_mode='human')
    observation, info = env.reset()
    for _ in range(n_steps):
        action, _states = model.predict(observation)
        action = int(action)  # Ensure action is an integer
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()


if __name__ == '__main__':
    main(n_runs=3, n_timesteps=300_000, trauma_step=None)
    plot_results(vertical_line=150_001 / 300_000)
