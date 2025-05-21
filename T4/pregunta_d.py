from pathlib import Path

import gymnasium as gym
import numpy as np
from plotting.csv_reader import load_lengths_from_csv_dir
from plotting.plotter import plot_learning_curves
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from tqdm import tqdm


def main(n_runs: int = 10, n_timesteps: int = 300_000):
    Path('T4/data/ddpg/').mkdir(parents=True, exist_ok=True)

    for run in tqdm(range(n_runs), desc='DDPG runs'):
        env = gym.make('MountainCarContinuous-v0')
        env = Monitor(env, f'T4/data/ddpg/ddpg_results_{run + 1}')

        n_actions = env.action_space.shape[0]
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions)
        )

        model = DDPG(
            policy='MlpPolicy',
            env=env,
            learning_rate=1e-3,
            batch_size=256,
            gradient_steps=1,
            train_freq=1,
            action_noise=action_noise,
            policy_kwargs=dict(net_arch=[400, 300]),
            verbose=1,
        )

        model.learn(total_timesteps=n_timesteps, log_interval=10)

        env.close()

    # Save the model
    model.save('T4/data/ddpg/ddpg_model')


def plot_results(column_length: int = 1500):
    lengths = load_lengths_from_csv_dir('T4/data/ddpg/', column_length, tail=False)
    plot_learning_curves(
        results={
            'DDPG': lengths,
        },
        title='DDPG en MountainCarContinuous-v0',
        xlabel='Episodio',
        ylabel='Largo promedio de episodio',
    )


def show_agent(agent_path: str, n_steps: int = 1000):
    model = DDPG.load(agent_path)
    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    observation, info = env.reset()
    for _ in range(n_steps):
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


if __name__ == '__main__':
    show_agent('T4/data/ddpg/ddpg_model')
