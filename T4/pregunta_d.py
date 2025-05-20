from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from tqdm import tqdm


def main(n_runs: int = 10, n_timesteps: int = 300_000):
    Path('T4/data/ddpg/').mkdir(parents=True, exist_ok=True)

    for run in tqdm(range(n_runs), desc='DDPG runs'):
        env = gym.make('MountainCarContinuous-v0')
        env = Monitor(env, f'T4/data/ddpg/ddpg_results_{run + 1}')

        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = DDPG(
            'MlpPolicy',
            env,
            learning_rate=0.0003,
            buffer_size=1_000_000,
            learning_starts=5000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=(1, 'step'),
            gradient_steps=1,
            action_noise=action_noise,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
        )

        model.learn(total_timesteps=n_timesteps, log_interval=10)

        env.close()


if __name__ == '__main__':
    main()
