import gymnasium as gym
import numpy as np
from agents.ActorCritic import ActorCritic
from plotting.plotter import plot_learning_curves
from tqdm import tqdm


def run_actor_critic(
    num_runs: int = 30,
    num_episodes: int = 1000,
    gamma: float = 1.0,
    lr_critic: float = 0.001,
    lr_actor: float = 0.0001,
):
    env = gym.make('MountainCarContinuous-v0')

    all_lengths = np.zeros((num_runs, num_episodes))
    for run in tqdm(range(num_runs), desc='Actor-Critic runs'):
        actor_critic = ActorCritic(gamma, lr_critic, lr_actor)

        for episode in range(num_episodes):
            observation, _ = env.reset()
            actor_critic.reset_episode_values()
            terminated = truncated = False
            episode_length = 0

            while not terminated and not truncated:
                action = actor_critic.sample_action(observation)
                next_observation, reward, terminated, truncated, _ = env.step([action])
                actor_critic.learn(observation, action, reward, next_observation, terminated)
                observation = next_observation
                episode_length += 1

            all_lengths[run][episode] = episode_length

        env.close()

    return all_lengths, actor_critic


def show_actor_critic(agent, n_steps=1000):
    env = gym.make('MountainCarContinuous-v0', render_mode='human')
    observation, info = env.reset()
    for _ in range(n_steps):
        action = agent.sample_action(observation)
        observation, reward, terminated, truncated, info = env.step([action])
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


def average_episode_lengths(lengths: np.ndarray, grouping_size: int = 10) -> np.ndarray:
    return lengths.reshape((lengths.shape[0], -1, grouping_size)).mean(axis=2)


def main(display_agent: bool = False):
    lengths, actor_critic = run_actor_critic()
    avg_lengths = average_episode_lengths(lengths)

    plot_learning_curves(
        results={
            'Actor-Critic': avg_lengths,
        },
        title='Actor-Critic con aproximación lineal',
        xlabel='Episodios (* 10)',
        ylabel='Longitud promedio de episodios',
    )

    if display_agent:
        show_actor_critic(actor_critic)


if __name__ == '__main__':
    main(display_agent=True)
