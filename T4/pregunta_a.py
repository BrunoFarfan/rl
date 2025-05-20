import gymnasium as gym
import numpy as np
from agents.QLearning import QLearning
from agents.Sarsa import Sarsa
from plotting.plotter import plot_learning_curves
from tqdm import tqdm


def run_experiment_qlearning(
    num_runs: int = 30,
    num_episodes: int = 1000,
    gamma: float = 1.0,
    epsilon: float = 0.0,
    alpha: float = 0.5 / 8,
):
    env = gym.make('MountainCar-v0')
    n_actions = env.action_space.n

    all_lengths = np.zeros((num_runs, num_episodes))
    for run in tqdm(range(num_runs), desc='Q-Learning runs'):
        qlearning = QLearning(n_actions, epsilon, alpha, gamma)
        for episode in range(num_episodes):
            observation, info = env.reset()
            terminated = truncated = False
            episode_length = 0

            while not terminated and not truncated:
                action = qlearning.sample_action(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                qlearning.learn(observation, action, reward, next_observation, terminated)
                observation = next_observation
                episode_length += 1

            all_lengths[run][episode] = episode_length

        env.close()

    return all_lengths, qlearning


def run_experiment_sarsa(
    num_runs: int = 30,
    num_episodes: int = 1000,
    gamma: float = 1.0,
    epsilon: float = 0.0,
    alpha: float = 0.5 / 8,
):
    env = gym.make('MountainCar-v0')
    n_actions = env.action_space.n

    all_lengths = np.zeros((num_runs, num_episodes))
    for run in tqdm(range(num_runs), desc='Sarsa runs'):
        sarsa = Sarsa(n_actions, epsilon, alpha, gamma)
        for episode in range(num_episodes):
            observation, info = env.reset()
            action = sarsa.sample_action(observation)
            terminated = truncated = False
            episode_length = 0

            while not terminated and not truncated:
                next_observation, reward, terminated, truncated, info = env.step(action)
                next_action = sarsa.sample_action(next_observation)
                sarsa.learn(observation, action, reward, next_observation, next_action, terminated)
                observation, action = next_observation, next_action
                episode_length += 1

            all_lengths[run][episode] = episode_length

        env.close()

    return all_lengths, sarsa


def average_episode_lengths(lengths: np.ndarray, grouping_size: int = 10) -> np.ndarray:
    return lengths.reshape((lengths.shape[0], -1, grouping_size)).mean(axis=2)


def show_agent(agent, n_steps=1000):
    env = gym.make('MountainCar-v0', render_mode='human')
    observation, info = env.reset()
    for _ in range(n_steps):
        action = agent.argmax(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
    env.close()


def main(display_agent: bool = False):
    q_learning_lengths, q_learning_agent = run_experiment_qlearning()
    q_learning_avg = average_episode_lengths(q_learning_lengths)

    sarsa_lengths, sarsa_agent = run_experiment_sarsa()
    sarsa_avg = average_episode_lengths(sarsa_lengths)

    plot_learning_curves(
        results={
            'Q-Learning': q_learning_avg,
            'Sarsa': sarsa_avg,
        },
        title='MountainCar-v0, Q-Learning vs Sarsa: aproximación lineal',
        xlabel='Episodios (* 10)',
        ylabel='Longitud del episodio promedio',
    )

    if display_agent:
        show_agent(q_learning_agent)
        show_agent(sarsa_agent)


if __name__ == '__main__':
    main(display_agent=True)
