from Environments.MultiGoalEnvs.RoomEnv import RoomEnv
from Agents.QLearning import QLearning
from Agents.Sarsa import Sarsa, NStepSarsa
from MainSimpleEnvs import play_simple_env
import numpy as np
from tqdm import tqdm
from Plotting.Plotter import plot_learning_curves
from concurrent.futures import ThreadPoolExecutor, as_completed


def play_room_env():
    num_episodes = 10
    for _ in range(num_episodes):
        env = RoomEnv()
        play_simple_env(env)


def run_q_learning(env: RoomEnv, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    agent = QLearning(alpha=alpha, gamma=gamma, epsilon=epsilon, initial_q_value=1.0)

    action_space = env.action_space

    episode_lengths = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_length = 0

        while not done:
            action = agent.get_action(state, action_space)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, action_space, done)
            state = next_state
            episode_length += 1

        episode_lengths.append(episode_length)

    return np.array(episode_lengths)


def run_multi_goal_q_learning(env: RoomEnv, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    agent = QLearning(alpha=alpha, gamma=gamma, epsilon=epsilon, initial_q_value=1.0)

    action_space = env.action_space

    episode_lengths = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_length = 0

        while not done:
            action = agent.get_action(state, action_space)
            next_state, reward, done = env.step(action)
            agent.multi_goal_update(state, action, reward, next_state, action_space, env.goals, done)
            state = next_state
            episode_length += 1

        episode_lengths.append(episode_length)

    return np.array(episode_lengths)


def run_sarsa(env: RoomEnv, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    agent = Sarsa(alpha=alpha, gamma=gamma, epsilon=epsilon, initial_q_value=1.0)

    action_space = env.action_space

    episode_lengths = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_length = 0
        action = agent.get_action(state, action_space)

        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.get_action(next_state, action_space)
            agent.update(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
            episode_length += 1

        episode_lengths.append(episode_length)

    return np.array(episode_lengths)


def run_multi_goal_sarsa(env: RoomEnv, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    agent = Sarsa(alpha=alpha, gamma=gamma, epsilon=epsilon, initial_q_value=1.0)

    action_space = env.action_space

    episode_lengths = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_length = 0
        action = agent.get_action(state, action_space)

        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.get_action(next_state, action_space)
            agent.multi_goal_update(state, action, reward, next_state, next_action, env.goals, done)
            state, action = next_state, next_action
            episode_length += 1

        episode_lengths.append(episode_length)

    return np.array(episode_lengths)


def run_n_step_sarsa(env: RoomEnv, n=8, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    agent = NStepSarsa(n=n, alpha=alpha, gamma=gamma, epsilon=epsilon, initial_q_value=1.0)

    action_space = env.action_space

    episode_lengths = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_length = 0
        agent.start_episode()
        action = agent.get_action(state, action_space)

        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.get_action(next_state, action_space)
            agent.step(state, action, reward, done)
            state, action = next_state, next_action
            episode_length += 1

        episode_lengths.append(episode_length)

    return np.array(episode_lengths)


def run_room(num_episodes: int = 500, num_runs: int = 100, epsilon: float = 0.1, alpha: float = 0.1, gamma: float = 0.99):
    env = RoomEnv()
    reults = {
        'Q-Learning': np.zeros((num_runs, num_episodes)),
        'SARSA': np.zeros((num_runs, num_episodes)),
        '8-step SARSA': np.zeros((num_runs, num_episodes)),
        'multi-goal Q-Learning': np.zeros((num_runs, num_episodes)),
        'multi-goal SARSA': np.zeros((num_runs, num_episodes)),
    }

    for run in tqdm(range(num_runs), desc='Running experiments'):
        reults['Q-Learning'][run, :] = run_q_learning(env, num_episodes, alpha, gamma, epsilon)
        reults['multi-goal Q-Learning'][run, :] = run_multi_goal_q_learning(env, num_episodes, alpha, gamma, epsilon)
        reults['SARSA'][run, :] = run_sarsa(env, num_episodes, alpha, gamma, epsilon)
        reults['multi-goal SARSA'][run, :] = run_multi_goal_sarsa(env, num_episodes, alpha, gamma, epsilon)
        reults['8-step SARSA'][run, :] = run_n_step_sarsa(env, n=8, num_episodes=num_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon)

    return reults


if __name__ == '__main__':
    results = run_room(num_episodes=500, num_runs=100, epsilon=0.1, alpha=0.1, gamma=0.99)

    plot_learning_curves(
        results,
        title='Comparación RoomEnv',
        ylabel='Pasos promedio por episodio',
    )
