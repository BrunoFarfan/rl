import numpy as np
from Agents.ModelBased import Dyna, RMax
from Agents.QLearning import QLearning
from Agents.Sarsa import NStepSarsa, Sarsa
from Environments.AbstractEnv import AbstractEnv
from Environments.SimpleEnvs.CliffEnv import CliffEnv
from Environments.SimpleEnvs.EscapeRoomEnv import EscapeRoomEnv
from Plotting.Plotter import plot_learning_curves
from tqdm import tqdm


def show(env, current_state, reward=None):
    env.show()
    print(f'Raw state: {current_state}')
    if reward:
        print(f'Reward: {reward}')


def get_action_from_user(valid_actions):
    key = input()
    while key not in valid_actions:
        key = input()
    return valid_actions[key]


def play_simple_env(simple_env):
    key2action = {'a': 'left', 'd': 'right', 'w': 'up', 's': 'down'}
    s = simple_env.reset()
    show(simple_env, s)
    done = False
    while not done:
        print('Action: ', end='')
        action = get_action_from_user(key2action)
        s, r, done = simple_env.step(action)
        show(simple_env, s, r)


def run_q_learning(
    env: AbstractEnv,
    num_episodes: int = 500,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    gamma: float = 1.0,
):
    agent = QLearning(epsilon=epsilon, alpha=alpha, gamma=gamma)

    action_space = env.action_space

    episode_returns = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, action_space)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, action_space, done)
            state = next_state
            total_reward += reward

        episode_returns.append(total_reward)

    return np.array(episode_returns)


def run_sarsa(
    env: AbstractEnv,
    num_episodes: int = 500,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    gamma: float = 1.0,
):
    agent = Sarsa(epsilon=epsilon, alpha=alpha, gamma=gamma)

    action_space = env.action_space

    episode_returns = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        action = agent.get_action(state, action_space)

        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.get_action(next_state, env.action_space)
            agent.update(state, action, reward, next_state, next_action, done)
            state, action = next_state, next_action
            total_reward += reward

        episode_returns.append(total_reward)

    return np.array(episode_returns)


def run_n_step_sarsa(
    env: AbstractEnv,
    num_episodes: int = 500,
    n: int = 4,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    gamma: float = 1.0,
):
    agent = NStepSarsa(n=n, epsilon=epsilon, alpha=alpha, gamma=gamma)

    action_space = env.action_space

    episode_returns = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        agent.start_episode()
        action = agent.get_action(state, action_space)
        while not done:
            next_state, reward, done = env.step(action)
            next_action = agent.get_action(next_state, action_space)
            agent.step(state, action, reward, done)
            state, action = next_state, next_action
            total_reward += reward

        agent.finish_episode()
        episode_returns.append(total_reward)

    return np.array(episode_returns)


def run_cliff(
    num_episodes: int = 500,
    num_runs: int = 100,
    epsilon: float = 0.1,
    alpha: float = 0.1,
    gamma: float = 1.0,
):
    env = CliffEnv()
    results = {
        'Q-Learning': np.zeros((num_runs, num_episodes)),
        'SARSA': np.zeros((num_runs, num_episodes)),
        '4-step SARSA': np.zeros((num_runs, num_episodes)),
    }

    for run in tqdm(range(num_runs), desc='Running experiments'):
        results['Q-Learning'][run, :] = run_q_learning(
            env, num_episodes=num_episodes, epsilon=epsilon, alpha=alpha, gamma=gamma
        )
        results['SARSA'][run, :] = run_sarsa(
            env, num_episodes=num_episodes, epsilon=epsilon, alpha=alpha, gamma=gamma
        )
        results['4-step SARSA'][run, :] = run_n_step_sarsa(
            env,
            num_episodes=num_episodes,
            n=4,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
        )

    plot_learning_curves(
        results,
        title='Comparación en CliffEnv',
        ylabel='Retorno promedio',
        y_min=-200,
        y_max=0,
    )


def run_dyna(
    env: EscapeRoomEnv,
    num_episodes: int = 20,
    alpha: float = 0.5,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    initial_q_value: float = 0.0,
    planning_steps: int = 5,
):
    agent = Dyna(alpha, epsilon, gamma, initial_q_value, planning_steps)

    action_space = env.action_space

    episode_returns = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, action_space)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, action_space, done)
            state = next_state
            total_reward += reward

        episode_returns.append(total_reward)

    return np.array(episode_returns)


def run_rmax(
    env: EscapeRoomEnv,
    num_episodes: int = 20,
    gamma: float = 1.0,
    k: int = 1,
    rmax: float = 1.0,
    theta: float = 1e-6,
    max_iterations: int = 1000,
):
    agent = RMax(gamma, k, rmax, theta, max_iterations)

    action_space = env.action_space

    episode_returns = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, action_space)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        episode_returns.append(total_reward)

    return np.array(episode_returns)


def run_escape_room(
    num_runs: int = 5,
    num_episodes: int = 20,
    alpha: float = 0.5,
    epsilon: float = 0.1,
    gamma: float = 1.0,
    initial_q_value: float = 0.0,
    planning_steps_array: list[int] = [0, 1, 10, 100, 1_000, 10_000],
    k: int = 1,
    rmax: float = 1.0,
    theta: float = 1e-6,
    max_iterations: int = 1000,
):
    env = EscapeRoomEnv()

    results = {}

    for run in tqdm(range(num_runs), desc=f'Running experiments'):
        for planning_steps in planning_steps_array:
            temp_results = run_dyna(
                env,
                num_episodes=num_episodes,
                alpha=alpha,
                epsilon=epsilon,
                gamma=gamma,
                initial_q_value=initial_q_value,
                planning_steps=planning_steps,
            )
            if f'Dyna - {planning_steps}' not in results:
                results[f'Dyna - {planning_steps}'] = np.zeros((num_runs, num_episodes))
            results[f'Dyna - {planning_steps}'][run, :] = temp_results

        temp_results = run_rmax(
            env,
            num_episodes=num_episodes,
            gamma=gamma,
            k=k,
            rmax=rmax,
            theta=theta,
            max_iterations=max_iterations,
        )
        if 'RMax' not in results:
            results['RMax'] = np.zeros((num_runs, num_episodes))
        results['RMax'][run, :] = temp_results

    return results


if __name__ == '__main__':
    # run_cliff(num_episodes=500, num_runs=100, epsilon=0.1, alpha=0.1, gamma=1.0)

    planning_steps_array = [0, 1, 10, 100, 1_000, 10_000]

    results = run_escape_room(planning_steps_array=planning_steps_array)

    for label, runs in results.items():
        mean = runs.mean(axis=0)

        print(f'{label}: {mean}')
