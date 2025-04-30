import numpy as np
from Agents.QLearning import QLearning
from Agents.Sarsa import NStepSarsa, Sarsa
from Environments.PartiallyObservableEnvs.InvisibleDoorEnv import InvisibleDoorEnv
from MainSimpleEnvs import get_action_from_user, play_simple_env, show
from MemoryWrappers.BinaryMemory import BinaryMemory
from MemoryWrappers.CustomKOrderMemory import CustomKOrderMemory
from MemoryWrappers.KOrderMemory import KOrderMemory
from Plotting.Plotter import plot_learning_curves
from tqdm import tqdm


def play_env_with_binary_memory():
    num_of_bits = 1
    env = InvisibleDoorEnv()
    env = BinaryMemory(env, num_of_bits)

    key2action = {'a': 'left', 'd': 'right', 'w': 'up', 's': 'down'}
    key2memory = {str(i): i for i in range(2**num_of_bits)}
    s = env.reset()
    show(env, s)
    done = False
    while not done:
        print('Environment action: ', end='')
        env_action = get_action_from_user(key2action)
        print(f'Memory action ({", ".join(key2memory.keys())}): ', end='')
        mem_action = get_action_from_user(key2memory)
        action = env_action, mem_action
        s, r, done = env.step(action)
        show(env, s, r)


def play_env_with_k_order_memory():
    memory_size = 2
    env = InvisibleDoorEnv()
    env = KOrderMemory(env, memory_size)
    play_simple_env(env)


def play_env_without_extra_memory():
    env = InvisibleDoorEnv()
    play_simple_env(env)


def run_q_learning(
    env: InvisibleDoorEnv,
    num_episodes: int = 1000,
    alpha: float = 0.1,
    epsilon: float = 0.01,
    gamma: float = 0.99,
    initial_q_values: float = 1.0,
):
    agent = QLearning(alpha=alpha, gamma=gamma, epsilon=epsilon, initial_q_value=initial_q_values)

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


def run_sarsa(
    env: InvisibleDoorEnv,
    num_episodes: int = 1000,
    alpha: float = 0.1,
    epsilon: float = 0.01,
    gamma: float = 0.99,
    initial_q_values: float = 1.0,
):
    agent = Sarsa(alpha=alpha, gamma=gamma, epsilon=epsilon, initial_q_value=initial_q_values)

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


def run_n_step_sarsa(
    env: InvisibleDoorEnv,
    n: int = 16,
    num_episodes: int = 1000,
    alpha: float = 0.1,
    epsilon: float = 0.01,
    gamma: float = 0.99,
    initial_q_values: float = 1.0,
):
    agent = NStepSarsa(
        n=n, alpha=alpha, gamma=gamma, epsilon=epsilon, initial_q_value=initial_q_values
    )

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

        agent.finish_episode()
        episode_lengths.append(episode_length)

    return np.array(episode_lengths)


def run_no_memory(
    num_runs: int = 30,
    num_episodes: int = 1000,
    alpha: float = 0.1,
    epsilon: float = 0.01,
    gamma: float = 0.99,
    initial_q_values: float = 1.0,
):
    env = InvisibleDoorEnv()
    results = {
        'Q-Learning no memory': np.zeros((num_runs, num_episodes)),
        'Sarsa no memory': np.zeros((num_runs, num_episodes)),
        '16-Step Sarsa no memory': np.zeros((num_runs, num_episodes)),
    }

    for run in tqdm(range(num_runs), desc='Running no memory experiments'):
        results['Q-Learning no memory'][run, :] = run_q_learning(
            env, num_episodes, alpha, epsilon, gamma, initial_q_values
        )
        results['Sarsa no memory'][run, :] = run_sarsa(
            env, num_episodes, alpha, epsilon, gamma, initial_q_values
        )
        results['16-Step Sarsa no memory'][run, :] = run_n_step_sarsa(
            env, 16, num_episodes, alpha, epsilon, gamma, initial_q_values
        )

    return results


def run_k_order_memory(
    memory_size: int = 2,
    num_runs: int = 30,
    num_episodes: int = 1000,
    alpha: float = 0.1,
    epsilon: float = 0.01,
    gamma: float = 0.99,
    initial_q_values: float = 1.0,
):
    env = InvisibleDoorEnv()
    env = KOrderMemory(env, memory_size)

    results = {
        'Q-Learning k-order memory': np.zeros((num_runs, num_episodes)),
        'Sarsa k-order memory': np.zeros((num_runs, num_episodes)),
        '16-Step Sarsa k-order memory': np.zeros((num_runs, num_episodes)),
    }

    for run in tqdm(range(num_runs), desc='Running k-order memory experiments'):
        results['Q-Learning k-order memory'][run, :] = run_q_learning(
            env, num_episodes, alpha, epsilon, gamma, initial_q_values
        )
        results['Sarsa k-order memory'][run, :] = run_sarsa(
            env, num_episodes, alpha, epsilon, gamma, initial_q_values
        )
        results['16-Step Sarsa k-order memory'][run, :] = run_n_step_sarsa(
            env, 16, num_episodes, alpha, epsilon, gamma, initial_q_values
        )

    return results


def run_binary_memory(
    num_of_bits: int = 1,
    num_runs: int = 30,
    num_episodes: int = 1000,
    alpha: float = 0.1,
    epsilon: float = 0.01,
    gamma: float = 0.99,
    initial_q_values: float = 1.0,
):
    env = InvisibleDoorEnv()
    env = BinaryMemory(env, num_of_bits)

    results = {
        'Q-Learning binary memory': np.zeros((num_runs, num_episodes)),
        'Sarsa binary memory': np.zeros((num_runs, num_episodes)),
        '16-Step Sarsa binary memory': np.zeros((num_runs, num_episodes)),
    }

    for run in tqdm(range(num_runs), desc='Running binary memory experiments'):
        results['Q-Learning binary memory'][run, :] = run_q_learning(
            env, num_episodes, alpha, epsilon, gamma, initial_q_values
        )
        results['Sarsa binary memory'][run, :] = run_sarsa(
            env, num_episodes, alpha, epsilon, gamma, initial_q_values
        )
        results['16-Step Sarsa binary memory'][run, :] = run_n_step_sarsa(
            env, 16, num_episodes, alpha, epsilon, gamma, initial_q_values
        )

    return results


def run_custom_k_order_memory(
    memory_size: int = 1,
    num_runs: int = 30,
    num_episodes: int = 1000,
    alpha: float = 0.1,
    epsilon: float = 0.01,
    gamma: float = 0.99,
    initial_q_values: float = 1.0,
):
    env = InvisibleDoorEnv()
    env = CustomKOrderMemory(env, memory_size)

    results = {
        'Q-Learning custom k-order memory': np.zeros((num_runs, num_episodes)),
        'Sarsa custom k-order memory': np.zeros((num_runs, num_episodes)),
        '16-Step Sarsa custom k-order memory': np.zeros((num_runs, num_episodes)),
    }

    for run in tqdm(range(num_runs), desc='Running custom k-order memory experiments'):
        results['Q-Learning custom k-order memory'][run, :] = run_q_learning(
            env, num_episodes, alpha, epsilon, gamma, initial_q_values
        )
        results['Sarsa custom k-order memory'][run, :] = run_sarsa(
            env, num_episodes, alpha, epsilon, gamma, initial_q_values
        )
        results['16-Step Sarsa custom k-order memory'][run, :] = run_n_step_sarsa(
            env, 16, num_episodes, alpha, epsilon, gamma, initial_q_values
        )

    return results


if __name__ == '__main__':
    # play_env_without_extra_memory()
    # play_env_with_k_order_memory()
    # play_env_with_binary_memory()

    # results_no_memory = run_no_memory()

    # results_k_order = run_k_order_memory()

    # results_binary = run_binary_memory()

    results_custom = run_custom_k_order_memory()

    # plot_learning_curves(
    #     results_no_memory,
    #     title='Comparación de agentes sin memoria en InvisibleDoorEnv',
    #     ylabel='Pasos promedio por episodio',
    # )

    # plot_learning_curves(
    #     results_k_order,
    #     title='Comparación de agentes con k-order memory en InvisibleDoorEnv',
    #     ylabel='Pasos promedio por episodio',
    # )

    # plot_learning_curves(
    #     results_binary,
    #     title='Comparación de agentes con binary memory en InvisibleDoorEnv',
    #     ylabel='Pasos promedio por episodio',
    # )

    plot_learning_curves(
        results_custom,
        title='Comparación de agentes con Custom k-order memory en InvisibleDoorEnv',
        ylabel='Pasos promedio por episodio',
    )
