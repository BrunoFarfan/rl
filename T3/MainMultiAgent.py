import numpy as np
from Agents.QLearning import QLearning
from Environments.MultiAgentEnvs.CentralizedHunterEnv import CentralizedHunterEnv
from Environments.MultiAgentEnvs.HunterAndPreyEnv import HunterAndPreyEnv
from Environments.MultiAgentEnvs.HunterEnv import HunterEnv
from MainSimpleEnvs import get_action_from_user, show
from Plotting.Plotter import plot_learning_curves
from tqdm import tqdm


def play_hunter_env():
    hunter_env = HunterAndPreyEnv()

    key2action = {'a': 'left', 'd': 'right', 'w': 'up', 's': 'down', '': 'None'}
    num_of_agents = hunter_env.num_of_agents
    s = hunter_env.reset()
    show(hunter_env, s)
    done = False
    while not done:
        print('Hunter A: ', end='')
        hunter1 = get_action_from_user(key2action)
        print('Hunter B: ', end='')
        hunter2 = get_action_from_user(key2action)
        action = hunter1, hunter2
        if num_of_agents == 3:
            print('Prey: ', end='')
            prey = get_action_from_user(key2action)
            action = hunter1, hunter2, prey
        s, r, done = hunter_env.step(action)
        show(hunter_env, s, r)


def run_centralized_q_learning(
    env_class=CentralizedHunterEnv,
    num_runs: int = 30,
    num_episodes: int = 50_000,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 0.95,
    start_values: float = 1.0,
):
    env = env_class()
    action_space = env.action_space

    episode_report_period = 100  # report every 100 episodes to not overload the output plot

    results = {
        'Q-Learning': np.zeros((num_runs, num_episodes // episode_report_period)),
    }

    for run in tqdm(range(num_runs), desc='Running Q-Learning on CentralizedHunterEnv'):
        agent = QLearning(alpha=alpha, epsilon=epsilon, gamma=gamma, initial_q_value=start_values)

        episode_lengths = []
        episode_length = 0
        for episode in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                action = agent.get_action(state, action_space)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, action_space, done)
                state = next_state
                episode_length += 1

            if episode % episode_report_period == 0:
                episode_length /= episode_report_period
                episode_lengths.append(episode_length)
                episode_length = 0

        results['Q-Learning'][run, :] = episode_lengths

    return results


def run_decentralized_cooperative_q_learning(
    env_class=HunterEnv,
    num_runs: int = 30,
    num_episodes: int = 50_000,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 0.95,
    start_values: float = 1.0,
):
    env = env_class()
    single_agent_action_space = env.single_agent_action_space

    episode_report_period = 100  # report every 100 episodes to not overload the output plot

    results = {
        'Q-Learning cooperativo': np.zeros((num_runs, num_episodes // episode_report_period)),
    }

    for run in tqdm(range(num_runs), desc='Running Q-Learning on HunterEnv'):
        agent1 = QLearning(alpha=alpha, epsilon=epsilon, gamma=gamma, initial_q_value=start_values)
        agent2 = QLearning(alpha=alpha, epsilon=epsilon, gamma=gamma, initial_q_value=start_values)

        episode_lengths = []
        episode_length = 0
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action1 = agent1.get_action(state, single_agent_action_space)
                action2 = agent2.get_action(state, single_agent_action_space)
                actions = (action1, action2)

                next_sate, (reward1, reward2), done = env.step(actions)

                agent1.update(state, action1, reward1, next_sate, single_agent_action_space, done)
                agent2.update(state, action2, reward2, next_sate, single_agent_action_space, done)

                state = next_sate
                episode_length += 1

            if episode % episode_report_period == 0:
                episode_length /= episode_report_period
                episode_lengths.append(episode_length)
                episode_length = 0

        results['Q-Learning cooperativo'][run, :] = episode_lengths

    return results


def run_decentralized_competitve_q_learning(
    env_class=HunterAndPreyEnv,
    num_runs: int = 30,
    num_episodes: int = 50_000,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 0.95,
    start_values: float = 1.0,
):
    env = env_class()
    single_agent_action_space = env.single_agent_action_space

    episode_report_period = 100  # report every 100 episodes to not overload the output plot

    results = {
        'Q-Learning competitivo': np.zeros((num_runs, num_episodes // episode_report_period)),
    }

    for run in tqdm(range(num_runs), desc='Running Q-Learning on HunterEnv'):
        hunter1 = QLearning(alpha=alpha, epsilon=epsilon, gamma=gamma, initial_q_value=start_values)
        hunter2 = QLearning(alpha=alpha, epsilon=epsilon, gamma=gamma, initial_q_value=start_values)
        prey = QLearning(alpha=alpha, epsilon=epsilon, gamma=gamma, initial_q_value=start_values)

        episode_lengths = []
        episode_length = 0
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action_hunter1 = hunter1.get_action(state, single_agent_action_space)
                action_hunter2 = hunter2.get_action(state, single_agent_action_space)
                action_prey = prey.get_action(state, single_agent_action_space)
                actions = (action_hunter1, action_hunter2, action_prey)

                next_sate, (reward1, reward2, reward3), done = env.step(actions)

                hunter1.update(
                    state,
                    action_hunter1,
                    reward1,
                    next_sate,
                    single_agent_action_space,
                    done,
                )
                hunter2.update(
                    state,
                    action_hunter2,
                    reward2,
                    next_sate,
                    single_agent_action_space,
                    done,
                )
                prey.update(
                    state,
                    action_prey,
                    reward3,
                    next_sate,
                    single_agent_action_space,
                    done,
                )

                state = next_sate
                episode_length += 1

            if episode % episode_report_period == 0:
                episode_length /= episode_report_period
                episode_lengths.append(episode_length)
                episode_length = 0

        results['Q-Learning competitivo'][run, :] = episode_lengths

    return results


if __name__ == '__main__':
    results_centralized = run_centralized_q_learning()
    results_cooperative = run_decentralized_cooperative_q_learning()
    results_competitive = run_decentralized_competitve_q_learning()

    results = {
        'Centralized Q-Learning': results_centralized['Q-Learning'],
        'Cooperative Q-Learning': results_cooperative['Q-Learning cooperativo'],
        'Competitive Q-Learning': results_competitive['Q-Learning competitivo'],
    }

    plot_learning_curves(
        results,
        title='HunterAndPreyEnv Q-Learning',
        ylabel='Pasos promedio por episodio',
        xlabel='Episodios (x100)',
    )
