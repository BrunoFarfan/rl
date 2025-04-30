import matplotlib.pyplot as plt
import numpy as np


def plot_average_reward(results_dict):
    plt.figure(figsize=(8, 6))

    num_of_steps = len(next(iter(results_dict.values()))[0])
    steps = np.arange(1, num_of_steps + 1)

    colors = plt.cm.get_cmap('tab10', len(results_dict))

    for i, (agent_name, (avg_rewards, _)) in enumerate(results_dict.items()):
        plt.plot(steps, avg_rewards, label=agent_name, color=colors(i))

    plt.ylabel('Average Reward')
    plt.xlabel('Steps')
    plt.legend()
    plt.title('Average Reward Over Time')
    plt.show()


def plot_optimal_action_percentage(results_dict):
    plt.figure(figsize=(8, 6))

    num_of_steps = len(next(iter(results_dict.values()))[1])
    steps = np.arange(1, num_of_steps + 1)

    colors = plt.cm.get_cmap('tab10', len(results_dict))

    for i, (agent_name, (_, avg_optimal_actions)) in enumerate(results_dict.items()):
        plt.plot(steps, avg_optimal_actions, label=agent_name, color=colors(i))

    plt.ylabel('% Optimal Action')
    plt.xlabel('Steps')
    plt.legend()
    plt.title('Percentage of Optimal Action Over Time')
    plt.show()
