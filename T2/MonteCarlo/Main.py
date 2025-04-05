from Environments.BlackjackEnv import BlackjackEnv
from Environments.CliffEnv import CliffEnv

from plot.plot_mc_results import plot_blackjack_results, plot_cliff_results
from plot.plot_mc_policy import plot_blackjack_policy, plot_cliff_policy

from plot.plot_mc_diagnostics import plot_blackjack_visits, plot_cliff_visits

from Evaluator.mc_control_es import MonteCarloControlEveryVisit

import matplotlib.pyplot as plt


def get_action_from_user(actions):
    print("Valid actions:")
    for i in range(len(actions)):
        print(f"{i}. {actions[i]}")
    print("Please select an action:")
    selected_id = -1
    while not (0 <= selected_id < len(actions)):
        selected_id = int(input())
    return actions[selected_id]


def play(env):
    actions = env.action_space
    state = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        env.show()
        action = get_action_from_user(actions)
        state, reward, done = env.step(action)
        total_reward += reward
    env.show()
    print("Done.")
    print(f"Total reward: {total_reward}")


def play_blackjack():
    env = BlackjackEnv()
    play(env)


def play_cliff():
    cliff_width = 6
    env = CliffEnv(cliff_width)
    play(env)


def run_blackjack_basic():
    env = BlackjackEnv()
    agent = MonteCarloControlEveryVisit(env, epsilon=0.01, gamma=1.0)

    agent.train(num_episodes=500_000)

    plot_blackjack_visits(agent.get_state_visits())
    plot_blackjack_policy(agent.get_policy())


def evaluate_policy(env, policy, num_episodes=1, gamma=1.0):
    total_return = 0.0

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        G = 0
        discount = 1.0

        while not done:
            greedy_action = max(policy[state], key=policy[state].get)
            state, reward, done = env.step(greedy_action)
            G += discount * reward
            discount *= gamma

        total_return += G

    return total_return / num_episodes


def run_experiment_blackjack(agent_class, env_class, epsilon=0.01, gamma=1.0,
                             num_runs=5, num_episodes=10_000_000,
                             eval_interval=500_000, eval_episodes=100_000):
    eval_points = [1] + list(range(eval_interval, num_episodes + 1, eval_interval))
    all_returns = []

    for run in range(num_runs):
        print(f"\n===> RUN {run + 1}")
        env = env_class()
        agent = agent_class(env, epsilon=epsilon, gamma=gamma)

        returns = []
        first_eval_done = False

        for episode in range(1, num_episodes + 1):
            episode_data = agent.generate_episode()
            agent.update(episode_data)

            if episode == 1 and not first_eval_done:
                ret = evaluate_policy(env, agent.get_policy(), eval_episodes, gamma)
                returns.append(ret)
                print(f"Eval (ep 1): {ret:.4f}")
                first_eval_done = True

            elif episode % eval_interval == 0:
                ret = evaluate_policy(env, agent.get_policy(), eval_episodes, gamma)
                returns.append(ret)
                print(f"Eval (ep {episode}): {ret:.4f}")

        all_returns.append(returns)

        # graficar la última ejec.
        if run == num_runs - 1:
            plot_blackjack_policy(agent.get_policy(), title=f"Política aprendida (run Nº{run+1})")
            plot_blackjack_visits(agent.get_state_visits(), title=f"Frecuencia de visitas (run {run+1})")

    plot_blackjack_results(all_returns, eval_points)

    return all_returns


def run_experiment_cliff(agent_class, env_class, epsilon=0.1, gamma=1.0,
                         num_runs=5, num_episodes=200_000, eval_interval=1_000):
    eval_points = [1] + list(range(eval_interval, num_episodes + 1, eval_interval))
    all_returns = []

    for run in range(num_runs):
        print(f"\n===> RUN {run + 1}")
        env = env_class(width=6)
        agent = agent_class(env, epsilon=epsilon, gamma=gamma)
        returns = []

        for episode in range(1, num_episodes + 1):
            episode_data = agent.generate_episode()
            agent.update(episode_data)

            if episode == 1 or episode % eval_interval == 0:
                avg_return = evaluate_policy(env, agent.get_policy(), num_episodes=1, gamma=gamma)
                returns.append(avg_return)
                print(f"Eval (ep {episode}): retorno = {avg_return:.2f}")

        all_returns.append(returns)

        if run == num_runs - 1:
            plot_cliff_policy(agent.get_policy(), width=6, height=4)
            plot_cliff_visits(agent.get_state_visits(), width=6, height=4)

    plot_cliff_results(all_returns, eval_points)

    return all_returns


if __name__ == '__main__':
    returns_cliff = run_experiment_cliff(
        agent_class=MonteCarloControlEveryVisit,
        env_class=CliffEnv,
        epsilon=0.1,
        gamma=1.0,
    )

    print("Retornos Cliff:", returns_cliff)

    returns_bj = run_experiment_blackjack(
        agent_class=MonteCarloControlEveryVisit,
        env_class=BlackjackEnv,
        epsilon=0.01,
        gamma=1.0
    )

    print("Retornos Blackjack:", returns_bj)
