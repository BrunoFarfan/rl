from Environments.BlackjackEnv import BlackjackEnv
from Environments.CliffEnv import CliffEnv
from plot.plot_mc_results import plot_blackjack_results, plot_cliff_results
from Evaluator.mc_control_es import mc_control_es
import random
from collections import defaultdict



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


def test_greedy_policy(env_class, policy, num_episodes=100_000):
    total_return = 0.0
    env = env_class()
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            if state not in policy:
                action = random.choice(env.action_space)
            else:
                action = max(policy[state], key=policy[state].get)
            state, reward, done = env.step(action)
            total_return += reward
    return total_return / num_episodes


def run_blackjack():
    epsilon = 0.01
    gamma = 1.0
    total_episodes = 10_000_000
    eval_every = 500_000

    num_runs = 5
    results = []

    for run in range(num_runs):
        print(f"Blackjack - Run {run + 1}")
        env = BlackjackEnv()
        Q = defaultdict(lambda: defaultdict(float))
        pi = {}
        rewards = []

        for episodes_done in range(0, total_episodes + 1, eval_every):
            train_block = eval_every if episodes_done > 0 else 1
            Q, pi = mc_control_es(env, gamma, epsilon, train_block, Q, pi)

            avg_return = test_greedy_policy(BlackjackEnv, pi)
            rewards.append(avg_return)
            print(f"  Episodios: {episodes_done}, Retorno promedio: {avg_return:.4f}")

        results.append(rewards)

    plot_blackjack_results(results, eval_every, total_episodes)


def run_cliff():
    epsilon = 0.1
    gamma = 1.0
    total_episodes = 200_000
    eval_every = 1_000
    num_runs = 5
    results = []

    for run in range(num_runs):
        print(f"Cliff - Run {run + 1}")
        env = CliffEnv(width=6)
        Q = defaultdict(lambda: defaultdict(float))
        pi = {}
        rewards = []

        for episodes_done in range(0, total_episodes + 1, eval_every):
            train_block = eval_every if episodes_done > 0 else 1
            Q, pi = mc_control_es(env, gamma, epsilon, train_block, Q, pi)

            env_eval = CliffEnv(width=6)
            s = env_eval.reset()
            done = False
            total_return = 0.0
            while not done:
                if s not in pi:
                    a = random.choice(env_eval.action_space)
                else:
                    a = max(pi[s], key=pi[s].get)
                s, r, done = env_eval.step(a)
                total_return += r
            rewards.append(total_return)
            print(f"  Episodios: {episodes_done}, Retorno: {total_return:.1f}")

        results.append(rewards)

    plot_cliff_results(results, eval_every, total_episodes)


if __name__ == '__main__':
    run_blackjack()
    run_cliff()
