import random
from collections import defaultdict
from Environments.AbstractEnv import AbstractEnv


def update_policy(state, Q, policy, epsilon: float, env: AbstractEnv):
    actions = list(Q[state].keys())
    if not actions:
        actions = env.action_space
    best_a = max(actions, key=lambda a: Q[state][a])
    n = len(actions)
    policy[state] = {a: epsilon / n for a in actions}
    policy[state][best_a] += 1 - epsilon


def mc_control_es(
        env: AbstractEnv,
        gamma: float=1.0,
        epsilon: float=0.1,
        num_episodes: int=100_000,
        Q: defaultdict=None,
        policy: dict=None
    ):
    if Q is None:
        Q = defaultdict(lambda: defaultdict(float))
    if policy is None:
        policy = {}
    N = defaultdict(lambda: defaultdict(int))

    for episode in range(num_episodes):
        episode = []
        state = env.reset()
        done = False

        while not done:
            if state not in policy:
                actions = env.action_space
                policy[state] = {a: 1 / len(actions) for a in actions}
            actions = list(policy[state].keys())
            probs = list(policy[state].values())
            action = random.choices(actions, weights=probs)[0]

            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        G = 0.0
        for t in reversed(range(len(episode))):
            state_t, action_t, reward_t1 = episode[t]
            G = gamma * G + reward_t1

            N[state_t][action_t] += 1
            alpha = 1 / N[state_t][action_t]
            Q[state_t][action_t] += alpha * (G - Q[state_t][action_t])

            update_policy(state_t, Q, policy, epsilon, env)

    return Q, policy
