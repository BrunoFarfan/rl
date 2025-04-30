import random
from collections import defaultdict

import numpy as np
from Environments.AbstractEnv import AbstractEnv


class MonteCarloControl:
    def __init__(self, env, epsilon=0.1, gamma=1.0):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: {a: 0.0 for a in env.action_space})
        self.returns = defaultdict(list)
        self.policy = defaultdict(
            lambda: {a: 1.0 / len(env.action_space) for a in env.action_space}
        )
        self.visit_counts = defaultdict(int)

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        done = False

        while not done:
            probs = list(self.policy[state].values())
            actions = list(self.policy[state].keys())
            action = random.choices(actions, weights=probs)[0]
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state

        return episode

    def update(self, episode):
        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward

            if (state, action) not in visited:
                visited.add((state, action))
                self.returns[(state, action)].append(G)
                self.Q[state][action] = np.mean(self.returns[(state, action)])
                self.visit_counts[state] += 1

                # Política greedy con ε-soft
                best_action = max(self.Q[state], key=self.Q[state].get)
                nA = len(self.env.action_space)
                for a in self.env.action_space:
                    if a == best_action:
                        self.policy[state][a] = 1 - self.epsilon + self.epsilon / nA
                    else:
                        self.policy[state][a] = self.epsilon / nA

    def train(self, num_episodes):
        for i in range(1, num_episodes + 1):
            episode = self.generate_episode()
            self.update(episode)
            if i % 10000 == 0:
                print(f'Entrenado en {i} episodios')

    def get_policy(self):
        return self.policy

    def get_state_visits(self):
        return self.visit_counts


class MonteCarloControlEveryVisit:
    def __init__(self, env: AbstractEnv, epsilon=0.1, gamma=1.0):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = defaultdict(lambda: {a: 0.0 for a in env.action_space})
        self.N = defaultdict(lambda: {a: 0 for a in env.action_space})
        self.policy = defaultdict(
            lambda: {a: 1.0 / len(env.action_space) for a in env.action_space}
        )
        self.visit_counts = defaultdict(int)

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        done = False

        while not done:
            probs = list(self.policy[state].values())
            actions = list(self.policy[state].keys())
            action = random.choices(actions, weights=probs)[0]
            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state

        return episode

    def update(self, episode):
        G = 0

        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = self.gamma * G + reward

            self.N[state][action] += 1
            alpha = 1 / self.N[state][action]
            self.Q[state][action] += alpha * (G - self.Q[state][action])
            self.visit_counts[state] += 1

            best_action = max(self.Q[state], key=self.Q[state].get)
            nA = len(self.env.action_space)
            for a in self.env.action_space:
                if a == best_action:
                    self.policy[state][a] = 1 - self.epsilon + self.epsilon / nA
                else:
                    self.policy[state][a] = self.epsilon / nA

    def train(self, num_episodes):
        for i in range(1, num_episodes + 1):
            episode = self.generate_episode()
            self.update(episode)
            if i % 10000 == 0:
                print(f'Entrenado en {i} episodios')

    def get_policy(self):
        return self.policy

    def get_state_visits(self):
        return self.visit_counts
