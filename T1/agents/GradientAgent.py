import numpy as np
import random
from agents.BaseAgent import BaseAgent


class GradientBanditAgent(BaseAgent):
    def __init__(self, num_of_actions: int, alpha: float, use_baseline: bool = True):
        self.num_of_actions = num_of_actions
        self.alpha = alpha
        self.use_baseline = use_baseline

        self.H = np.zeros(num_of_actions, dtype=np.float64)
        self.R_bar = 0
        self.N = 0

    def get_action(self) -> int:
        exp_H = np.exp(self.H - np.max(self.H))
        softmax_probs = exp_H / np.sum(exp_H)

        return np.random.choice(self.num_of_actions, p=softmax_probs)

    def learn(self, action: int, reward: float) -> None:
        self.N += 1
        softmax_probs = np.exp(self.H - np.max(self.H)) / np.sum(np.exp(self.H - np.max(self.H)))

        if self.use_baseline:
            self.R_bar += (1 / self.N) * (reward - self.R_bar)

        baseline = self.R_bar if self.use_baseline else 0

        for a in range(self.num_of_actions):
            if a == action:
                self.H[a] += self.alpha * (reward - baseline) * (1 - softmax_probs[a])
            else:
                self.H[a] -= self.alpha * (reward - baseline) * softmax_probs[a]
