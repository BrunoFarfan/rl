import numpy as np
import random
from agents.BaseAgent import BaseAgent


class ConstantAgent(BaseAgent):
    def __init__(self, num_of_actions: int, epsilon: float, alpha: float, initial_Q: float):
        self.num_of_actions = num_of_actions
        self.epsilon = epsilon
        self.alpha = alpha

        self.Q = np.ones(num_of_actions, dtype=np.float64) * float(initial_Q)
        self.N = np.zeros(num_of_actions, dtype=np.int32)

    def get_action(self) -> int:
        if random.random() <= self.epsilon:
            return random.randrange(self.num_of_actions)
        else:
            return np.random.choice(np.flatnonzero(self.Q == self.Q.max()))

    def learn(self, action: int, reward: float) -> None:
        self.N[action] += 1
        self.Q[action] += self.alpha * (reward - self.Q[action])
