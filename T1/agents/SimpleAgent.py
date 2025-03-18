from agents.BaseAgent import BaseAgent
import numpy as np
import random


class SimpleAgent(BaseAgent):
    def __init__(self, num_of_actions: int, epsilon: float):
        self.num_of_actions = num_of_actions
        self.epsilon = epsilon
        self.Q = np.zeros(num_of_actions)
        self.N = np.zeros(num_of_actions)

    def get_action(self) -> int:
        if random.random() <= self.epsilon:
            return random.randrange(self.num_of_actions)
        else:
            return np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
        
    def learn(self, action: int, reward: float) -> None:
        self.N[action] += 1
        self.Q[action] += (1 / self.N[action]) * (reward - self.Q[action])