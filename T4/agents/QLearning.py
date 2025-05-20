import random

import numpy as np
from FeatureExtractor import FeatureExtractor


class QLearning:
    def __init__(self, num_actions: int, epsilon: float, alpha: float, gamma: float):
        self.__num_actions = num_actions
        self.__epsilon = epsilon
        self.__alpha = alpha
        self.__gamma = gamma
        self.__feature_extractor = FeatureExtractor()
        self.__num_features = self.__feature_extractor.num_of_features
        self.__weights = np.zeros(self.__num_features)

    def sample_action(self, observation):
        if random.random() < self.__epsilon:
            return random.randrange(self.__num_actions)
        return self.argmax(observation)

    def argmax(self, observation):
        a_max = None
        q_max = float('-inf')
        for action in range(self.__num_actions):
            q_value = self.__get_q_estimate(observation, action)
            if q_value > q_max:
                q_max = q_value
                a_max = [action]
            elif q_value == q_max:
                a_max.append(action)
        return random.choice(a_max)

    def __get_q_estimate(self, observation, action):
        x = self.__feature_extractor.get_features(observation, action)
        return np.dot(self.__weights, x)

    def learn(self, observation, action, reward, next_observation, done):
        x_current = self.__feature_extractor.get_features(observation, action)
        q_current = np.dot(self.__weights, x_current)

        if done:
            td_target = reward
        else:
            q_next = max(
                self.__get_q_estimate(next_observation, a) for a in range(self.__num_actions)
            )
            td_target = reward + self.__gamma * q_next

        td_error = td_target - q_current

        self.__weights += self.__alpha * td_error * x_current
