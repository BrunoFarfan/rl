import numpy as np
from FeatureExtractor import FeatureExtractor


class ActorCritic:
    def __init__(self, gamma: float, alpha_v: float, alpha_pi: float):
        self.__alpha_v = alpha_v
        self.__alpha_pi = alpha_pi
        self.__gamma = gamma
        self.__feature_extractor = FeatureExtractor()
        self.__num_features = self.__feature_extractor.num_of_features
        self.__weights_v = np.zeros(self.__num_features)
        self.__theta_mu = np.zeros(self.__num_features)
        self.__theta_sigma = np.zeros(self.__num_features)
        self.__I = None

    def reset_episode_values(self):
        self.__I = 1.0

    def sample_action(self, observation):
        x = self.__feature_extractor.get_features(observation)
        mu = np.dot(self.__theta_mu, x)
        sigma = np.exp(np.dot(self.__theta_sigma, x))
        return np.random.normal(loc=mu, scale=sigma, size=None)

    def learn(self, observation, action, reward, next_observation, done):
        x = self.__feature_extractor.get_features(observation)
        v = np.dot(self.__weights_v, x)

        if done:
            v_next = 0.0
        else:
            x_next = self.__feature_extractor.get_features(next_observation)
            v_next = np.dot(self.__weights_v, x_next)

        # TD error
        delta = reward + self.__gamma * v_next - v

        # --------------------
        # Critic update
        self.__weights_v += self.__alpha_v * delta * x
        # --------------------

        # --------------------
        # Actor update (mu and sigma)
        mu = np.dot(self.__theta_mu, x)
        sigma = np.exp(np.dot(self.__theta_sigma, x))
        sigma_squared = sigma**2

        grad_log_mu = (action - mu) / sigma_squared * x
        grad_log_sigma = ((action - mu) ** 2 / sigma_squared - 1) * x

        self.__theta_mu += self.__alpha_pi * self.__I * delta * grad_log_mu
        self.__theta_sigma += self.__alpha_pi * self.__I * delta * grad_log_sigma
        # --------------------

        # Acumulador de descuento
        self.__I *= self.__gamma
