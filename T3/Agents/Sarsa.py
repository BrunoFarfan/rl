import random
from collections import defaultdict
from typing import Any, Union, Sequence

State = Any
Action = Union[str, int]
Reward = Union[float, int]


class Sarsa:
    def __init__(self, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 1.0, initial_q_value: float = 0.0):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_table = defaultdict(lambda: defaultdict(lambda: initial_q_value))

    def get_action(self, state: State, action_space: list[Action]) -> Action:
        """Select an action using epsilon-greedy policy.

        If a random number is less than epsilon, select a random action.
        Otherwise, select the action with the highest Q-value.
        If there are multiple actions with the same Q-value, select one randomly.

        Args:
            state (State): The current state.
            action_space (list[str]): The list of possible actions.

        Returns:
            str: The selected action.
        """
        if random.random() < self.epsilon:
            return random.choice(action_space)
        q_values = self.q_table[state]
        max_value = max([q_values[a] for a in action_space])
        best_actions = [a for a in action_space if q_values[a] == max_value]
        return random.choice(best_actions)

    def update(self, state: State, action: Action, reward: Reward, next_state: State, next_action: Action, done: bool):
        """Perform Sarsa update.

        The Q-value for the current state-action pair is updated based on the received reward
        and the Q-value of the next state-action pair.

        Args:
            state (State): The current state.
            action (str): The action taken.
            reward (float): The reward received.
            next_state (State): The next state.
            next_action (str): The action taken in the next state.
            done (bool): Whether the episode has ended.
        """
        next_q = self.q_table[next_state][next_action] if not done else 0.0
        td_target = reward + self.gamma * next_q
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def get_q_table(self):
        return self.q_table


class NStepSarsa:
    def __init__(self, n: int, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 1.0, initial_q_value: float = 0.0):
        self.n = n
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_table = defaultdict(lambda: defaultdict(lambda: initial_q_value))

        # reseted at the start of each episode
        self.t = 0
        self.T = float("inf")
        self.states = [] # size n+1 ring
        self.actions = []
        self.rewards = []

    def start_episode(self) -> None:
        """Call once, right before starting a new episode.

        This resets the internal state of the agent.
        """
        self.t = 0
        self.T = float("inf")
        self.states = [None] * (self.n + 1)
        self.actions = [None] * (self.n + 1)
        self.rewards = [0.0] * (self.n + 1)

    def get_action(self, state: State, action_space: Sequence[Action]) -> Action:
        """Get action using epsilon-greedy policy

        If a random number is less than epsilon, select a random action.
        Otherwise, select the action with the highest Q-value.
        If there are multiple actions with the same Q-value, select one randomly.

        Args:
            state (State): The current state.
            action_space (Sequence[Action]): The list of possible actions.

        Returns:
            Action: The selected action.
        """
        if random.random() < self.epsilon:
            return random.choice(action_space)
        qv = self.q_table[state]
        max_q = max(qv[a] for a in action_space)
        best = [a for a in action_space if qv[a] == max_q]
        return random.choice(best)

    def step(self, state: State, action: Action, reward: Reward, done: bool = False) -> None:
        """Perform a step in the n-step Sarsa algorithm.

        This method should be called at each time step after observing the current state and action,
        taking the action, and receiving the reward. It updates the internal state and Q-values
        based on the observed transition.

        Args:
            state (State): The current state.
            action (Action): The action taken.
            reward (Reward): The reward received after taking the action.
            done (bool): Whether the episode has ended. If True, marks the terminal time T = t+1.
        """
        N = self.n + 1
        # store (S_t, A_t, R_{t+1})
        idx = self.t % N
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[(self.t + 1) % N] = reward

        if done:
            # episode length = T = t+1
            self.T = self.t + 1

        # tau is the time whose estimate is being updated
        tau = self.t - self.n + 1
        if tau >= 0:
            # form G_{tau:tau+n}
            G = 0.0
            # sum discounted rewards R_{tau+1} .. R_{min(tau+n, T)}
            upper = min(tau + self.n, self.T)
            for i in range(tau + 1, upper + 1):
                G += (self.gamma ** (i - tau - 1)) * self.rewards[i % N]

            # bootstrap if tau+n < T
            if tau + self.n < self.T:
                s_tp_n = self.states[(tau + self.n) % N]
                a_tp_n = self.actions[(tau + self.n) % N]
                G += (self.gamma ** self.n) * self.q_table[s_tp_n][a_tp_n]

            # update Q(S_tau, A_tau)
            s_tau = self.states[tau % N]
            a_tau = self.actions[tau % N]
            error = G - self.q_table[s_tau][a_tau]
            self.q_table[s_tau][a_tau] += self.alpha * error

        # move to next time
        self.t += 1

    def finish_episode(self) -> None:
        """Finish the episode and flush remaining updates.

        This method should be called once after the last step of the episode.
        It ensures that all remaining updates are performed up to tau = T-1.
        """
        N = self.n + 1
        # ensure T is set to the number of step() calls
        self.T = self.t
        # keep updating until tau = T−1
        while True:
            tau = self.t - self.n + 1
            if tau >= 0 and tau <= self.T - 1:
                # no bootstrap since tau+n >= T here
                G = 0.0
                for i in range(tau + 1, self.T + 1):
                    G += (self.gamma ** (i - tau - 1)) * self.rewards[i % N]

                s_tau = self.states[tau % N]
                a_tau = self.actions[tau % N]
                error = G - self.q_table[s_tau][a_tau]
                self.q_table[s_tau][a_tau] += self.alpha * error

            if tau == self.T - 1:
                break
            self.t += 1

    def get_q_table(self):
        return self.q_table
