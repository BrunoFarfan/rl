import math
import random
from collections import defaultdict
from typing import Any, Union

State = Any
Action = Union[str, int, tuple]
Reward = Union[float, int]


class Dyna:
    def __init__(
        self,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        initial_q_value: float = 0.0,
        planning_steps: int = 5,
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_table = defaultdict(lambda: defaultdict(lambda: initial_q_value))
        self.planning_steps = planning_steps

        self.model = {}

        self.state_action_map = defaultdict(set)

    def get_action(self, state: State, action_space: list[Action]) -> Action:
        """Select an action using ε-greedy policy.

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
        max_q = max([q_values[a] for a in action_space])
        best_actions = [a for a in action_space if q_values[a] == max_q]
        return random.choice(best_actions)

    def update(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: State,
        next_action_space: list[Action],
        done: bool,
    ):
        """Perform Q-learning update.

        The Q-value for the current state-action pair is updated based on the received reward
        and the maximum Q-value of the next state.

        Args:
            state (State): The current state.
            action (str): The action taken.
            reward (float): The reward received.
            next_state (State): The next state.
            next_action_space (list[str]): The list of possible actions in the next state.
            done (bool): Whether the episode has ended.

        """
        max_q_next = (
            max([self.q_table[next_state][a] for a in next_action_space]) if not done else 0.0
        )
        td_target = reward + self.gamma * max_q_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

        self._update_model(state, action, reward, next_state, done)

        self._plan(next_action_space)

    def _update_model(
        self, state: State, action: Action, reward: Reward, next_state: State, done: bool
    ):
        key = (state, action)
        self.model[key] = (reward, next_state, done)
        self.state_action_map[state].add(action)

    def _plan(self, action_space: list[Action]):
        for _ in range(self.planning_steps):
            state = random.choice(list(self.state_action_map.keys()))
            action = random.choice(list(self.state_action_map[state]))
            reward, next_state, done = self.model[(state, action)]

            if done:
                max_q_next = 0.0

            else:
                max_q_next = max([self.q_table[next_state][a] for a in action_space])

            td_target = reward + self.gamma * max_q_next
            td_error = td_target - self.q_table[state][action]

            self.q_table[state][action] += self.alpha * td_error


class RMax:
    def __init__(
        self,
        gamma: float = 1.0,
        k: int = 5,
        r_max: float = 1.0,
        theta: float = 1e-6,
        max_iter: int = 1000,
    ):
        self.gamma = gamma
        self.k = k
        self.r_max = r_max
        self.theta = theta
        self.max_iter = max_iter

        self.N_t = defaultdict(int)
        self.N_p = defaultdict(lambda: defaultdict(int))
        self.states = set()
        self.terminals = set()

        self.V = {}

    def get_action(self, state: State, action_space: list[Action]) -> Action:
        self.states.add(state)

        self._plan(action_space)

        best_a, best_val = None, -math.inf
        for a in action_space:
            key = (state, a)
            if self.N_t[key] < self.k:
                q = self.upper_bound
            else:
                total = self.N_t[key]
                q = 0.0
                for (s2, r), cnt in self.N_p[key].items():
                    prob = cnt / total
                    q += prob * (r + self.gamma * self.V.get(s2, 0.0))
            if q > best_val:
                best_val, best_a = q, a

        return best_a

    def update(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: State,
        done: bool,
    ):
        self.states.add(state)
        self.states.add(next_state)
        if done:
            self.terminals.add(next_state)

        key = (state, action)
        self.N_t[key] += 1
        self.N_p[key][(next_state, reward)] += 1

    def _plan(self, action_space):
        if self.gamma < 1.0:
            upper_bound = self.r_max / (1.0 - self.gamma)
        else:
            upper_bound = self.r_max * self.k

        self.upper_bound = upper_bound

        for s in self.states:
            self.V[s] = 0.0 if s in self.terminals else upper_bound

        self._run_value_iteration(action_space)

    def _run_value_iteration(self, action_space: list[Action]):
        for _ in range(self.max_iter):
            delta = 0.0
            v_new = {}

            for s in self.states:
                if s in self.terminals:
                    v_new[s] = 0.0
                else:
                    best_q = -math.inf
                    for a in action_space:
                        key = (s, a)
                        if self.N_t[key] < self.k:
                            q = self.upper_bound
                        else:
                            total = self.N_t[key]
                            q = 0.0
                            for (s2, r), cnt in self.N_p[key].items():
                                prob = cnt / total
                                v_next = 0.0 if s2 in self.terminals else self.V[s2]
                                q += prob * (r + self.gamma * v_next)
                        best_q = max(best_q, q)
                    v_new[s] = best_q

                delta = max(delta, abs(self.V.get(s, 0.0) - v_new[s]))

            self.V = v_new
            if delta < self.theta:
                break
