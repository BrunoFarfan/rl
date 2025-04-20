import random
from collections import defaultdict
from typing import Any, Union

State = Any
Action = Union[str, int]
Reward = Union[float, int]


class QLearning:
    def __init__(self, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 1.0, initial_q_value: float = 0.0):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_table = defaultdict(lambda: defaultdict(lambda: initial_q_value))

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
        max_value = max([q_values[a] for a in action_space])
        best_actions = [a for a in action_space if q_values[a] == max_value]
        return random.choice(best_actions)

    def update(self, state: State, action: Action, reward: Reward, next_state: State, next_action_space: list[Action], done: bool):
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
        max_q_next = max([self.q_table[next_state][a] for a in next_action_space]) if not done else 0.0
        td_target = reward + self.gamma * max_q_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def multi_goal_update(self, state: State, action: Action, reward: Reward, next_state: State, next_action_space: list[Action], goals: list[State], done: bool):
        position, goal = state
        next_position, _ = next_state

        for possible_goal in goals:
            possible_state = (position, possible_goal)
            next_possible_state = (next_position, possible_goal)

            if possible_goal == goal:
                td_target = reward
                td_error = td_target - self.q_table[possible_state][action]
                self.q_table[possible_state][action] += self.alpha * td_error

            else:
                max_q_next = max([self.q_table[next_possible_state][a] for a in next_action_space]) if not done else 0.0
                td_target = self.gamma * max_q_next
                td_error = td_target - self.q_table[possible_state][action]
                self.q_table[possible_state][action] += self.alpha * td_error

    def get_q_table(self):
        return self.q_table
