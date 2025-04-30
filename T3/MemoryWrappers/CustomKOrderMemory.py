from collections import deque

from Environments.AbstractEnv import AbstractEnv


class CustomKOrderMemory(AbstractEnv):
    def __init__(self, env, memory_size):
        self.__env = env
        self.__memory_size = memory_size
        self.__buffer = deque(maxlen=memory_size)
        self.__env_state = None

    @property
    def action_space(self):
        return [(env_a, mem_a) for env_a in self.__env.action_space for mem_a in ('save', 'ignore')]

    def reset(self):
        self.__env_state = self.__env.reset()
        self.__buffer.clear()

        return self.__get_state()

    def __get_state(self):
        return self.__env_state, tuple(self.__buffer)

    def step(self, action):
        env_action, mem_action = action

        if mem_action == 'save':
            self.__buffer.append(self.__env_state)

        next_state, reward, done = self.__env.step(env_action)
        self.__env_state = next_state

        return self.__get_state(), reward, done

    def show(self):
        self.__env.show()
        print(f'Buffer: {self.__buffer}')
