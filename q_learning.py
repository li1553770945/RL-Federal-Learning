import numpy as np
import random
from state import ClientState
from typing import Dict
from collections import defaultdict


class QLearning:
    def __init__(self):
        # 定义状态空间大小
        self.n_states = 5

        # 定义动作空间大小
        self.n_actions = 2

        # 定义学习率
        self.learning_rate = 0.1

        # 定义折扣因子
        self.discount_factor = 0.99

        # 定义epsilon值，用于epsilon-greedy策略
        self.epsilon = 0.1

        # 定义Q-table并初始化为0
        self.Q: Dict[ClientState, np.ndarray] = defaultdict(lambda: np.zeros(10))
        self.state = ClientState()
        self.action = 0

    def set_initial_state(self, state: ClientState) -> None:
        # 设置初始状态
        self.state = state

    def epsilon_greedy(self) -> int:
        if random.uniform(0, 1) < self.epsilon:
            # 随机选择动作
            action = np.random.randint(self.n_actions)
        else:
            # 选择当前状态下具有最高Q值的动作
            action = np.argmax(self.Q[self.state])
        return action

    def get_action(self) -> int:
        # 从当前状态选择一个action
        action = self.epsilon_greedy()
        self.action = action
        return action

    def update(self, new_state: ClientState, reward:int) -> None:
        # 执行Q-learning算法

        # 根据Q-learning更新规则更新Q值
        td_error = reward + self.discount_factor * np.max(self.Q[new_state]) - self.Q[self.state][self.action]
        self.Q[self.state][self.action] += self.learning_rate * td_error
        self.state = new_state


    def get_max_q(self) -> int:
        return np.max(self.Q[self.state])
