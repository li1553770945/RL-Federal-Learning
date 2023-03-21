import numpy as np
import random
from server.state import ClientState
from typing import Dict
from collections import defaultdict


class QLearning:
    def __init__(self):
        # 定义动作空间大小
        self.n_actions = 5

        # 定义学习率
        self.learning_rate = 0.9

        # 定义折扣因子
        self.discount_factor = 0.1

        # 定义epsilon值，用于epsilon-greedy策略
        self.epsilon = 0.1

        # 定义Q-table并初始化为0
        self.Q: Dict[ClientState, np.ndarray] = defaultdict(lambda: np.zeros(self.n_actions))
        self.state = ClientState()
        self.action = 0

        self.device_state = dict()

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

    def update(self, reward: int, new_state: ClientState = None) -> None:
        # 如果没有更新state，则还是当前的state
        new_state = self.state if new_state is None else new_state
        # 根据Q-learning更新规则更新Q值
        td_error = reward + self.discount_factor * np.max(self.Q[new_state]) - self.Q[self.state][self.action]
        self.Q[self.state][self.action] += self.learning_rate * td_error
        self.state = new_state

    def get_max_q(self) -> int:
        return np.max(self.Q[self.state])
