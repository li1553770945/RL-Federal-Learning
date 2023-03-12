import numpy as np
import random
from state import ClientState
class QLearning:
    def __init__(self):
        # 定义状态空间大小
        self.n_states = 10

        # 定义动作空间大小
        self.n_actions = 2

        # 定义学习率
        self.learning_rate = 0.1

        # 定义折扣因子
        self.discount_factor = 0.99

        # 定义epsilon值，用于epsilon-greedy策略
        self.epsilon = 0.1

        # 定义Q-table并初始化为0
        self.Q = np.zeros((self.n_states, self.n_actions))

    def epsilon_greedy(self,Q, state, epsilon):
        if random.uniform(0, 1) < epsilon:
            # 随机选择动作
            action = np.random.randint(self.n_actions)
        else:
            # 选择当前状态下具有最高Q值的动作
            action = np.argmax(Q[state, :])
        return action

    def update(self):
        # 执行Q-learning算法
        for i in range(1000):
            # 初始化状态
            state = 0

            # 循环直到到达终止状态
            while state < self.n_states - 1:
                # 选择动作
                action = self.epsilon_greedy(self.Q, state, self.epsilon)

                # 执行动作并观察下一个状态和奖励
                next_state = state + action
                reward = 0 if next_state < self.n_states - 1 else 1

                # 根据Q-learning更新规则更新Q值
                td_target = reward + self.discount_factor * np.max(self.Q[next_state, :])
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.learning_rate * td_error

                # 更新状态
                state = next_state


