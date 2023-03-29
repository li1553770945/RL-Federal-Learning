S_CONV = 2  # 卷积层数量
S_FC = 3  # 全连接层数量
S_RC = 0  # 循环层数量
BATCH_SIZE = 64  # batch_size
LOCAL_EPOCH = 5  # 本地训练轮数

RANDOM_SELECT_RATE = 0.1  # 多大的概率随机选择参与的客户端
QLEARNING_EPSILON = 0.1  # 多大的概率随机选择动作
PARTICIPANT_DEVICES = 30  # 每次选择的客户端数量
NUM_CLIENTS = 100  # 一共多少个客户端
NUM_CPUS = 0.1  # 每个客户端占用的CPU资源，根据ray的定义
NUM_ROUNDS = 100  # 总共训练几轮

REWARD_ACC_RATE = 1  # 最终计算奖励时，精确率的系数,论文中的alpha
REWARD_ACC_IMPROVE_RATE = 5  # 最终计算奖励时，精确率提升所占的系统，论文中的beta

ENERGY_BASE = 10
ENERGY_TIMES = [10, 9, 7, 4, 1]  # 对于每个action，能量消耗乘以的倍率
TIME_USE_TIMES = [1, 2, 3, 4, 5]  # 对于每个action，时间乘以的倍率
