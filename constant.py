S_CONV = 2  # 卷积层数量
S_FC = 3  # 全连接层数量
S_RC = 0  # 循环层数量
BATCH_SIZE = 32  # batch_size
LOCAL_EPOCH = 3  # 本地训练轮数

RANDOM_SELECT_RATE = 0.1  # 多大的概率随机选择参与的客户端
QLEARNING_EPSILON = 0.1  # 多大的概率随机选择动作
PARTICIPANT_DEVICES = 30  # 每次选择的客户端数量

NUM_CLIENTS = 100  # 一共多少个客户端
NUM_CPUS = 0.1  # 每个客户端占用的CPU资源，根据ray的定义
NUM_ROUNDS = 50  # 总共训练几轮

REWARD_ACC_RATE = 1  # 最终计算奖励时，精确率的系数,论文中的alpha
REWARD_ACC_IMPROVE_RATE = 10  # 最终计算奖励时，精确率提升所占的系统，论文中的beta

LOW_PERFORMANCE_RATE = 0.3  # 低端设备所占的百分比
NORMAL_PERFORMANCE_RATE = 0.4  # 中端设备所占的百分比
HIGH_PERFORMANCE_RATE = 0.3  # 高端设备所占的百分比
