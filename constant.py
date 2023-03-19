S_CONV = 1  # 卷积层数量
S_FC = 2  # 全连接层数量
S_RC = 0  # 循环层数量
BATCH_SIZE = 64  # batch_size
LOCAL_EPOCH = 5  # 本地训练轮数

RANDOM_SELECT_RATE = 0.1  # 多大的概率随机选择参与的客户端

PARTICIPANT_DEVICES = 10  # 每次选择的客户端数量
NUM_CLIENTS = 100  # 一共多少个客户端
NUM_CPUS = 0.1  # 每个客户端占用的CPU资源，根据ray的定义
NUM_ROUNDS = 10  # 总共训练几轮

REWARD_ACC_RATE = 1  # 最终计算奖励时，精确率的系数
REWARD_ACC_IMPROVE_RATE = 1  # 最终计算奖励时，精确率提升所占的系统
