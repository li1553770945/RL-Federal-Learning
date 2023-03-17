S_CONV = 1 # 卷积层数量
S_FC = 2 #  全连接层数量
S_RC = 0 #  循环层数量
BATCH_SIZE = 64 # batch_size
LOCAL_EPOCH = 5 # 本地训练轮数

RANDOM_SELECT_RATE = 0.1 # 多大的概率随机选择参与的客户端

PARTICIPANT_DECICES = 10 # 每次选择的客户端数量
NUM_CLIENTS = 100 # 一共多少个客户端
NUM_CPUS = 0.1 # 每个客户端占用的CPU资源，根据ray的定义
NUM_ROUNDS = 3 # 总共训练几轮