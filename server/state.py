import numpy as np
from constant import *


class ClientState:
    def __init__(self,cid:str):
        self.cid = cid
        if S_CONV < 10:
            self.s_conv = 0  # Small (<10), medium (<20), large (<30), larger (>=40)
        elif S_CONV < 20:
            self.s_conv = 1
        elif S_CONV < 30:
            self.s_conv = 2
        else:
            self.s_conv = 3

        if S_FC < 10:
            self.s_fc = 0
        else:
            self.s_fc = 1

        if S_RC < 5:
            self.s_rc = 0
        elif S_RC < 10:
            self.s_rc = 1
        else:
            self.s_rc = 2

        if BATCH_SIZE < 8:
            self.s_batch_size = 0
        elif BATCH_SIZE < 32:
            self.s_batch_size = 1
        else:
            self.s_batch_size = 2

        if LOCAL_EPOCH < 5:
            self.s_epoch = 0
        elif LOCAL_EPOCH < 10:
            self.s_epoch = 1
        else:
            self.s_epoch = 2

        if NUM_CLIENTS < 10:
            self.s_participant_device = 0
        elif NUM_CLIENTS < 50:
            self.s_participant_device = 1
        else:
            self.s_participant_device = 2

        self.co_cpu = 0
        self.co_memory = 0
        self.network_bandwidth = 0
        self.data_class_num = 0  # 具有多少个类别的数据

    def __hash__(self):
        vector = np.zeros((10,))
        vector[0] = self.s_conv
        vector[1] = self.s_fc
        vector[2] = self.s_rc
        vector[3] = self.s_batch_size
        vector[4] = self.s_epoch
        vector[5] = self.s_participant_device
        if self.co_cpu == 0:
            vector[6] = 0
        elif self.co_cpu < 25:
            vector[6] = 1
        elif self.co_cpu < 75:
            vector[6] = 2
        else:
            vector[6] = 3

        if self.co_memory == 0:
            vector[7] = 0
        elif self.co_memory < 25:
            vector[7] = 1
        elif self.co_memory < 75:
            vector[7] = 2
        else:
            vector[7] = 3

        if self.network_bandwidth < 40:
            vector[8] = 0
        else:
            vector[8] = 1

        if self.data_class_num / NUM_CLASSES < 0.25:
            vector[9] = 0
        elif self.data_class_num != NUM_CLASSES:
            vector[9] = 1
        else:
            vector[9] = 2
        return hash(tuple(vector))  # 把向量转换成元组


