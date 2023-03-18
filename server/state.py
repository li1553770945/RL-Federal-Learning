import numpy as np
from constant import *


class ClientState:
    def __init__(self):
        if S_CONV < 10:
            self.CONV = 0  # Small (<10), medium (<20), large (<30), larger (>=40)
        elif S_CONV < 20:
            self.CONV = 1
        elif S_CONV < 30:
            self.CONV = 2
        else:
            self.CONV = 3

        if S_FC < 10:
            self.FC = 0
        else:
            self.FC = 1

        if S_RC < 5:
            self.RC = 0
        elif S_RC < 10:
            self.RC = 1
        else:
            self.RC = 2

        if BATCH_SIZE < 8:
            self.B = 0
        elif BATCH_SIZE < 32:
            self.B = 1
        else:
            self.B = 2

        if LOCAL_EPOCH < 5:
            self.E = 0
        elif LOCAL_EPOCH < 10:
            self.E = 1
        else:
            self.E = 2

        if NUM_CLIENTS < 10:
            self.K = 0
        elif NUM_CLIENTS < 50:
            self.K = 1
        else:
            self.K = 2

        self.CoCPU = 0
        self.CoMem = 0
        self.Network = 0
        self.Data = 0

    def __hash__(self):
        vector = np.zeros((10,))
        return hash(tuple(vector))  # 把向量转换成元组

    def set_co_cpu(self, utilization):
        if utilization == 0:
            self.CoCPU = 0
        elif utilization < 25:
            self.CoCPU = 1
