import  numpy as np

S_CONV = 1 # 卷积层数量
S_FC = 2

class ClientState:
    def __init__(self):
        self.S_CONV = S_CONV
        self.S_FC = S_FC

    def to_numpy(self):
        data = np.zeros((10,))
        pass
