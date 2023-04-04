import numpy as np
import matplotlib.pyplot as plt
from typing import List
if __name__ == "__main__":
    energy = np.load("../log/energy_qlearning.npy")
    energy = energy.tolist()
    acc = np.load("../log/accuracy_qlearning.npy")
    acc = acc.tolist()
    x = [x for x in range(0,len(energy))]
    plt.plot(x, acc, lw=2, ls='-', c='r', alpha=1,label="acc")
    plt.plot(x, energy, lw=2, ls='-', c='b', alpha=1,label="energy")
    plt.legend()
    plt.plot()
    plt.xlabel("round")
    plt.show()


