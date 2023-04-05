import numpy as np
import matplotlib.pyplot as plt
from typing import List

if __name__ == "__main__":
    target = "accuracy"
    labels = ["qlearning", "random"]
    color = ['r', 'b']

    for i in range(0, len(labels)):
        label = labels[i]
        data = np.load("log/{}_{}.npy".format(target, label))
        data = data.tolist()
        x = [x for x in range(0, len(data))]
        plt.plot(x, data, lw=2, ls='-', c=color[i], alpha=1, label=label)

    plt.legend()
    plt.plot()
    plt.xlabel("round")
    plt.show()


