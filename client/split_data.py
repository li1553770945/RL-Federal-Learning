import time
import sys

sys.path.append("..")
from dataset_utils import get_cifar_10,do_fl_partitioning
import flwr as fl
from client import FlowerClient


if __name__ == "__main__":

    num_clients = 10
    train_path, testset = get_cifar_10()
    fed_dir = do_fl_partitioning(
        train_path, pool_size=num_clients, alpha=1000, num_classes=10, val_ratio=0.1
    )
    print(fed_dir)

