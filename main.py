import random

import flwr as fl
from server.dataset_utils import get_cifar_10, do_fl_partitioning
from client.client import FlowerClient, get_evaluate_fn
from server.q_learning import QLearning
from server.client_manager import RLManager
from constant import *
from strategy.fedavg import RLFedAvg
from typing import List, Dict
from server.server import RLServer, fit_config
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument("--method", type=str, default="qlearning", choices=['qlearning', 'random', 'performance'])


def generate_state() -> (List[QLearning], Dict[str, QLearning]):
    qs: List[QLearning] = list()
    cid2q: Dict[str, QLearning] = dict()

    network_bandwidth = np.random.normal(40, 30, NUM_CLIENTS)

    for i in range(0, NUM_CLIENTS):  # 高中低三种性能

        if i / NUM_CLIENTS < HIGH_PERFORMANCE_RATE:
            q = QLearning(0)
        elif i / NUM_CLIENTS < HIGH_PERFORMANCE_RATE + NORMAL_PERFORMANCE_RATE:
            q = QLearning(1)
        else:
            q = QLearning(2)

        q.state.network_bandwidth = network_bandwidth[i]

        qs.append(q)
        cid2q[str(i)] = q

    return qs, cid2q


def client_fn(cid: str):
    # create a single client instance
    return FlowerClient(cid, fed_dir)


def save(data: List, name: str):
    x = [x for x in range(0, NUM_ROUNDS)]
    fig = plt.figure(figsize=(4, 4), dpi=300)
    plt.plot(x, data, lw=2, ls='-', c='b', alpha=1)
    plt.plot()
    plt.xlabel("round")
    plt.ylabel(name)
    fig.savefig("log/"+name)

    np.save("log/{}.npy".format(name),np.array(data))



if __name__ == "__main__":
    args = parser.parse_args()
    method = args.method

    pool_size = NUM_CLIENTS  # number of dataset partions (= number of total clients)
    client_resources = {
        "num_cpus": NUM_CPUS
    }  # each client will get allocated 1 CPUs

    # Download CIFAR-10 dataset
    train_path, testset = get_cifar_10()

    qs, cid2q = generate_state()

    # part ition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    fed_dir = do_fl_partitioning(
        train_path, pool_size=pool_size, alpha=1000, num_classes=10, val_ratio=0.1
    )

    energy_list = list()
    acc_list = list()
    # configure the strategy
    strategy = RLFedAvg(
        fraction_fit=0,
        fraction_evaluate=0,
        min_fit_clients=PARTICIPANT_DEVICES,
        min_evaluate_clients=5,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(testset),  # centralised evaluation of global model
        cid2q=cid2q,
        method=method
    )

    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": True}

    server = RLServer(strategy=strategy, client_manager=RLManager(qs, method=method), cid2q=cid2q,acc_list=acc_list,energy_list=energy_list)
    server.set_max_workers(1)
    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        ray_init_args=ray_init_args,
        server=server,
    )
    save(acc_list, "accuracy_"+args.method)
    save(energy_list, "energy_"+args.method)
