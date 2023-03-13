import flwr as fl
from dataset_utils import get_cifar_10, do_fl_partitioning

from client import FlowerClient, fit_config, get_evaluate_fn

from client_manager import RLManager

if __name__ == "__main__":
    # parse input arguments
    NUM_CPUS = 0.1
    NUM_ROUNDS = 3
    pool_size = 100 # number of dataset partions (= number of total clients)
    client_resources = {
        "num_cpus": NUM_CPUS
    }  # each client will get allocated 1 CPUs

    # Download CIFAR-10 dataset
    train_path, testset = get_cifar_10()

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    fed_dir = do_fl_partitioning(
        train_path, pool_size=pool_size, alpha=1000, num_classes=10, val_ratio=0.1
    )

    # configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(testset),  # centralised evaluation of global model
    )


    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(cid, fed_dir)


    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": True}

    server = fl.server.Server(strategy=strategy, client_manager=RLManager())
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
