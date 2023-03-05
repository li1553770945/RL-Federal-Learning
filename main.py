import flwr as fl
from client import fit_config, get_evaluate_fn,FlowerClient
from dataset_utils import get_cifar_10, do_fl_partitioning
from client_manager import RLManager

if __name__ == "__main__":
    num_clients = 5  # 总设备数量
    num_client_cpus = 1  # 每个设备的CPU数量
    num_rounds = 5 # 总共训练的几轮

    client_resources = {
        "num_cpus": num_client_cpus
    }

    train_path, testset = get_cifar_10()
    fed_dir = do_fl_partitioning(
        train_path, pool_size=num_clients, alpha=1000, num_classes=10, val_ratio=0.1
    )

    # configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(testset),  # centralised evaluation of global model
    )


    def client_fn(cid: str):
        # create a single client instance
        print("new client:{}".format(cid))
        return FlowerClient(cid, fed_dir)


    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
        client_manager=RLManager(),
    )
