import sys

sys.path.append("..")
import flwr as fl
import torch
from typing import Dict
from flwr.common.typing import Scalar
from network import Net, train, test,get_params,set_params
from dataset_utils import get_dataloader
import ray

# Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str):
        self.cid = cid
        self.fed_dir = fed_dir_data
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.net = Net()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return get_params(self.net)

    def fit(self, parameters, config):
        print("train:{}".format(self.cid))
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = 0
        trainloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
        )

        # Send model to device
        self.net.to(self.device)

        # Train
        train(self.net, trainloader, epochs=config["epochs"], device=self.device)

        # Return local model and statistics
        return get_params(self.net), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = 0
        valloader = get_dataloader(
            self.fed_dir, self.cid, is_train=False, batch_size=1, workers=num_workers
        )

        # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = test(self.net, valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}



