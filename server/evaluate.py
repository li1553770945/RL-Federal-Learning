import sys

from flwr.common import Scalar

sys.path.append("..")

import torchvision
import torch
from typing import Callable, Optional, Dict, Tuple, Any

from network import Net, set_params, test
import flwr as fl


def get_evaluate_fn(
        testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ) -> tuple[Any, dict[str, Any]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = Net()
        set_params(model, parameters)
        model.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate
