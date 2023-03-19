from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    FitIns,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import copy
from server.q_learning import QLearning


# flake8: noqa: E501
class RLFedAvg(FedAvg):

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            cid2q: Dict[str, QLearning],
    ) -> None:
        super(RLFedAvg, self).__init__(fraction_fit=fraction_fit,
                                       fraction_evaluate=fraction_evaluate,
                                       min_fit_clients=min_fit_clients,
                                       min_evaluate_clients=min_evaluate_clients,
                                       min_available_clients=min_available_clients,
                                       evaluate_fn=evaluate_fn,
                                       on_fit_config_fn=on_fit_config_fn,
                                       on_evaluate_config_fn=on_evaluate_config_fn,
                                       accept_failures=accept_failures,
                                       initial_parameters=initial_parameters,
                                       fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                                       evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                                       )
        self.cid2q = cid2q

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        basic_config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            basic_config = self.on_fit_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        fits: List[Tuple[ClientProxy, FitIns]] = list()
        for client in clients:
            config = copy.deepcopy(basic_config)
            config['action'] = self.cid2q[client.cid].get_action()
            # config['co_running_cpu_use'] =
            # config['co_running_mem_use']
            ins = FitIns(parameters, config)
            fits.append((client, ins))
        return fits
