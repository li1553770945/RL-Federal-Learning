# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower server."""
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

import flwr.server
from flwr.common import (
    DisconnectRes,
    EvaluateRes,
    FitRes,
    Scalar,
)
from flwr.common.logger import log
from flwr.server import History
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from constant import *
from .q_learning import QLearning

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]


class RLServer(flwr.server.Server):
    """Flower server."""

    def __init__(
            self,
            *,
            client_manager: ClientManager,
            strategy: Strategy,
            cid2q: Dict[str, QLearning],

    ) -> None:
        super(RLServer, self).__init__(client_manager=client_manager, strategy=strategy)
        self.last_acc = 0
        self.cid2q = cid2q

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()
        total_energy = 0
        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                total_energy += self.update_qlearning(fit_reses=res_fit[2], acc=metrics_cen['accuracy'])
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s,total energy use:%d", elapsed,total_energy)
        return history

    def update_qlearning(self, fit_reses: FitResultsAndFailures, acc: float)->int:

        successes, fails = fit_reses
        r_energy_global = 0

        # 如果没有提升精度，奖励直接是R-100


        # 计算全局消耗能量
        for success in successes:
            client, fit_res = success
            metrics = fit_res.metrics
            Ecomp = metrics['Ecomp']
            Ecomm = metrics['Ecomm']
            r_energy_local = Ecomm + Ecomp
            r_energy_global += r_energy_local

        if acc < self.last_acc:
            R = int(acc * 100 - 100)
            for success in successes:
                client, _ = success
                self.cid2q[client.cid].update(reward=R)
            return r_energy_global

        r_acc = acc * 100
        r_acc_pre = self.last_acc * 100
        # 优化每个客户端
        for success in successes:
            client, fit_res = success
            metrics = fit_res.metrics
            Ecomp = metrics['Ecomp']
            Ecomm = metrics['Ecomm']
            r_energy_local = Ecomm + Ecomp
            reward = 0 - r_energy_global/PARTICIPANT_DEVICES - r_energy_local \
                     + REWARD_ACC_RATE * r_acc \
                     + REWARD_ACC_IMPROVE_RATE * (r_acc - r_acc_pre)

            self.cid2q[client.cid].update(reward=reward)

        self.last_acc = acc
        return r_energy_global


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": LOCAL_EPOCH,  # number of local epochs
        "batch_size": BATCH_SIZE,
    }
    return config
