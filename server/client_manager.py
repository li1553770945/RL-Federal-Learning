from flwr.server import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
import random
import threading
from typing import Dict, Optional, List, Tuple
from flwr.common.logger import log
from logging import INFO
from server.q_learning import QLearning
from constant import RANDOM_SELECT_RATE


class RLManager(SimpleClientManager):

    def __init__(self, cid2q:Dict[str, QLearning] , method: str) -> None:
        super(RLManager, self).__init__()
        self.cid2q = cid2q
        self.method = method

    def sample(
            self,
            num_clients: int,
            min_num_clients: Optional[int] = None,
            criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []
        if random.uniform(0, 1) < RANDOM_SELECT_RATE or self.method == "random":
            sampled_cids = random.sample(available_cids, num_clients)
            log(INFO, "select {} device based randomly".format(num_clients))
        elif self.method == "qlearning":
            maxqs: List[Tuple[str, int]] = list()  # 包含客户端id和最大q值的列表
            for cid in available_cids:
                maxqs.append((cid, self.cid2q[cid].get_max_q()))
            maxqs_sorted = sorted(maxqs, key=lambda x: x[1], reverse=True)  # 排序
            log(INFO, "select {} device based on max q".format(num_clients))
            sampled_cids = [x[0] for x in maxqs_sorted[:num_clients]]  # 选择最大的num_clients个
        elif self.method == "performance":
            max_performance: List[Tuple[str, int]] = list()  # 包含客户端id和performance的列表，注意performance越小性能越好
            for cid in available_cids:
                max_performance.append((cid, self.cid2q[cid].performance))
            max_performance_sorted = sorted(max_performance, key=lambda x: x[1])  # 排序
            log(INFO, "select {} device based on max performance".format(num_clients))
            sampled_cids = [x[0] for x in max_performance_sorted[:num_clients]]  # 选择最大的num_clients个
        return [self.clients[cid] for cid in sampled_cids]
