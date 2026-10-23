"""Socket-free test/local transport using Flower's actual message/server APIs.

This is a same-process simulation, not OS-level data isolation. The server sees
ClientProxy objects and scalar/tensor responses, never a client's URL dataset.
"""
import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import SimpleClientManager

from src.server.flower_strategy import FederatedServer


class LocalClientProxy(ClientProxy):
    def __init__(self, cid, endpoint):
        super().__init__(str(cid))
        self._endpoint = endpoint.to_client()

    def get_properties(self, ins, timeout, group_id):
        return self._endpoint.get_properties(ins)

    def get_parameters(self, ins, timeout, group_id):
        return self._endpoint.get_parameters(ins)

    def fit(self, ins, timeout, group_id):
        return self._endpoint.fit(ins)

    def evaluate(self, ins, timeout, group_id):
        return self._endpoint.evaluate(ins)

    def reconnect(self, ins, timeout, group_id):
        return fl.common.DisconnectRes(reason="local simulation complete")


def run_local(strategy, endpoints):
    if len(endpoints) != 5:
        raise ValueError("exactly five local endpoints are required")
    manager = SimpleClientManager()
    for cid, endpoint in enumerate(endpoints):
        manager.register(LocalClientProxy(cid, endpoint))
    server = FederatedServer(client_manager=manager, strategy=strategy)
    # Sequential calls avoid shared RNG races and avoid parallel GPU allocation.
    server.max_workers = 1
    history, seconds = server.fit(strategy.cfg["rounds"], timeout=300)
    return server, history, seconds
