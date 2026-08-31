"""Fixed-membership rendezvous using PyTorch's legacy TCPStore."""

from datetime import timedelta
from torch.distributed import TCPStore, PrefixStore
from torch.distributed.elastic.rendezvous import (
    RendezvousHandler,
    RendezvousInfo,
    RendezvousStoreInfo,
    rendezvous_handler_registry,
)
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint


class LegacyStaticRendezvous(RendezvousHandler):
    def __init__(self, params):
        self.address, self.port = parse_rendezvous_endpoint(params.endpoint, -1)
        self.rank = params.get_as_int("rank")
        self.world_size = params.max_nodes
        timeout = params.get_as_int("timeout", 180)
        if (
            params.min_nodes != params.max_nodes
            or type(self.rank) is not int
            or not 0 <= self.rank < self.world_size
            or not 0 < self.port < 65536
            or timeout < 1
            or not params.run_id
        ):
            raise ValueError(
                "Legacy static rendezvous needs fixed node count, rank, endpoint, timeout and run ID"
            )
        self.timeout = timedelta(seconds=timeout)
        self.run_id, self._store = params.run_id, None

    def get_backend(self):
        return "aster_static_tcp"

    @property
    def use_agent_store(self):
        return True

    def is_closed(self):
        return False

    def set_closed(self):
        pass

    def num_nodes_waiting(self):
        return 0

    def get_run_id(self):
        return self.run_id

    def shutdown(self):
        return True

    def next_rendezvous(self):
        if self._store is None:
            self._store = TCPStore(
                self.address,
                self.port,
                self.world_size,
                self.rank == 0,
                self.timeout,
                multi_tenant=True,
                use_libuv=False,
            )
        return RendezvousInfo(
            PrefixStore(self.run_id, self._store),
            self.rank,
            self.world_size,
            RendezvousStoreInfo(self.address, self.port),
        )


def main():

    from torch.distributed.run import main as torchrun_main

    rendezvous_handler_registry.register("aster_static_tcp", LegacyStaticRendezvous)
    torchrun_main()


if __name__ == "__main__":
    main()
