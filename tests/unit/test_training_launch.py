from pathlib import Path
from unittest.mock import patch
import subprocess
import pytest

from aster.training.launch import DistributedEnvironment, LaunchConfig, launch


def test_environment_complete_and_consistent():
    assert DistributedEnvironment.from_mapping({}).world_size == 1
    values = {
        "RANK": "3",
        "WORLD_SIZE": "4",
        "LOCAL_RANK": "1",
        "LOCAL_WORLD_SIZE": "2",
        "MASTER_ADDR": "10.0.0.1",
        "MASTER_PORT": "29500",
    }
    assert DistributedEnvironment.from_mapping(values).rank == 3
    with pytest.raises(ValueError):
        DistributedEnvironment.from_mapping({"RANK": "0"})
    with pytest.raises(ValueError):
        DistributedEnvironment.from_mapping({**values, "LOCAL_RANK": "0"})
    with pytest.raises(ValueError):
        DistributedEnvironment.from_mapping({**values, "MASTER_PORT": "0"})
    with pytest.raises(ValueError):
        LaunchConfig(nnodes=2)


def test_launcher_preview_shell_false_and_failure_propagates():
    config = LaunchConfig(nproc_per_node=2, timeout_seconds=17)
    script = Path(__file__)
    arguments = ["--value", "a space; not a shell"]
    with patch("aster.training.launch.subprocess.run") as run:
        command = launch(config, script, arguments)
        run.assert_not_called()
        assert command[-2:] == arguments and "--rdzv-conf=timeout=17" in command
        launch(config, script, arguments, execute=True)
        assert run.call_args.args[0] == command
        assert run.call_args.kwargs["shell"] is False and run.call_args.kwargs["check"] is True
        assert run.call_args.kwargs["env"]["ASTER_DISTRIBUTED_TIMEOUT"] == "17"
        run.side_effect = subprocess.CalledProcessError(2, command)
        with pytest.raises(subprocess.CalledProcessError):
            launch(config, script, execute=True)


def test_legacy_store_is_an_explicit_separate_torchrun_backend():
    config = LaunchConfig(nproc_per_node=2, store_backend="legacy_tcp")
    command = config.command(__file__)
    assert command[2] == "aster.training.torchrun_compat"
    assert "--rdzv-backend=aster_static_tcp" in command and "--max-restarts=0" in command
    assert "--rdzv-conf=rank=0,timeout=180" in command
    with patch("aster.training.launch.subprocess.run") as run:
        launch(config, __file__, execute=True)
        assert run.call_args.kwargs["env"]["USE_LIBUV"] == "0"
        assert run.call_args.kwargs["env"]["ASTER_DISTRIBUTED_STORE_BACKEND"] == "legacy_tcp"
    with pytest.raises(ValueError):
        LaunchConfig(launcher="native", store_backend="legacy_tcp")


def test_legacy_static_store_uses_real_tcp_and_validates_before_binding():
    import socket
    from torch.distributed import TCPStore
    from torch.distributed.elastic.rendezvous import RendezvousParameters
    from aster.training.torchrun_compat import LegacyStaticRendezvous

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    def parameters(**values):
        return RendezvousParameters("aster_static_tcp", f"127.0.0.1:{port}", "test", 1, 1, **values)

    with pytest.raises(ValueError):
        LegacyStaticRendezvous(parameters())
    with pytest.raises(ValueError):
        LegacyStaticRendezvous(parameters(rank=2))
    server = LegacyStaticRendezvous(parameters(rank=0, timeout=10))
    info = server.next_rendezvous()
    info.store.set("probe", "ready")
    client = TCPStore("127.0.0.1", port, 1, False, use_libuv=False)
    assert client.get("test/probe") == b"ready"
    assert not server._store.libuvBackend
    assert server.next_rendezvous().store.get("probe") == b"ready"
