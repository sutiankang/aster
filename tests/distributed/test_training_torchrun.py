from pathlib import Path
import socket
import os
import pytest

from aster.training.launch import LaunchConfig, launch


@pytest.mark.parametrize("launcher", ["native", "torchrun"])
def test_real_launcher_two_cpu_processes(launcher):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    store_backend = "legacy_tcp" if os.name == "nt" and launcher == "torchrun" else "default"
    completed = launch(
        LaunchConfig(
            nproc_per_node=2,
            master_port=port,
            timeout_seconds=30,
            launcher=launcher,
            store_backend=store_backend,
        ),
        Path(__file__).with_name("launch_worker.py"),
        execute=True,
        capture_output=True,
    )
    assert "ASTER_LAUNCH_OK rank=0" in completed.stdout
    assert "ASTER_LAUNCH_OK rank=1" in completed.stdout
