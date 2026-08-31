import os
import time
import pytest

from aster.core import read_json
from aster.evaluation.generative import (
    DistributionProtocol,
    ExtractorPin,
    evaluate_media_directories,
)
from aster.evaluation.suites import EvaluationGrant


@pytest.mark.skipif(
    not os.environ.get("ASTER_APPROVED_GENERATIVE_EVAL"),
    reason="No explicitly approved local official source/weights/data; never auto-download",
)
def test_real_pinned_public_extractor(tmp_path):
    config = read_json(os.environ["ASTER_APPROVED_GENERATIVE_EVAL"])
    assert config["approved_local_execution"] is True
    arguments = dict(config["protocol"])
    arguments["extractor"] = ExtractorPin(**arguments["extractor"])
    protocol = DistributionProtocol(**arguments)
    grant = EvaluationGrant(
        protocol.id,
        ("official_evaluator", "torchscript_execution"),
        time.monotonic() + config["deadline_seconds"],
    )
    report = evaluate_media_directories(
        protocol,
        config["reference_root"],
        config["generated_root"],
        source_root=config["source_root"],
        weights_path=config["weights_path"],
        grant=grant,
        output_directory=tmp_path / "official-evaluation",
        device=config.get("device", "cpu"),
    )
    assert report["status"] == "ok", report
    assert set(report["metrics"]) == set(protocol.metrics)
    assert report["actual_reference_samples"] == report["expected_reference_samples"]
    assert report["actual_generated_samples"] == report["expected_generated_samples"]
