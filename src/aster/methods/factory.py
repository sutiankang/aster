"""Explicit objective construction; unknown algorithms fail rather than selecting a fallback."""

from .supervised import CrossEntropyObjective, SigmoidContrastiveObjective
from .generation import (
    FlowObjective,
    FlowPath,
    DiffusionObjective,
    DiffusionSchedule,
    EDMObjective,
    AutoencoderObjective,
)
from .actions import ACTObjective, PiActionObjective
from .world_model import WorldModelObjective
from .masked_diffusion import MaskedDiffusionObjective
from .mtp import MultiTokenPredictionObjective
from .video_generation import WanVideoObjective
from .groot import GrootFlowObjective
from .muzero import MuZeroObjective
from .stochastic_flow import GaussianFlowObjective
from .masked_autoencoding import MaskedAutoencodingObjective
from .meanflow import MeanFlowObjective
from .planet import PlaNetObjective
from .vmc import VMCVAEObjective, MDNRNNObjective
from .cosmos3 import (
    Cosmos3FlowObjective,
    Cosmos3VisualFlowObjective,
    Wan22AutoencoderObjective,
    Cosmos3AudioAutoencoderObjective,
)
from .cosmos_predict1 import CosmosPredict1Objective
from .genie import GenieVQObjective, GenieDynamicsObjective, GenieWorldObjective


def build_objective(configuration):
    options = dict(configuration)
    name = options.pop("name")
    if name == "lewm":
        from .lewm import LeWMObjective

        return LeWMObjective(**options)
    if name == "dspark":
        from .dspark import DSparkObjective

        return DSparkObjective(**options)
    if name == "flow":
        path = FlowPath(**options.pop("path", {}))
        return FlowObjective(path, **options)
    if name == "diffusion":
        schedule = DiffusionSchedule.create(**options.pop("schedule", {}))
        return DiffusionObjective(schedule, **options)
    if name == "cosmos3_avae2":
        for key, expected in (
            ("type", "cosmos3_avae2"),
            ("reconstruction", "unclipped_waveform_l1"),
            ("kl_definition", "source_twice_gaussian_kl_per_latent_frame"),
        ):
            if key in options and options.pop(key) != expected:
                raise ValueError(f"Cosmos3 AVAE2 objective metadata differs: {key}")
        return Cosmos3AudioAutoencoderObjective(**options)
    types = {
        "edm": EDMObjective,
        "autoencoder": AutoencoderObjective,
        "act": ACTObjective,
        "pi": PiActionObjective,
        "rssm": WorldModelObjective,
        "cross_entropy": CrossEntropyObjective,
        "siglip": SigmoidContrastiveObjective,
        "masked_diffusion": MaskedDiffusionObjective,
        "mtp": MultiTokenPredictionObjective,
        "wan_video": WanVideoObjective,
        "groot_flow": GrootFlowObjective,
        "muzero": MuZeroObjective,
        "gaussian_flow": GaussianFlowObjective,
        "drifting_mae": MaskedAutoencodingObjective,
        "meanflow": MeanFlowObjective,
        "planet": PlaNetObjective,
        "vmc_vae": VMCVAEObjective,
        "vmc_mdn": MDNRNNObjective,
        "cosmos3_flow": Cosmos3FlowObjective,
        "cosmos_predict1": CosmosPredict1Objective,
        "genie_vq": GenieVQObjective,
        "genie_dynamics": GenieDynamicsObjective,
        "genie_world": GenieWorldObjective,
        "wan22_vae": Wan22AutoencoderObjective,
        "cosmos3_visual_flow": Cosmos3VisualFlowObjective,
    }
    if name not in types:
        raise ValueError(
            f"Unknown objective: {name}; multirole methods use explicit Method lifecycle"
        )
    return types[name](**options)
