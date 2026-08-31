"""Native training runtime without upstream Trainer or Engine delegation."""

from .trainer import Trainer, StepResult
from .parallel import (
    ParallelConfig,
    ParallelContext,
    Group,
    ColumnParallelLinear,
    RowParallelLinear,
    vocab_parallel_cross_entropy,
)
from .state import EMA
from .runtime_state import apply_runtime_state
from .sequence import (
    SequenceParallelMLP,
    gather_sequence,
    reduce_scatter_sequence,
    context_parallel_attention,
)
from .experts import ExpertParallelMLP
from .pipeline import PipelineStage, VirtualPipelineStage, PipelineObjective, PipelineLossSpec
from .ring import ring_context_parallel_attention
from .activation import checkpoint_activation
from .optim import Muon
from .muon import MuonWithAuxAdam, MuonFactory
from .fp8 import FP8Linear, FP8Recipe, FP8Quantizer
from .gtp import rematerialize_weights
from .causal_parallel import (
    TensorParallelCausalLM,
    TensorParallelCrossEntropyObjective,
    parallelize_causal_lm,
)
from .causal_pipeline import CausalPipelineStage, CausalPipelineCrossEntropyObjective
from .moe_parallel import (
    ExpertParallelCausalLM,
    ExpertParallelCrossEntropyObjective,
    parallelize_mixtral,
)
from .moe_tensor_parallel import (
    ExpertTensorParallelCausalLM,
    ExpertTensorParallelCrossEntropyObjective,
    parallelize_mixtral_tensor,
)

__all__ = [
    "Trainer",
    "StepResult",
    "ParallelConfig",
    "ParallelContext",
    "Group",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "vocab_parallel_cross_entropy",
    "EMA",
    "SequenceParallelMLP",
    "gather_sequence",
    "reduce_scatter_sequence",
    "context_parallel_attention",
    "ring_context_parallel_attention",
    "ExpertParallelMLP",
    "PipelineStage",
    "VirtualPipelineStage",
    "PipelineObjective",
    "PipelineLossSpec",
    "checkpoint_activation",
    "Muon",
    "FP8Linear",
    "FP8Recipe",
    "FP8Quantizer",
    "rematerialize_weights",
]
__all__ += [
    "TensorParallelCausalLM",
    "TensorParallelCrossEntropyObjective",
    "parallelize_causal_lm",
]
__all__ += ["CausalPipelineStage", "CausalPipelineCrossEntropyObjective"]
__all__ += ["ExpertParallelCausalLM", "ExpertParallelCrossEntropyObjective", "parallelize_mixtral"]
__all__ += [
    "ExpertTensorParallelCausalLM",
    "ExpertTensorParallelCrossEntropyObjective",
    "parallelize_mixtral_tensor",
]
__all__ += ["apply_runtime_state"]
__all__ += ["MuonWithAuxAdam", "MuonFactory"]
