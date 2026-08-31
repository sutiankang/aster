"""Mamba-1 with input-dependent step sizes and state-space projections."""

from dataclasses import asdict, dataclass
from typing import ClassVar
import math
import torch
from torch import nn
import torch.nn.functional as F
from aster.core import TokenOutput
from aster.nn import RMSNorm
from aster.nn.ssm import MambaState, selective_scan
from .decoder import configuration_key
from .serialization import LocalModelMixin


@dataclass(frozen=True)
class MambaConfig:
    architecture: ClassVar[str] = "mamba"
    vocab_size: int = 32
    hidden_size: int = 32
    state_size: int = 4
    num_hidden_layers: int = 2
    layer_norm_epsilon: float = 1e-5
    expand: int = 2
    conv_kernel: int = 4
    use_bias: bool = False
    use_conv_bias: bool = True
    hidden_act: str = "silu"
    initializer_range: float = 0.1
    residual_in_fp32: bool = True
    time_step_rank: int | str = "auto"
    time_step_scale: float = 1.0
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_init_scheme: str = "random"
    time_step_floor: float = 1e-4
    rescale_prenorm_residual: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.time_step_rank == "auto":
            object.__setattr__(self, "time_step_rank", math.ceil(self.hidden_size / 16))
        if (
            not isinstance(self.time_step_rank, int)
            or min(
                self.vocab_size,
                self.hidden_size,
                self.state_size,
                self.num_hidden_layers,
                self.expand,
                self.conv_kernel,
                self.time_step_rank,
            )
            < 1
        ):
            raise ValueError("Invalid Mamba dimensions")
        if self.hidden_act != "silu" or self.time_step_init_scheme not in {"random", "constant"}:
            raise ValueError("Unsupported Mamba formula")
        if (
            min(
                self.layer_norm_epsilon,
                self.initializer_range,
                self.time_step_scale,
                self.time_step_min,
                self.time_step_floor,
            )
            <= 0
            or self.time_step_min > self.time_step_max
        ):
            raise ValueError("Invalid Mamba numerical/time-step scale")

    @property
    def intermediate_size(self):
        return self.expand * self.hidden_size

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


class MambaMixer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        width = c.intermediate_size
        self.conv1d = nn.Conv1d(width, width, c.conv_kernel, groups=width, bias=c.use_conv_bias)
        self.in_proj = nn.Linear(c.hidden_size, 2 * width, bias=c.use_bias)
        self.x_proj = nn.Linear(width, c.time_step_rank + 2 * c.state_size, bias=False)
        self.dt_proj = nn.Linear(c.time_step_rank, width)
        self.A_log = nn.Parameter(torch.empty(width, c.state_size))
        self.D = nn.Parameter(torch.ones(width))
        self.out_proj = nn.Linear(width, c.hidden_size, bias=c.use_bias)

    def initialize_dynamics(self):
        c = self.config
        with torch.no_grad():
            self.A_log.copy_(
                torch.arange(1, c.state_size + 1, device=self.A_log.device)
                .float()
                .log()
                .expand(c.intermediate_size, -1)
            )
            self.D.fill_(1)
            scale = c.time_step_rank**-0.5 * c.time_step_scale
            if c.time_step_init_scheme == "constant":
                self.dt_proj.weight.fill_(scale)
            else:
                self.dt_proj.weight.uniform_(-scale, scale)
            dt = (
                (
                    torch.rand_like(self.dt_proj.bias)
                    * (math.log(c.time_step_max) - math.log(c.time_step_min))
                    + math.log(c.time_step_min)
                )
                .exp()
                .clamp_min(c.time_step_floor)
            )

            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
            if c.rescale_prenorm_residual:
                self.out_proj.weight.div_(math.sqrt(c.num_hidden_layers))

    def forward(self, hidden, padding, previous):
        c = self.config
        b, length, _ = hidden.shape
        if padding is not None:
            hidden = hidden * padding[..., None].to(hidden.dtype)
        inputs, gate = self.in_proj(hidden).chunk(2, -1)
        history = (
            inputs.new_zeros(b, c.intermediate_size, c.conv_kernel - 1)
            if previous is None
            else previous[0]
        )
        concatenated = torch.cat((history, inputs.transpose(1, 2)), -1)

        convolved = F.silu(self.conv1d(concatenated)).transpose(1, 2)
        if padding is not None:
            convolved = convolved * padding[..., None].to(convolved.dtype)
        time, select_b, select_c = self.x_proj(convolved).split(
            (c.time_step_rank, c.state_size, c.state_size), -1
        )
        dt = F.softplus(F.linear(time, self.dt_proj.weight, self.dt_proj.bias))
        output, memory = selective_scan(
            convolved,
            dt,
            self.A_log,
            select_b,
            select_c,
            self.D,
            gate,
            previous[1] if previous is not None else None,
        )
        history = (
            concatenated[..., -(c.conv_kernel - 1) :]
            if c.conv_kernel > 1
            else concatenated[..., :0]
        )
        return self.out_proj(output.to(hidden.dtype)), (history, memory)


class MambaBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.norm, self.mixer = RMSNorm(c.hidden_size, c.layer_norm_epsilon), MambaMixer(c)

    def forward(self, hidden, padding, previous):
        residual = hidden.float() if self.config.residual_in_fp32 else hidden
        update, present = self.mixer(
            self.norm(hidden.to(self.norm.weight.dtype)), padding, previous
        )
        return residual + update, present


class MambaForCausalLM(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_key = configuration_key(config)
        self.backbone = nn.Module()
        self.backbone.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.backbone.layers = nn.ModuleList(
            MambaBlock(config) for _ in range(config.num_hidden_layers)
        )
        self.backbone.norm_f = RMSNorm(config.hidden_size, config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
                nn.init.normal_(module.weight, std=config.initializer_range)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        for block in self.backbone.layers:
            block.mixer.initialize_dynamics()
        if config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embeddings.weight

    def get_input_embeddings(self):
        return self.backbone.embeddings

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one input representation")
        if position_ids is not None:
            raise ValueError("Mamba has no absolute positional embedding; do not pass position_ids")
        hidden = self.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        c = self.config
        if hidden.ndim != 3 or hidden.shape[-1] != c.hidden_size or hidden.shape[1] == 0:
            raise ValueError("Invalid Mamba input shape")
        b, length, _ = hidden.shape
        seen = 0
        if state is not None:
            if (
                not isinstance(state, MambaState)
                or state.kind != "mamba_ssm"
                or state.model_key != self.model_key
                or len(state.layers) != c.num_hidden_layers
            ):
                raise ValueError("Mamba state type/model mismatch")
            seen = state.seen_tokens
            use_cache = True
            for conv, memory in state.layers:
                if conv.shape != (b, c.intermediate_size, c.conv_kernel - 1) or memory.shape != (
                    b,
                    c.intermediate_size,
                    c.state_size,
                ):
                    raise ValueError("Mamba conv/memory state layout mismatch")
        padding = None
        if attention_mask is not None:
            if (
                attention_mask.shape != (b, seen + length)
                or not ((attention_mask == 0) | (attention_mask == 1)).all()
            ):
                raise ValueError(
                    "Mamba padding mask must cover complete context with zero/one values"
                )
            padding = attention_mask[:, seen:]
        states, layers = [], []
        for i, block in enumerate(self.backbone.layers):
            hidden, present = block(hidden, padding, state.layers[i] if state is not None else None)
            if output_hidden_states:
                states.append(hidden)
            if use_cache:
                layers.append(present)
        hidden = self.backbone.norm_f(hidden)
        if output_hidden_states:
            states.append(hidden)
        updated = MambaState(tuple(layers), seen + length, self.model_key) if use_cache else None
        return TokenOutput(
            self.lm_head(hidden.to(self.lm_head.weight.dtype)).float(),
            updated,
            tuple(states) if output_hidden_states else None,
        )
