"""Separate visual understanding and generation encoders sharing one language backbone."""

from dataclasses import asdict, dataclass, field
from typing import ClassVar
import torch
from torch import nn
import torch.nn.functional as F
from aster.nn import LayerNorm
from aster.nn.attention import scaled_attention
from .config import LlamaConfig
from .decoder import CausalLM, configuration_key
from .serialization import LocalModelMixin
from .siglip import SigLIPVisionEmbeddings
from .vision import VisionOutput
from .multimodal import replace_image_tokens


@dataclass(frozen=True)
class JanusVisionConfig:
    architecture: ClassVar[str] = "janus_vision"
    hidden_size: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_channels: int = 3
    image_size: int = 8
    patch_size: int = 2
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    mlp_ratio: float = 2.0
    attention_bias: bool = True
    hidden_dropout_rate: float = 0.0
    projection_dim: int = 32
    projection_dropout: float = 0.0
    use_qk_norm: bool = False
    initializer_range: float = 0.02
    depth: int = 2
    num_image_tokens: int = 16

    def __post_init__(self):
        if (
            min(
                self.hidden_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.num_channels,
                self.image_size,
                self.patch_size,
                self.projection_dim,
                self.depth,
            )
            < 1
            or self.hidden_size % self.num_attention_heads
        ):
            raise ValueError("Invalid Janus vision dimensions")
        if (
            self.image_size % self.patch_size
            or self.num_image_tokens != (self.image_size // self.patch_size) ** 2
        ):
            raise ValueError("Janus visual token count must match patch grid")
        if (
            self.hidden_act != "gelu"
            or self.mlp_ratio <= 0
            or min(self.layer_norm_eps, self.initializer_range) <= 0
        ):
            raise ValueError("Unsupported Janus formula/numerics")
        if any(
            not 0 <= p < 1
            for p in (self.attention_dropout, self.hidden_dropout_rate, self.projection_dropout)
        ):
            raise ValueError("Invalid Janus dropout")
        if self.use_qk_norm:
            raise ValueError(
                "Janus use_qk_norm is not admitted: official current head/LayerNorm dimensions disagree"
            )

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


@dataclass(frozen=True)
class JanusVQConfig:
    architecture: ClassVar[str] = "janus_vq"
    embed_dim: int = 4
    num_embeddings: int = 16
    double_latent: bool = False
    latent_channels: int = 32
    in_channels: int = 3
    base_channels: int = 32
    channel_multiplier: tuple[int, ...] = (1, 2)
    num_res_blocks: int = 1
    dropout: float = 0.0
    initializer_range: float = 0.02
    num_patches: int = 4
    out_channels: int = 3
    projection_dim: int = 32
    num_hidden_layers: int = 2
    hidden_act: str = "gelu"
    image_token_embed_dim: int = 32
    beta: float = 0.25

    def __post_init__(self):
        object.__setattr__(self, "channel_multiplier", tuple(self.channel_multiplier))
        if (
            not self.channel_multiplier
            or min(
                self.embed_dim,
                self.num_embeddings,
                self.latent_channels,
                self.in_channels,
                self.base_channels,
                self.num_res_blocks,
                self.num_patches,
                self.out_channels,
                self.projection_dim,
                self.num_hidden_layers,
                self.image_token_embed_dim,
                *self.channel_multiplier,
            )
            < 1
        ):
            raise ValueError("Invalid Janus VQ dimensions")
        if self.base_channels % 32 or any(
            self.base_channels * m % 32 for m in self.channel_multiplier
        ):
            raise ValueError("Official Janus VQ uses 32 groups; channels must be divisible by 32")
        if (
            self.double_latent
            or self.hidden_act != "gelu"
            or self.initializer_range <= 0
            or self.beta < 0
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("Unsupported Janus VQ mode/numerics")

    @property
    def image_size(self):
        return self.num_patches * 2 ** (len(self.channel_multiplier) - 1)

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


@dataclass(frozen=True)
class JanusConfig:
    architecture: ClassVar[str] = "janus"
    text_config: LlamaConfig = field(default_factory=LlamaConfig)
    vision_config: JanusVisionConfig = field(default_factory=JanusVisionConfig)
    vq_config: JanusVQConfig = field(default_factory=JanusVQConfig)
    image_token_id: int = 31

    def __post_init__(self):
        if type(self.text_config) is not LlamaConfig:
            raise ValueError("Janus currently admits its official Llama text backbone")
        if (
            self.vision_config.projection_dim != self.text_config.hidden_size
            or self.vq_config.projection_dim != self.text_config.hidden_size
            or self.vq_config.image_token_embed_dim != self.text_config.hidden_size
        ):
            raise ValueError("Janus connector widths must match the language hidden width")
        if (
            self.vq_config.num_patches
            != self.vision_config.image_size // self.vision_config.patch_size
            or not 0 <= self.image_token_id < self.text_config.vocab_size
        ):
            raise ValueError("Invalid Janus token/grid contract")

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "text_config": self.text_config.to_dict(),
            "vision_config": self.vision_config.to_dict(),
            "vq_config": self.vq_config.to_dict(),
            "image_token_id": self.image_token_id,
        }


def _initialize(module, std):
    for item in module.modules():
        if isinstance(item, (nn.Linear, nn.Conv2d, nn.Embedding)):
            nn.init.normal_(item.weight, std=std)
            if getattr(item, "bias", None) is not None:
                nn.init.zeros_(item.bias)


class JanusAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.q_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=c.attention_bias)
        self.k_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=c.attention_bias)
        self.v_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=c.attention_bias)
        self.projection_layer = nn.Linear(c.hidden_size, c.hidden_size)

    def forward(self, hidden):
        b, s, h = hidden.shape
        c = self.config

        def split(proj):
            return proj(hidden).reshape(b, s, c.num_attention_heads, -1).transpose(1, 2)

        visible = torch.ones(b, 1, s, s, device=hidden.device, dtype=torch.bool)
        mixed = scaled_attention(
            split(self.q_proj),
            split(self.k_proj),
            split(self.v_proj),
            visible,
            dropout=c.attention_dropout,
            training=self.training,
        )
        output = self.projection_layer(mixed.transpose(1, 2).reshape(b, s, h))
        return F.dropout(output, c.projection_dropout, self.training)


class JanusVisionMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.fc1 = nn.Linear(c.hidden_size, int(c.hidden_size * c.mlp_ratio))
        self.fc2 = nn.Linear(int(c.hidden_size * c.mlp_ratio), c.hidden_size)

    def forward(self, hidden):
        hidden = F.dropout(F.gelu(self.fc1(hidden)), self.config.hidden_dropout_rate, self.training)
        return F.dropout(self.fc2(hidden), self.config.hidden_dropout_rate, self.training)


class JanusVisionLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = JanusAttention(c)
        self.mlp = JanusVisionMLP(c)
        self.layer_norm1 = LayerNorm(c.hidden_size, c.layer_norm_eps)
        self.layer_norm2 = LayerNorm(c.hidden_size, c.layer_norm_eps)

    def forward(self, hidden):
        hidden = hidden + self.self_attn(self.layer_norm1(hidden))
        return hidden + self.mlp(self.layer_norm2(hidden))


class JanusVisionModel(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embeddings = SigLIPVisionEmbeddings(config)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            JanusVisionLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.post_layernorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        _initialize(self, config.initializer_range)

    def forward(self, pixel_values, *, interpolate_pos_encoding=False, output_hidden_states=False):
        hidden = self.embeddings(pixel_values, interpolate_pos_encoding)
        states = [hidden] if output_hidden_states else None
        for layer in self.encoder.layers:
            hidden = layer(hidden)
            if states is not None:
                states.append(hidden)
        hidden = self.post_layernorm(hidden)
        return VisionOutput(
            hidden, self.post_layernorm(hidden[:, 0]), tuple(states) if states is not None else None
        )


class JanusAligner(nn.Module):
    def __init__(self, input_size, output_size, depth):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(output_size, output_size) for _ in range(depth - 1)
        )

    def forward(self, hidden):
        hidden = self.fc1(hidden)
        for layer in self.hidden_layers:
            hidden = layer(F.gelu(hidden))
        return hidden


class VQResidual(nn.Module):
    def __init__(self, c, input_channels, output_channels):
        super().__init__()
        self.dropout = c.dropout
        self.norm1 = nn.GroupNorm(32, input_channels, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, output_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        if input_channels != output_channels:
            self.nin_shortcut = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, hidden):
        residual = self.nin_shortcut(hidden) if hasattr(self, "nin_shortcut") else hidden
        hidden = self.conv1(F.silu(self.norm1(hidden)))
        return residual + self.conv2(
            F.dropout(F.silu(self.norm2(hidden)), self.dropout, self.training)
        )


class VQAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.q, self.k, self.v, self.proj_out = (nn.Conv2d(channels, channels, 1) for _ in range(4))

    def forward(self, hidden):
        normed = self.norm(hidden)
        b, c, h, w = hidden.shape
        q, k, v = (proj(normed).flatten(2) for proj in (self.q, self.k, self.v))

        weights = ((q.transpose(1, 2) @ k) * c**-0.5).softmax(-1)
        return hidden + self.proj_out((v @ weights.transpose(1, 2)).reshape(b, c, h, w))


class VQScale(nn.Module):
    def __init__(self, channels, down):
        super().__init__()
        self.down = down
        self.conv = nn.Conv2d(
            channels, channels, 3, stride=2 if down else 1, padding=0 if down else 1
        )

    def forward(self, hidden):

        hidden = (
            F.pad(hidden, (0, 1, 0, 1))
            if self.down
            else F.interpolate(hidden, scale_factor=2, mode="nearest")
        )
        return self.conv(hidden)


class VQMiddle(nn.Module):
    def __init__(self, c, channels):
        super().__init__()
        self.block_1, self.attn_1, self.block_2 = (
            VQResidual(c, channels, channels),
            VQAttention(channels),
            VQResidual(c, channels, channels),
        )

    def forward(self, hidden):
        return self.block_2(self.attn_1(self.block_1(hidden)))


class VQEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv_in = nn.Conv2d(c.in_channels, c.base_channels, 3, padding=1)
        self.down = nn.ModuleList()
        channels = c.base_channels
        for i, multiplier in enumerate(c.channel_multiplier):
            level = nn.Module()
            level.block = nn.ModuleList()
            level.attn = nn.ModuleList()
            output = c.base_channels * multiplier
            for _ in range(c.num_res_blocks):
                level.block.append(VQResidual(c, channels, output))
                channels = output
                if i == len(c.channel_multiplier) - 1:
                    level.attn.append(VQAttention(channels))
            if i != len(c.channel_multiplier) - 1:
                level.downsample = VQScale(channels, True)
            self.down.append(level)
        self.mid = VQMiddle(c, channels)
        self.norm_out = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_out = nn.Conv2d(channels, c.latent_channels, 3, padding=1)

    def forward(self, pixels):
        hidden = self.conv_in(pixels)
        for level in self.down:
            for i, block in enumerate(level.block):
                hidden = block(hidden)
                if len(level.attn):
                    hidden = level.attn[i](hidden)
            if hasattr(level, "downsample"):
                hidden = level.downsample(hidden)
        return self.conv_out(F.silu(self.norm_out(self.mid(hidden))))


class VQDecoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        channels = c.base_channels * c.channel_multiplier[-1]
        self.conv_in = nn.Conv2d(c.latent_channels, channels, 3, padding=1)
        self.mid = VQMiddle(c, channels)
        self.up = nn.ModuleList()
        for i in reversed(range(len(c.channel_multiplier))):
            level = nn.Module()
            level.block = nn.ModuleList()
            level.attn = nn.ModuleList()
            output = c.base_channels * c.channel_multiplier[i]
            for _ in range(c.num_res_blocks + 1):
                level.block.append(VQResidual(c, channels, output))
                channels = output
                if i == len(c.channel_multiplier) - 1:
                    level.attn.append(VQAttention(channels))
            if i:
                level.upsample = VQScale(channels, False)
            self.up.append(level)
        self.norm_out = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_out = nn.Conv2d(channels, c.out_channels, 3, padding=1)

    def forward(self, latents):
        hidden = self.mid(self.conv_in(latents))
        for level in self.up:
            for i, block in enumerate(level.block):
                hidden = block(hidden)
                if len(level.attn):
                    hidden = level.attn[i](hidden)
            if hasattr(level, "upsample"):
                hidden = level.upsample(hidden)
        return self.conv_out(F.silu(self.norm_out(hidden)))


@dataclass
class VQOutput:
    last_hidden_state: torch.Tensor
    quantized_last_hidden_state: torch.Tensor
    image_tokens: torch.Tensor
    commitment_errors: torch.Tensor
    codebook_errors: torch.Tensor


class VectorQuantizer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.embedding = nn.Embedding(c.num_embeddings, c.embed_dim)

    def forward(self, latents):
        z = latents.permute(0, 2, 3, 1).contiguous()
        flat = z.reshape(-1, z.shape[-1])
        embeddings = self.embedding.weight
        distances = (
            flat.square().sum(-1, keepdim=True)
            + embeddings.square().sum(-1)
            - 2 * (flat @ embeddings.T)
        )
        indices = distances.argmin(-1)
        quantized = self.embedding(indices).reshape_as(z)

        commitment, codebook = (quantized.detach() - z).square(), (quantized - z.detach()).square()
        straight_through = z + (quantized - z).detach()
        return (
            straight_through.permute(0, 3, 1, 2).contiguous(),
            indices.reshape(z.shape[:3]),
            commitment,
            codebook,
        )


class JanusVQModel(LocalModelMixin, nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder, self.decoder = VQEncoder(config), VQDecoder(config)
        self.quantize = VectorQuantizer(config)
        self.quant_conv = nn.Conv2d(config.latent_channels, config.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.embed_dim, config.latent_channels, 1)
        _initialize(self, config.initializer_range)

    def encode(self, pixel_values):
        c = self.config
        if pixel_values.ndim != 4 or pixel_values.shape[1:] != (
            c.in_channels,
            c.image_size,
            c.image_size,
        ):
            raise ValueError("VQ pixels must match the declared image/code grid")
        hidden = self.encoder(pixel_values)
        quantized, indices, commitment, codebook = self.quantize(self.quant_conv(hidden))
        return VQOutput(hidden, quantized, indices.flatten(1), commitment, codebook)

    def decode(self, image_tokens):
        c = self.config
        if image_tokens.ndim != 2 or image_tokens.shape[1] != c.num_patches**2:
            raise ValueError("Discrete image token count must match VQ grid")

        embeddings = F.normalize(self.quantize.embedding(image_tokens), dim=-1)
        latents = embeddings.reshape(
            image_tokens.shape[0], c.num_patches, c.num_patches, c.embed_dim
        ).permute(0, 3, 1, 2)
        return self.decoder(self.post_quant_conv(latents))

    def reconstruct(self, pixel_values):
        """Use a straight-through VQ path so reconstruction gradients reach the encoder;
        argmin itself is not differentiated."""
        result = self.encode(pixel_values)
        return self.decoder(self.post_quant_conv(result.quantized_last_hidden_state)), result

    def forward(self, pixel_values):
        result = self.encode(pixel_values)
        return self.decode(result.image_tokens), result


class JanusGenerationHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.proj_out = nn.Linear(c.image_token_embed_dim, c.projection_dim)
        self.vision_head = nn.Linear(c.projection_dim, c.num_embeddings)

    def forward(self, hidden):
        return self.vision_head(F.gelu(self.proj_out(hidden)))


class JanusForConditionalGeneration(CausalLM):
    def __init__(self, config):
        nn.Module.__init__(self)
        text = CausalLM(config.text_config)
        self.config, self.model_key = config, configuration_key(config)
        self.model = nn.Module()
        self.model.language_model = text.model
        self.lm_head = text.lm_head
        self.model.vision_model = JanusVisionModel(config.vision_config)
        self.model.aligner = JanusAligner(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            config.vision_config.depth,
        )
        self.model.vqmodel = JanusVQModel(config.vq_config)
        self.model.generation_embeddings = nn.Embedding(
            config.vq_config.num_embeddings, config.vq_config.embed_dim
        )
        self.model.generation_aligner = JanusAligner(
            config.vq_config.embed_dim,
            config.vq_config.projection_dim,
            config.vq_config.num_hidden_layers,
        )
        self.model.generation_head = JanusGenerationHead(config.vq_config)
        for module in (
            self.model.aligner,
            self.model.generation_embeddings,
            self.model.generation_aligner,
            self.model.generation_head,
        ):
            _initialize(module, config.text_config.initializer_range)

    def get_decoder(self):
        return self.model.language_model

    @property
    def decoder_config(self):
        return self.config.text_config

    def get_image_features(self, pixels):
        return self.model.aligner(self.model.vision_model(pixels).last_hidden_state)

    def prepare_embeddings_for_image_generation(self, image_tokens):
        return self.model.generation_aligner(self.model.generation_embeddings(image_tokens))

    def decode_image_tokens(self, image_tokens):

        return self.model.vqmodel.decode(image_tokens)

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        pixel_values=None,
        image_token_mask=None,
        attention_mask=None,
        position_ids=None,
        state=None,
        use_cache=False,
        output_hidden_states=False,
        output_kind="text",
    ):
        if output_kind not in {"text", "image_codes"}:
            raise ValueError("Janus output_kind must be text or image_codes")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one input token/embedding tensor")
        hidden = self.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        mask = (
            input_ids == self.config.image_token_id if input_ids is not None else image_token_mask
        )
        if (
            image_token_mask is not None
            and input_ids is not None
            and not torch.equal(mask, image_token_mask)
        ):
            raise ValueError("Explicit Janus visual mask disagrees with token IDs")
        if pixel_values is not None:
            if state is not None or mask is None:
                raise ValueError("Images need explicit prefill placeholders and a fresh state")
            hidden = replace_image_tokens(hidden, self.get_image_features(pixel_values), mask)
        elif mask is not None and mask.any():
            raise ValueError("Image placeholders need their pixels")
        result = super().forward(
            inputs_embeds=hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            state=state,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states or output_kind == "image_codes",
        )
        if output_kind == "image_codes":
            result.logits = self.model.generation_head(result.hidden_states[-1])
            if not output_hidden_states:
                result.hidden_states = None
        result.auxiliary = {**(result.auxiliary or {}), "output_kind": output_kind}
        return result
