"""Architecture-specific configuration classes with explicit supported fields."""

from dataclasses import asdict, dataclass, field
from typing import ClassVar
from aster.nn import RopeConfig


@dataclass(frozen=True)
class LlamaConfig:
    architecture: ClassVar[str] = "llama"
    vocab_size: int = 32
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int | None = None
    max_position_embeddings: int = 128
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    rope: RopeConfig = field(default_factory=RopeConfig)

    def __post_init__(self):
        dimensions = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.max_position_embeddings,
        )
        if min(dimensions) < 1 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Invalid decoder dimensions/head ratios")
        if self.head_dim is None and self.hidden_size % self.num_attention_heads:
            raise ValueError("An implicit head_dim requires divisible hidden_size")
        if (
            self.attention_head_dim < 2
            or self.attention_head_dim % 2
            or self.rms_norm_eps <= 0
            or self.initializer_range <= 0
        ):
            raise ValueError("Invalid head_dim/normalization/initialization")
        if not 0 <= self.attention_dropout < 1 or not isinstance(self.rope, RopeConfig):
            raise ValueError("Invalid dropout/RoPE config")

    @property
    def attention_head_dim(self):
        return self.head_dim or self.hidden_size // self.num_attention_heads

    def window_for_layer(self, index):
        return None

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


@dataclass(frozen=True)
class Qwen2Config(LlamaConfig):
    architecture: ClassVar[str] = "qwen2"
    use_sliding_window: bool = False
    sliding_window: int = 32
    max_window_layers: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.sliding_window < 1 or self.max_window_layers < 0:
            raise ValueError("Invalid Qwen2 layer/window plan")

    def window_for_layer(self, index):
        return (
            self.sliding_window
            if self.use_sliding_window and index >= self.max_window_layers
            else None
        )


@dataclass(frozen=True)
class Qwen3Config(LlamaConfig):
    architecture: ClassVar[str] = "qwen3"
    sliding_window: int | None = None
    layer_types: tuple[str, ...] | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError("sliding_window must be positive")
        if self.layer_types is not None and (
            len(self.layer_types) != self.num_hidden_layers
            or any(x not in {"full_attention", "sliding_attention"} for x in self.layer_types)
        ):
            raise ValueError("layer_types must give every layer's real attention pattern")
        if (
            self.layer_types
            and "sliding_attention" in self.layer_types
            and self.sliding_window is None
        ):
            raise ValueError("Sliding layers require a window")

    def window_for_layer(self, index):
        return (
            self.sliding_window
            if self.layer_types and self.layer_types[index] == "sliding_attention"
            else None
        )


@dataclass(frozen=True)
class MistralConfig(LlamaConfig):
    architecture: ClassVar[str] = "mistral"
    sliding_window: int = 32

    def __post_init__(self):
        super().__post_init__()
        if self.sliding_window < 1:
            raise ValueError("Mistral sliding_window must be positive")

    def window_for_layer(self, index):
        return self.sliding_window


@dataclass(frozen=True)
class MixtralConfig(MistralConfig):
    architecture: ClassVar[str] = "mixtral"
    num_local_experts: int = 4
    num_experts_per_tok: int = 2
    router_jitter_noise: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if (
            not 1 <= self.num_experts_per_tok <= self.num_local_experts
            or not 0 <= self.router_jitter_noise < 1
        ):
            raise ValueError("Invalid Mixtral router configuration")


@dataclass(frozen=True)
class DeepSeekV3Config(LlamaConfig):
    architecture: ClassVar[str] = "deepseek_v3"
    rope: RopeConfig = field(default_factory=lambda: RopeConfig(interleaved=True))
    num_key_value_heads: int = 4
    kv_lora_rank: int = 8
    q_lora_rank: int | None = 12
    qk_nope_head_dim: int = 4
    qk_rope_head_dim: int = 4
    v_head_dim: int = 8
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 16
    first_k_dense_replace: int = 1
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    attention_bias: bool = False

    def __post_init__(self):
        super().__post_init__()
        if (
            min(
                self.kv_lora_rank,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.v_head_dim,
                self.n_routed_experts,
                self.n_shared_experts,
                self.moe_intermediate_size,
                self.n_group,
            )
            < 1
        ):
            raise ValueError("Invalid MLA/expert dimensions")
        if self.qk_rope_head_dim % 2 or self.q_lora_rank is not None and self.q_lora_rank < 1:
            raise ValueError("Invalid MLA query/rotary dimensions")
        if self.num_key_value_heads != self.num_attention_heads:
            raise ValueError(
                "MLA is not GQA: decoded content has one independent vector per query head"
            )
        if self.n_routed_experts % self.n_group or self.n_routed_experts // self.n_group < 2:
            raise ValueError("DeepSeek group scoring needs at least two experts per group")
        if (
            not 1 <= self.topk_group <= self.n_group
            or not 1
            <= self.num_experts_per_tok
            <= self.topk_group * self.n_routed_experts // self.n_group
        ):
            raise ValueError("Selected groups cannot contain the requested expert count")
        if (
            not 0 <= self.first_k_dense_replace <= self.num_hidden_layers
            or self.routed_scaling_factor <= 0
        ):
            raise ValueError("Invalid dense/MoE schedule or routing scale")


@dataclass(frozen=True)
class BertConfig:
    architecture: ClassVar[str] = "bert_mlm"
    vocab_size: int = 32
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    max_position_embeddings: int = 128
    type_vocab_size: int = 2
    pad_token_id: int = 0
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if (
            min(
                self.vocab_size,
                self.hidden_size,
                self.intermediate_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.max_position_embeddings,
                self.type_vocab_size,
            )
            < 1
        ):
            raise ValueError("Invalid BERT dimensions")
        if (
            self.hidden_size % self.num_attention_heads
            or not 0 <= self.pad_token_id < self.vocab_size
        ):
            raise ValueError("Invalid BERT attention/padding configuration")
        if (
            self.layer_norm_eps <= 0
            or self.initializer_range <= 0
            or any(
                not 0 <= p < 1
                for p in (self.hidden_dropout_prob, self.attention_probs_dropout_prob)
            )
        ):
            raise ValueError("Invalid BERT numerics")

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


@dataclass(frozen=True)
class T5Config:
    architecture: ClassVar[str] = "t5"
    vocab_size: int = 32
    d_model: int = 32
    d_kv: int = 8
    d_ff: int = 64
    num_layers: int = 2
    num_decoder_layers: int = 2
    num_heads: int = 4
    relative_attention_num_buckets: int = 16
    relative_attention_max_distance: int = 64
    dropout_rate: float = 0.0
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    feed_forward_proj: str = "relu"
    tie_word_embeddings: bool = True
    scale_decoder_outputs: bool = True
    pad_token_id: int = 0
    decoder_start_token_id: int = 0

    def __post_init__(self):
        if (
            min(
                self.vocab_size,
                self.d_model,
                self.d_kv,
                self.d_ff,
                self.num_layers,
                self.num_decoder_layers,
                self.num_heads,
            )
            < 1
        ):
            raise ValueError("Invalid T5 dimensions")
        if (
            self.relative_attention_num_buckets < 4
            or self.relative_attention_num_buckets % 4
            or self.relative_attention_max_distance <= self.relative_attention_num_buckets // 2
        ):
            raise ValueError("Invalid T5 relative bucket scale")
        if self.feed_forward_proj not in {"relu", "gated-gelu", "gated-silu"}:
            raise ValueError("Unsupported T5 feed-forward formula")
        if (
            not 0 <= self.dropout_rate < 1
            or min(self.layer_norm_epsilon, self.initializer_factor) <= 0
        ):
            raise ValueError("Invalid T5 numerics")
        if any(
            not 0 <= x < self.vocab_size for x in (self.pad_token_id, self.decoder_start_token_id)
        ):
            raise ValueError("Invalid T5 special token")
        if not self.tie_word_embeddings:
            raise ValueError(
                "Canonical T5 shares encoder/decoder/head embeddings; use scale_decoder_outputs to control its independent output scale"
            )

    def to_dict(self):
        return {"architecture": self.architecture, **asdict(self)}


CONFIG_TYPES = {
    c.architecture: c
    for c in (
        LlamaConfig,
        Qwen2Config,
        Qwen3Config,
        MistralConfig,
        MixtralConfig,
        DeepSeekV3Config,
        BertConfig,
        T5Config,
    )
}


def config_from_dict(values):
    if isinstance(values, dict) and values.get("architecture") == "dspark_gemma4":
        from .dspark_gemma4 import Gemma4DSparkConfig

        fields = dict(values)
        fields.pop("architecture")
        return Gemma4DSparkConfig(**fields)
    if isinstance(values, dict) and values.get("architecture") == "dspark_qwen3":
        from .dspark import DSparkConfig

        fields = dict(values)
        fields.pop("architecture")
        return DSparkConfig(**fields)
    from .vision import CLIPVisionConfig
    from .multimodal import LlavaConfig
    from .generative import UNetConfig, DiTConfig, AutoencoderConfig
    from .world import RSSMConfig
    from .actions import ACTConfig
    from .policies import DiffusionPolicyConfig, PiConfig
    from .hybrid import Qwen3NextConfig
    from .families import Gemma3TextConfig, Llama4TextConfig
    from .qwen_vl import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
    from .kimi import KimiK25Config, KimiK25VisionConfig
    from .siglip import SigLIPConfig, SigLIPVisionConfig, SigLIPTextConfig
    from .janus import JanusConfig, JanusVisionConfig, JanusVQConfig
    from .gpt import GPT2Config
    from .mamba import MambaConfig
    from .llada import LLaDAConfig
    from .sparse import DeepSeekV32Config
    from .jepa import JEPAEncoderConfig, JEPAConfig
    from .pi_vla import PiVLAConfig
    from .deepseek_v4 import DeepSeekV4Config
    from .qwen35 import Qwen35TextConfig, Qwen35MoETextConfig, Qwen35Config, Qwen35VisionConfig
    from .dinov2 import DinoVisionConfig
    from .openvla import OpenVLAConfig
    from .tdmpc2 import TDMPC2Config, TDMPC2PolicyConfig
    from .groot import GrootActionConfig, GrootConfig
    from .gemma4 import Gemma4TextConfig
    from .gemma4_vl import Gemma4VisionConfig, Gemma4Config
    from .video_world import WanVideoConfig
    from .video_vae import WanVAEConfig
    from .blip2 import Blip2QFormerConfig, Blip2VisionConfig, Blip2Config
    from .qwen_mtp import QwenMTPConfig
    from .conservative import CQLPolicyConfig
    from .kimi_k3 import KimiK3TextConfig
    from .muzero import MuZeroConfig
    from .kimi_k3_vl import KimiK3VisionConfig, KimiK3Config
    from .qwen4_exp import Qwen4ExpTextConfig
    from .qwen4_exp_vl import Qwen4ExpVisionConfig, Qwen4ExpConfig
    from .drifting import DriftingConfig
    from .drifting_features import MAEResNetConfig
    from .ocr2_vision import OCR2SAMConfig, OCR2VisualConfig
    from .ocr2 import OCR2TextConfig, OCR2Config
    from .interval_dit import IntervalDiTConfig
    from .planet import PlaNetConfig
    from .cosmos3 import Cosmos3Config
    from .vmc import VMCVAEConfig, MDNRNNConfig, VMCControllerConfig
    from .cosmos_predict1 import CosmosPredict1Config, CosmosPredict1ModelConfig
    from .genie import (
        GenieTokenizerConfig,
        GenieActionConfig,
        GenieDynamicsConfig,
        GenieWorldConfig,
    )
    from .wan22_vae import Wan22VAEConfig
    from .cosmos3_vlm import Cosmos3VLMConfig
    from .perceptual import LPIPSConfig
    from .cosmos3_audio import Cosmos3AudioConfig
    from .adversarial import PatchDiscriminatorConfig
    from .vit import ViTConfig
    from .lewm import LeWMConfig

    values = dict(values)
    architecture = values.pop("architecture")
    types = {
        **CONFIG_TYPES,
        "clip_vision": CLIPVisionConfig,
        "llava": LlavaConfig,
        "unet2d": UNetConfig,
        "dit": DiTConfig,
        "autoencoder_kl": AutoencoderConfig,
        "rssm": RSSMConfig,
        "act": ACTConfig,
        "qwen3_next": Qwen3NextConfig,
        "gemma3_text": Gemma3TextConfig,
        "llama4_text": Llama4TextConfig,
        "qwen3_vl": Qwen3VLConfig,
        "qwen3_vl_text": Qwen3VLTextConfig,
        "qwen3_vl_vision": Qwen3VLVisionConfig,
        "kimi_k25": KimiK25Config,
        "kimi_k25_vision": KimiK25VisionConfig,
        "diffusion_policy": DiffusionPolicyConfig,
        "pi_action_expert": PiConfig,
    }
    types.update(
        siglip=SigLIPConfig, siglip_vision=SigLIPVisionConfig, siglip_text=SigLIPTextConfig
    )
    types.update(janus=JanusConfig, janus_vision=JanusVisionConfig, janus_vq=JanusVQConfig)
    types.update(gpt2=GPT2Config)
    types.update(mamba=MambaConfig)
    types.update(llada=LLaDAConfig)
    types.update(deepseek_v32=DeepSeekV32Config)
    types.update(jepa_encoder=JEPAEncoderConfig, jepa=JEPAConfig)
    types.update(pi_vla=PiVLAConfig)
    types.update(deepseek_v4=DeepSeekV4Config)
    types.update(
        qwen3_5_text=Qwen35TextConfig,
        qwen3_5_moe_text=Qwen35MoETextConfig,
        qwen3_5=Qwen35Config,
        qwen3_5_vision=Qwen35VisionConfig,
    )
    types.update(dinov2_register_vision=DinoVisionConfig, openvla=OpenVLAConfig)
    types.update(tdmpc2_world=TDMPC2Config, tdmpc2_policy=TDMPC2PolicyConfig)
    types.update(groot_n17_action=GrootActionConfig, groot_n17=GrootConfig)
    types.update(gemma4_text=Gemma4TextConfig)
    types.update(gemma4_vision=Gemma4VisionConfig, gemma4=Gemma4Config)
    types.update(wan21_video=WanVideoConfig, wan21_vae=WanVAEConfig)
    types.update(
        blip2_qformer=Blip2QFormerConfig, blip2_vision=Blip2VisionConfig, blip2=Blip2Config
    )
    types.update(qwen_mtp=QwenMTPConfig)
    types.update(cql_policy=CQLPolicyConfig)
    types.update(kimi_k3_text=KimiK3TextConfig)
    types.update(muzero_vector=MuZeroConfig)
    types.update(kimi_k3_vision=KimiK3VisionConfig, kimi_k3=KimiK3Config)
    types.update(qwen4_exp_text=Qwen4ExpTextConfig)
    types.update(qwen4_exp_vision=Qwen4ExpVisionConfig, qwen4_exp=Qwen4ExpConfig)
    types.update(drifting_generator=DriftingConfig)
    types.update(drifting_mae=MAEResNetConfig)
    types.update(ocr2_sam=OCR2SAMConfig, ocr2_visual=OCR2VisualConfig)
    types.update(ocr2_text=OCR2TextConfig, deepseek_ocr2=OCR2Config)
    types.update(interval_dit=IntervalDiTConfig)
    types.update(planet=PlaNetConfig)
    types.update(cosmos3_mot=Cosmos3Config)
    types.update(vmc_vae=VMCVAEConfig, vmc_mdn_rnn=MDNRNNConfig, vmc_controller=VMCControllerConfig)
    types.update(
        cosmos_predict1=CosmosPredict1Config, cosmos_predict1_model=CosmosPredict1ModelConfig
    )
    types.update(
        genie_tokenizer=GenieTokenizerConfig,
        genie_action=GenieActionConfig,
        genie_dynamics=GenieDynamicsConfig,
        genie_world=GenieWorldConfig,
    )
    types.update(wan22_vae=Wan22VAEConfig)
    types.update(cosmos3_vlm=Cosmos3VLMConfig)
    types.update(lpips=LPIPSConfig)
    types.update(cosmos3_avae2=Cosmos3AudioConfig)
    types.update(patch_discriminator=PatchDiscriminatorConfig)
    types.update(vit=ViTConfig, lewm=LeWMConfig)
    if architecture not in types:
        raise ValueError(f"Unknown/unimplemented architecture: {architecture}")
    if "rope" in values:
        values["rope"] = RopeConfig(**values["rope"])
    if "rope_local" in values:
        values["rope_local"] = RopeConfig(**values["rope_local"])
    if values.get("layer_types") is not None:
        values["layer_types"] = tuple(values["layer_types"])
    for name in ("channel_mult", "attention_levels", "down_dims"):
        if name in values:
            values[name] = tuple(values[name])
    if architecture in {
        "llava",
        "qwen3_vl",
        "kimi_k25",
        "siglip",
        "janus",
        "qwen3_5",
        "gemma4",
        "kimi_k3",
        "qwen4_exp",
        "deepseek_ocr2",
    }:
        values["text_config"] = config_from_dict(values["text_config"])
        values["vision_config"] = config_from_dict(values["vision_config"])
        if isinstance(values.get("vision_feature_layer"), list):
            values["vision_feature_layer"] = tuple(values["vision_feature_layer"])
    if architecture == "janus":
        values["vq_config"] = config_from_dict(values["vq_config"])
    if architecture == "openvla":
        from aster.data.actions import ActionSpec

        for name in ("text_config", "dino_config", "siglip_config"):
            values[name] = config_from_dict(values[name])
        if values.get("action_spec") is not None:
            values["action_spec"] = ActionSpec(**values["action_spec"])
    if architecture == "groot_n17":
        for name in ("backbone_config", "action_config"):
            values[name] = config_from_dict(values[name])
    if architecture == "blip2":
        for name in ("vision_config", "qformer_config", "text_config"):
            values[name] = config_from_dict(values[name])
    if architecture == "qwen_mtp":
        values["text_config"] = config_from_dict(values["text_config"])
    if architecture == "ocr2_visual":
        values["sam_config"] = config_from_dict(values["sam_config"])
        values["decoder_config"] = config_from_dict(values["decoder_config"])
    if architecture == "genie_world":
        values["action"] = config_from_dict(values["action"])
        values["dynamics"] = config_from_dict(values["dynamics"])
    return types[architecture](**values)
