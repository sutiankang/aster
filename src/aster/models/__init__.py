"""Local model construction with explicit architecture-specific implementations."""

from .config import (
    LlamaConfig,
    Qwen2Config,
    Qwen3Config,
    MistralConfig,
    MixtralConfig,
    DeepSeekV3Config,
    BertConfig,
    T5Config,
)
from .decoder import CausalLM
from .serialization import LocalModelMixin
from .moe import MixtralForCausalLM, DeepSeekV3ForCausalLM
from .bert import BertForMaskedLM
from .t5 import T5ForConditionalGeneration
from .vision import CLIPVisionConfig, CLIPVisionModel
from .multimodal import LlavaConfig, LlavaForConditionalGeneration
from .generative import UNetConfig, UNet2D, DiTConfig, DiT, AutoencoderConfig, AutoencoderKL
from .world import RSSMConfig, RSSMWorldModel
from .actions import ACTConfig, ACTPolicy
from .policies import DiffusionPolicyConfig, DiffusionPolicy1D, PiConfig, PiActionExpert
from .hybrid import Qwen3NextConfig, Qwen3NextForCausalLM
from .families import Gemma3TextConfig, Gemma3ForCausalLM, Llama4TextConfig, Llama4ForCausalLM
from .qwen_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLTextForCausalLM,
    Qwen3VLVisionModel,
)
from .kimi import (
    KimiK25Config,
    KimiK25VisionConfig,
    KimiK25ForConditionalGeneration,
    KimiK25VisionModel,
)
from .siglip import (
    SigLIPConfig,
    SigLIPVisionConfig,
    SigLIPTextConfig,
    SigLIPModel,
    SigLIPVisionModel,
    SigLIPTextModel,
)
from .janus import (
    JanusConfig,
    JanusVisionConfig,
    JanusVQConfig,
    JanusForConditionalGeneration,
    JanusVisionModel,
    JanusVQModel,
)
from .gpt import GPT2Config, GPT2ForCausalLM
from .mamba import MambaConfig, MambaForCausalLM
from .llada import LLaDAConfig, LLaDAForMaskedLM
from .sparse import DeepSeekV32Config, DeepSeekV32ForCausalLM
from .jepa import JEPAEncoderConfig, JEPAEncoder, JEPAConfig, JEPAModel
from .pi_vla import PiVLAConfig, PiVLA
from .deepseek_v4 import DeepSeekV4Config, DeepSeekV4ForCausalLM
from .qwen35 import (
    Qwen35TextConfig,
    Qwen35MoETextConfig,
    Qwen35Config,
    Qwen35VisionConfig,
    Qwen35ForCausalLM,
    Qwen35ForConditionalGeneration,
)
from .dinov2 import DinoVisionConfig, DinoVisionModel
from .openvla import OpenVLAConfig, OpenVLAForActionPrediction
from .tdmpc2 import TDMPC2Config, TDMPC2WorldModel, TDMPC2PolicyConfig, TDMPC2Policy
from .groot import GrootActionConfig, GrootActionHead, GrootConfig, GrootVLA, GrootCondition
from .gemma4 import Gemma4TextConfig, Gemma4ForCausalLM
from .gemma4_vl import (
    Gemma4VisionConfig,
    Gemma4VisionModel,
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    pack_gemma4_images,
)
from .video_world import WanVideoConfig, WanVideoDiT
from .video_vae import WanVAEConfig, WanVideoVAE
from .blip2 import (
    Blip2QFormerConfig,
    Blip2QFormerModel,
    Blip2VisionConfig,
    Blip2VisionModel,
    Blip2Config,
    Blip2ForConditionalGeneration,
)
from .qwen_mtp import QwenMTPConfig, QwenMTPHead, QwenMTPForCausalLM
from .conservative import CQLPolicyConfig, CQLPolicy
from .kimi_k3 import KimiK3TextConfig, KimiK3ForCausalLM
from .muzero import MuZeroConfig, MuZeroModel
from .kimi_k3_vl import (
    KimiK3VisionConfig,
    KimiK3VisionModel,
    KimiK3Config,
    KimiK3ForConditionalGeneration,
)
from .qwen4_exp import Qwen4ExpTextConfig, Qwen4ExpForCausalLM
from .qwen4_exp_vl import Qwen4ExpVisionConfig, Qwen4ExpConfig, Qwen4ExpForConditionalGeneration
from .drifting import DriftingConfig, DriftingGenerator
from .drifting_features import MAEResNetConfig, MAEResNet
from .ocr2_vision import OCR2SAMConfig, OCR2SAMEncoder, OCR2VisualConfig, OCR2VisualEncoder
from .ocr2 import OCR2TextConfig, OCR2ForCausalLM, OCR2Config, OCR2ForConditionalGeneration
from .interval_dit import IntervalDiTConfig, IntervalDiT
from .planet import PlaNetConfig, PlaNetWorldModel
from .cosmos3 import (
    Cosmos3Config,
    Cosmos3MoT,
    Cosmos3Vision,
    Cosmos3Sequence,
    Cosmos3Output,
    cosmos3_positions,
)
from .vmc import VMCVAEConfig, VMCVAE, MDNRNNConfig, MDNRNN, VMCControllerConfig, VMCController
from .cosmos_predict1 import (
    CosmosPredict1Config,
    CosmosPredict1DiT,
    CosmosPredict1Condition,
    CosmosPredict1ModelConfig,
    CosmosPredict1Model,
)
from .genie import (
    GenieTokenizerConfig,
    GenieTokenizer,
    GenieActionConfig,
    GenieLatentAction,
    GenieDynamicsConfig,
    GenieDynamics,
    GenieWorldConfig,
    GenieWorld,
)
from .wan22_vae import Wan22VAEConfig, Wan22VideoVAE
from .cosmos3_vlm import Cosmos3VLMConfig, Cosmos3VLM, Cosmos3VLMState
from .perceptual import LPIPSConfig, LPIPS
from .cosmos3_audio import Cosmos3AudioConfig, Cosmos3AudioCodec
from .adversarial import PatchDiscriminatorConfig, PatchDiscriminator
from .vit import ViTConfig, ViTModel
from .lewm import LeWMConfig, LeWorldModel


def build_model(config):
    if getattr(config, "architecture", None) == "dspark_gemma4":
        from .dspark_gemma4 import Gemma4DSparkConfig, Gemma4DSparkDraft

        if type(config) is not Gemma4DSparkConfig:
            raise ValueError("Gemma4 DSpark factory requires exact native config")
        return Gemma4DSparkDraft(config)
    if getattr(config, "architecture", None) == "dspark_qwen3":
        from .dspark import DSparkConfig, DSparkDraft

        if type(config) is not DSparkConfig:
            raise ValueError("DSpark factory requires exact native DSparkConfig")
        return DSparkDraft(config)
    if type(config) in (LlamaConfig, Qwen2Config, Qwen3Config, MistralConfig):
        return CausalLM(config)
    if type(config) is MixtralConfig:
        return MixtralForCausalLM(config)
    if type(config) is DeepSeekV3Config:
        return DeepSeekV3ForCausalLM(config)
    if type(config) is BertConfig:
        return BertForMaskedLM(config)
    if type(config) is T5Config:
        return T5ForConditionalGeneration(config)
    if type(config) is CLIPVisionConfig:
        return CLIPVisionModel(config)
    if type(config) is LlavaConfig:
        return LlavaForConditionalGeneration(config)
    if type(config) is Qwen3NextConfig:
        return Qwen3NextForCausalLM(config)
    if type(config) is Gemma3TextConfig:
        return Gemma3ForCausalLM(config)
    if type(config) is Llama4TextConfig:
        return Llama4ForCausalLM(config)
    if type(config) is Qwen3VLConfig:
        return Qwen3VLForConditionalGeneration(config)
    if type(config) is Qwen3VLTextConfig:
        return Qwen3VLTextForCausalLM(config)
    if type(config) is Qwen3VLVisionConfig:
        return Qwen3VLVisionModel(config)
    if type(config) is KimiK25Config:
        return KimiK25ForConditionalGeneration(config)
    if type(config) is KimiK25VisionConfig:
        return KimiK25VisionModel(config)
    implementations = {
        UNetConfig: UNet2D,
        DiTConfig: DiT,
        AutoencoderConfig: AutoencoderKL,
        SigLIPConfig: SigLIPModel,
        SigLIPVisionConfig: SigLIPVisionModel,
        SigLIPTextConfig: SigLIPTextModel,
        JanusConfig: JanusForConditionalGeneration,
        JanusVisionConfig: JanusVisionModel,
        JanusVQConfig: JanusVQModel,
        GPT2Config: GPT2ForCausalLM,
        MambaConfig: MambaForCausalLM,
        LLaDAConfig: LLaDAForMaskedLM,
        DeepSeekV32Config: DeepSeekV32ForCausalLM,
        JEPAEncoderConfig: JEPAEncoder,
        JEPAConfig: JEPAModel,
        PiVLAConfig: PiVLA,
        DeepSeekV4Config: DeepSeekV4ForCausalLM,
        Qwen35TextConfig: Qwen35ForCausalLM,
        Qwen35MoETextConfig: Qwen35ForCausalLM,
        Qwen35Config: Qwen35ForConditionalGeneration,
        Qwen35VisionConfig: Qwen3VLVisionModel,
        DinoVisionConfig: DinoVisionModel,
        OpenVLAConfig: OpenVLAForActionPrediction,
        TDMPC2Config: TDMPC2WorldModel,
        TDMPC2PolicyConfig: TDMPC2Policy,
        GrootActionConfig: GrootActionHead,
        GrootConfig: GrootVLA,
        Gemma4TextConfig: Gemma4ForCausalLM,
        Gemma4VisionConfig: Gemma4VisionModel,
        Gemma4Config: Gemma4ForConditionalGeneration,
        WanVideoConfig: WanVideoDiT,
        WanVAEConfig: WanVideoVAE,
        Blip2QFormerConfig: Blip2QFormerModel,
        Blip2VisionConfig: Blip2VisionModel,
        Blip2Config: Blip2ForConditionalGeneration,
        QwenMTPConfig: QwenMTPForCausalLM,
        CQLPolicyConfig: CQLPolicy,
        KimiK3TextConfig: KimiK3ForCausalLM,
        MuZeroConfig: MuZeroModel,
        KimiK3VisionConfig: KimiK3VisionModel,
        KimiK3Config: KimiK3ForConditionalGeneration,
        Qwen4ExpTextConfig: Qwen4ExpForCausalLM,
        Qwen4ExpVisionConfig: Qwen3VLVisionModel,
        Qwen4ExpConfig: Qwen4ExpForConditionalGeneration,
        DriftingConfig: DriftingGenerator,
        MAEResNetConfig: MAEResNet,
        OCR2SAMConfig: OCR2SAMEncoder,
        OCR2VisualConfig: OCR2VisualEncoder,
        OCR2TextConfig: OCR2ForCausalLM,
        OCR2Config: OCR2ForConditionalGeneration,
        IntervalDiTConfig: IntervalDiT,
        PlaNetConfig: PlaNetWorldModel,
        Cosmos3Config: Cosmos3MoT,
        VMCVAEConfig: VMCVAE,
        MDNRNNConfig: MDNRNN,
        VMCControllerConfig: VMCController,
        CosmosPredict1Config: CosmosPredict1DiT,
        CosmosPredict1ModelConfig: CosmosPredict1Model,
        GenieTokenizerConfig: GenieTokenizer,
        GenieActionConfig: GenieLatentAction,
        GenieDynamicsConfig: GenieDynamics,
        GenieWorldConfig: GenieWorld,
        Wan22VAEConfig: Wan22VideoVAE,
        Cosmos3VLMConfig: Cosmos3VLM,
        LPIPSConfig: LPIPS,
        Cosmos3AudioConfig: Cosmos3AudioCodec,
        PatchDiscriminatorConfig: PatchDiscriminator,
        ViTConfig: ViTModel,
        LeWMConfig: LeWorldModel,
        RSSMConfig: RSSMWorldModel,
        ACTConfig: ACTPolicy,
        DiffusionPolicyConfig: DiffusionPolicy1D,
        PiConfig: PiActionExpert,
    }
    if type(config) in implementations:
        return implementations[type(config)](config)
    raise ValueError(
        f"Architecture is not implemented by this factory yet: {type(config).__name__}"
    )


def load_model(path):
    return LocalModelMixin.from_pretrained(path)


__all__ = [
    "build_model",
    "load_model",
    "CausalLM",
    "LlamaConfig",
    "Qwen2Config",
    "Qwen3Config",
    "MistralConfig",
    "MixtralConfig",
    "DeepSeekV3Config",
    "BertConfig",
    "T5Config",
    "BertForMaskedLM",
    "T5ForConditionalGeneration",
    "CLIPVisionConfig",
    "CLIPVisionModel",
    "LlavaConfig",
    "LlavaForConditionalGeneration",
]
__all__ += ["GrootActionConfig", "GrootActionHead", "GrootConfig", "GrootVLA", "GrootCondition"]
__all__ += ["Gemma4TextConfig", "Gemma4ForCausalLM"]
__all__ += ["Gemma4VisionConfig", "Gemma4VisionModel", "pack_gemma4_images"]
__all__ += ["Gemma4Config", "Gemma4ForConditionalGeneration"]
__all__ += ["WanVideoConfig", "WanVideoDiT", "WanVAEConfig", "WanVideoVAE"]
__all__ += [
    "Blip2QFormerConfig",
    "Blip2QFormerModel",
    "Blip2VisionConfig",
    "Blip2VisionModel",
    "Blip2Config",
    "Blip2ForConditionalGeneration",
]
__all__ += ["QwenMTPConfig", "QwenMTPHead", "QwenMTPForCausalLM"]
__all__ += ["CQLPolicyConfig", "CQLPolicy"]
__all__ += ["KimiK3TextConfig", "KimiK3ForCausalLM"]
__all__ += ["MuZeroConfig", "MuZeroModel"]
__all__ += [
    "KimiK3VisionConfig",
    "KimiK3VisionModel",
    "KimiK3Config",
    "KimiK3ForConditionalGeneration",
]
__all__ += ["Qwen4ExpTextConfig", "Qwen4ExpForCausalLM"]
__all__ += ["Qwen4ExpVisionConfig", "Qwen4ExpConfig", "Qwen4ExpForConditionalGeneration"]
__all__ += ["DriftingConfig", "DriftingGenerator"]
__all__ += ["MAEResNetConfig", "MAEResNet"]
__all__ += ["OCR2SAMConfig", "OCR2SAMEncoder", "OCR2VisualConfig", "OCR2VisualEncoder"]
__all__ += ["OCR2TextConfig", "OCR2ForCausalLM", "OCR2Config", "OCR2ForConditionalGeneration"]
__all__ += ["IntervalDiTConfig", "IntervalDiT"]
__all__ += ["PlaNetConfig", "PlaNetWorldModel"]
__all__ += [
    "Cosmos3Config",
    "Cosmos3MoT",
    "Cosmos3Vision",
    "Cosmos3Sequence",
    "Cosmos3Output",
    "cosmos3_positions",
]
__all__ += [
    "VMCVAEConfig",
    "VMCVAE",
    "MDNRNNConfig",
    "MDNRNN",
    "VMCControllerConfig",
    "VMCController",
]
__all__ += [
    "CosmosPredict1Config",
    "CosmosPredict1DiT",
    "CosmosPredict1Condition",
    "CosmosPredict1ModelConfig",
    "CosmosPredict1Model",
]
__all__ += [
    "GenieTokenizerConfig",
    "GenieTokenizer",
    "GenieActionConfig",
    "GenieLatentAction",
    "GenieDynamicsConfig",
    "GenieDynamics",
    "GenieWorldConfig",
    "GenieWorld",
]
__all__ += ["Wan22VAEConfig", "Wan22VideoVAE"]
__all__ += ["Cosmos3VLMConfig", "Cosmos3VLM", "Cosmos3VLMState"]
__all__ += ["LPIPSConfig", "LPIPS"]
__all__ += ["Cosmos3AudioConfig", "Cosmos3AudioCodec"]
__all__ += ["PatchDiscriminatorConfig", "PatchDiscriminator"]
__all__ += ["ViTConfig", "ViTModel", "LeWMConfig", "LeWorldModel"]
__all__ += ["DSparkConfig", "DSparkDraft"]
__all__ += ["Gemma4DSparkConfig", "Gemma4DSparkDraft"]
__all__ += [
    "UNetConfig",
    "UNet2D",
    "DiTConfig",
    "DiT",
    "AutoencoderConfig",
    "AutoencoderKL",
    "RSSMConfig",
    "RSSMWorldModel",
    "ACTConfig",
    "ACTPolicy",
]
__all__ += ["Qwen3NextConfig", "Qwen3NextForCausalLM"]
__all__ += ["Gemma3TextConfig", "Gemma3ForCausalLM", "Llama4TextConfig", "Llama4ForCausalLM"]
__all__ += [
    "Qwen3VLConfig",
    "Qwen3VLTextConfig",
    "Qwen3VLVisionConfig",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLTextForCausalLM",
    "Qwen3VLVisionModel",
]
__all__ += [
    "KimiK25Config",
    "KimiK25VisionConfig",
    "KimiK25ForConditionalGeneration",
    "KimiK25VisionModel",
]
__all__ += ["DiffusionPolicyConfig", "DiffusionPolicy1D", "PiConfig", "PiActionExpert"]
__all__ += [
    "SigLIPConfig",
    "SigLIPVisionConfig",
    "SigLIPTextConfig",
    "SigLIPModel",
    "SigLIPVisionModel",
    "SigLIPTextModel",
]
__all__ += [
    "JanusConfig",
    "JanusVisionConfig",
    "JanusVQConfig",
    "JanusForConditionalGeneration",
    "JanusVisionModel",
    "JanusVQModel",
]
__all__ += ["GPT2Config", "GPT2ForCausalLM"]
__all__ += ["MambaConfig", "MambaForCausalLM"]
__all__ += ["LLaDAConfig", "LLaDAForMaskedLM"]
__all__ += ["DeepSeekV32Config", "DeepSeekV32ForCausalLM"]
__all__ += ["JEPAEncoderConfig", "JEPAEncoder", "JEPAConfig", "JEPAModel"]
__all__ += ["PiVLAConfig", "PiVLA"]
__all__ += ["DeepSeekV4Config", "DeepSeekV4ForCausalLM"]
__all__ += [
    "Qwen35TextConfig",
    "Qwen35MoETextConfig",
    "Qwen35Config",
    "Qwen35VisionConfig",
    "Qwen35ForCausalLM",
    "Qwen35ForConditionalGeneration",
]
__all__ += ["DinoVisionConfig", "DinoVisionModel", "OpenVLAConfig", "OpenVLAForActionPrediction"]
__all__ += ["TDMPC2Config", "TDMPC2WorldModel", "TDMPC2PolicyConfig", "TDMPC2Policy"]


def __getattr__(name):

    if name in {"DSparkConfig", "DSparkDraft"}:
        from .dspark import DSparkConfig, DSparkDraft

        return {"DSparkConfig": DSparkConfig, "DSparkDraft": DSparkDraft}[name]
    if name in {"Gemma4DSparkConfig", "Gemma4DSparkDraft"}:
        from .dspark_gemma4 import Gemma4DSparkConfig, Gemma4DSparkDraft

        return {"Gemma4DSparkConfig": Gemma4DSparkConfig, "Gemma4DSparkDraft": Gemma4DSparkDraft}[
            name
        ]
    raise AttributeError(name)
