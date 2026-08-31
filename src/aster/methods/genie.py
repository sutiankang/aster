"""Video codebook, latent-action, and dynamics training with globally normalized losses."""

import math
import torch
from torch import nn
import torch.nn.functional as F

from ..core import LossBundle, LossTerm
from ..models.genie import (
    GenieTokenizerConfig,
    GenieActionConfig,
    GenieWorldConfig,
    GenieDynamicsConfig,
    _validate_video,
)


def _valid_frames(batch, shape, device):
    valid = batch.get("valid")
    if valid is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if valid.shape != shape or valid.dtype != torch.bool or valid.device != device:
        raise ValueError("Genie valid must be aligned bool [B,T]")
    if ((~valid[:, :-1]) & valid[:, 1:]).any():
        raise ValueError("Genie valid frames must be contiguous prefixes")
    return valid


def _vq_terms(output, target, valid, commitment_cost):
    error = (output.reconstruction.float() - target.float()).square()
    pixel_mask = valid[:, :, None, None, None].expand_as(error)
    encoding = output.encoding
    latent_mask = valid.reshape(*valid.shape, *((1,) * (encoding.quantized.ndim - 2))).expand_as(
        encoding.quantized
    )
    return (
        LossTerm(
            error.masked_select(pixel_mask).sum(),
            pixel_mask.sum(dtype=torch.int64),
            "pixel",
            "reconstruction",
        ),
        LossTerm(
            encoding.codebook_errors.masked_select(latent_mask).sum(),
            latent_mask.sum(dtype=torch.int64),
            "latent_coordinate",
            "codebook",
        ),
        LossTerm(
            encoding.commitment_errors.masked_select(latent_mask).sum(),
            latent_mask.sum(dtype=torch.int64),
            "latent_coordinate",
            "commitment",
            commitment_cost,
        ),
    )


class GenieVQObjective(nn.Module):
    """Normalize pixel reconstruction and latent/codebook terms independently."""

    def __init__(self, *, sequence_length=16, commitment_cost=0.25):
        super().__init__()
        if (
            type(sequence_length) is not int
            or sequence_length < 2
            or not math.isfinite(commitment_cost)
            or commitment_cost < 0
        ):
            raise ValueError("Invalid Genie VQ objective")
        self.sequence_length, self.commitment_cost = sequence_length, commitment_cost

    def config_dict(self):
        return dict(
            type="genie_vq",
            sequence_length=self.sequence_length,
            commitment_cost=self.commitment_cost,
            codebook_update="gradient",
            reconstruction="mean_pixel_mse",
        )

    def _validate(self, model, batch):
        if not isinstance(model.config, GenieTokenizerConfig) or set(batch) - {"video", "valid"}:
            raise ValueError(
                "Genie VQ objective requires tokenizer or action model and pixel video"
            )
        _validate_video(batch["video"], model.config, minimum_frames=2)
        if batch["video"].shape[1] != self.sequence_length:
            raise ValueError(
                "Genie sequence length must be fixed before distributed parameter gathers"
            )
        _valid_frames(batch, batch["video"].shape[:2], batch["video"].device)

    def preflight_microbatches(self, model, batches):
        for batch in batches:
            self._validate(model, batch)
        return batches

    def forward(self, model, batch):
        self._validate(model, batch)
        video = batch["video"]
        output = model(video)
        valid = _valid_frames(batch, video.shape[:2], video.device)
        action = isinstance(model.config, GenieActionConfig)
        return LossBundle(
            _vq_terms(
                output,
                video[:, 1:] if action else video,
                valid[:, 1:] if action else valid,
                self.commitment_cost,
            )
        )


def _validate_tokens(tokens, actions, c, sequence_length):
    if (
        not isinstance(c, GenieDynamicsConfig)
        or tokens.ndim != 3
        or tokens.shape[1:] != (sequence_length, c.spatial_tokens)
        or len(tokens) < 1
        or tokens.dtype != torch.int64
        or (tokens < 0).any()
        or (tokens >= c.vocab_size).any()
    ):
        raise ValueError(
            "Genie training tokens must be genuine unmasked code indices of fixed length"
        )
    if actions is not None and (
        actions.shape != (len(tokens), sequence_length - 1, c.action_dim)
        or actions.device != tokens.device
        or not actions.is_floating_point()
        or not torch.isfinite(actions).all()
    ):
        raise ValueError("Genie training latent actions differ from token sequence")


def _validate_mask(batch, tokens, valid):
    mask = batch.get("mask")
    if mask is not None and (
        mask.shape != tokens.shape
        or mask.dtype != torch.bool
        or mask.device != tokens.device
        or mask[:, 0].any()
        or (mask & ~valid[:, :, None]).any()
    ):
        raise ValueError("Genie training mask may cover only valid future-frame tokens")


def _mask_tokens(batch, tokens, valid, mask_token_id):
    mask = batch.get("mask")
    if mask is None:
        rate = 0.5 + 0.5 * torch.rand(len(tokens), 1, 1, device=tokens.device)
        mask = torch.rand(tokens.shape, device=tokens.device) < rate
        mask[:, 0] = False
        mask = mask & valid[:, :, None]
    return tokens.masked_fill(mask, mask_token_id), mask


def _token_term(logits, targets, mask):
    loss = F.cross_entropy(logits.flatten(0, 2), targets.flatten(), reduction="none").reshape_as(
        targets
    )
    return LossTerm(
        loss.masked_select(mask).sum(),
        mask.sum(dtype=torch.int64),
        "masked_video_token",
        "dynamics_ce",
    )


class GenieDynamicsObjective(nn.Module):
    def __init__(self, *, sequence_length=16):
        super().__init__()
        if type(sequence_length) is not int or sequence_length < 2:
            raise ValueError("Invalid Genie dynamics sequence length")
        self.sequence_length = sequence_length

    def config_dict(self):
        return dict(
            type="genie_dynamics",
            sequence_length=self.sequence_length,
            mask="bernoulli_uniform_0.5_1",
            supervision="masked_same_frame",
            action_alignment="previous_to_current",
        )

    def _validate(self, model, batch):
        if set(batch) - {"tokens", "actions", "valid", "mask"}:
            raise ValueError("Unknown Genie dynamics batch field")
        tokens = batch["tokens"]
        _validate_tokens(tokens, batch["actions"], model.config, self.sequence_length)
        valid = _valid_frames(batch, tokens.shape[:2], tokens.device)
        _validate_mask(batch, tokens, valid)

    def preflight_microbatches(self, model, batches):
        for batch in batches:
            self._validate(model, batch)
        return batches

    def forward(self, model, batch):
        self._validate(model, batch)
        tokens = batch["tokens"]
        valid = _valid_frames(batch, tokens.shape[:2], tokens.device)
        masked, mask = _mask_tokens(batch, tokens, valid, model.config.mask_token_id)
        return LossBundle((_token_term(model(masked, batch["actions"].detach()), tokens, mask),))


class GenieWorldObjective(nn.Module):
    def __init__(self, *, sequence_length=16, commitment_cost=0.25, dynamics_weight=1.0):
        super().__init__()
        self.vq = GenieVQObjective(sequence_length=sequence_length, commitment_cost=commitment_cost)
        if not math.isfinite(dynamics_weight) or dynamics_weight < 0:
            raise ValueError("Invalid Genie dynamics weight")
        self.dynamics_weight = dynamics_weight

    def config_dict(self):
        return dict(
            type="genie_world",
            sequence_length=self.vq.sequence_length,
            commitment_cost=self.vq.commitment_cost,
            dynamics_weight=self.dynamics_weight,
            supervision="masked_same_frame",
            action_gradient="stop",
        )

    def _validate(self, model, batch):
        if not isinstance(model.config, GenieWorldConfig) or set(batch) - {
            "video",
            "tokens",
            "valid",
            "mask",
        }:
            raise ValueError(
                "Genie joint objective requires explicit world composite/pixel-token batch"
            )
        _validate_video(batch["video"], model.config.action, minimum_frames=2)
        if (
            batch["video"].shape[:2] != batch["tokens"].shape[:2]
            or batch["video"].device != batch["tokens"].device
        ):
            raise ValueError("Genie pixel/token time or batch alignment differs")
        _validate_tokens(batch["tokens"], None, model.config.dynamics, self.vq.sequence_length)
        valid = _valid_frames(batch, batch["tokens"].shape[:2], batch["tokens"].device)
        _validate_mask(batch, batch["tokens"], valid)

    def preflight_microbatches(self, model, batches):
        for batch in batches:
            self._validate(model, batch)
        return batches

    def forward(self, model, batch):
        self._validate(model, batch)
        tokens, video = batch["tokens"], batch["video"]
        valid = _valid_frames(batch, tokens.shape[:2], tokens.device)
        masked, mask = _mask_tokens(batch, tokens, valid, model.config.dynamics.mask_token_id)
        action, logits = model(video, masked)
        terms = _vq_terms(action, video[:, 1:], valid[:, 1:], self.vq.commitment_cost)
        ce = _token_term(logits, tokens, mask)
        ce.weight = self.dynamics_weight
        return LossBundle((*terms, ce))


@torch.no_grad()
def encode_genie_video(tokenizer, video, *, valid=None):

    if type(tokenizer.config) is not GenieTokenizerConfig:
        raise ValueError("Genie world token targets require the video tokenizer, not action codes")
    modes = {module: module.training for module in tokenizer.modules()}
    try:
        tokenizer.eval()
        result = dict(video=video, tokens=tokenizer.encode(video).indices.detach())
        if valid is not None:
            _valid_frames(dict(valid=valid), video.shape[:2], video.device)
            result["valid"] = valid
        return result
    finally:
        for module, mode in modes.items():
            module.training = mode
