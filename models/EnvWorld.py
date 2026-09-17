"""Environment-first world model for deterministic multi-step forecasting.

The framework passes one multivariate history ``x`` with shape ``[B, L, C]``.
``target_indices`` identifies the target series; all remaining channels are treated
as environment variables.  The model first predicts the complete future environment
trajectory and then conditions a second direct decoder on it to predict the target.
The wrapper finally restores the original channel order so it remains compatible
with the repository's standard training and evaluation pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ForecastOutput:
    """Detailed outputs returned by ``Model(..., return_components=True)``."""

    target: Tensor
    environment: Tensor
    environment_used: Tensor
    target_normalized: Tensor
    environment_normalized: Tensor
    environment_gate: Tensor


class _InstanceNormalizer(nn.Module):
    """Per-window reversible normalization without learned affine parameters."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps

    def normalize(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean = x.mean(dim=1, keepdim=True).detach()
        variance = x.var(dim=1, keepdim=True, unbiased=False).detach()
        scale = torch.sqrt(variance + self.eps)
        return (x - mean) / scale, mean, scale

    @staticmethod
    def denormalize(x: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        return x * scale + mean


class _PatchSequenceEncoder(nn.Module):
    """PatchTST-style temporal patching followed by a Transformer encoder."""

    def __init__(
        self,
        input_dim: int,
        context_length: int,
        patch_length: int,
        patch_stride: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.context_length = context_length
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.num_patches = (
            max(0, context_length - patch_length) + patch_stride - 1
        ) // patch_stride + 1
        covered = (self.num_patches - 1) * patch_stride + patch_length
        self.pad_right = covered - context_length

        self.patch_projection = nn.Linear(input_dim * patch_length, d_model)
        self.position = nn.Parameter(torch.empty(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"history must have shape [B,L,C], got {tuple(x.shape)}")
        expected = (self.context_length, self.input_dim)
        if tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"expected history [B,{expected[0]},{expected[1]}], got {tuple(x.shape)}"
            )
        if self.pad_right:
            x = F.pad(x, (0, 0, 0, self.pad_right), mode="replicate")
        patches = x.unfold(1, self.patch_length, self.patch_stride)
        patches = patches.contiguous().flatten(start_dim=2)
        tokens = self.patch_projection(patches) + self.position
        return self.encoder(self.dropout(tokens))


def _decoder_stack(
    d_model: int,
    n_heads: int,
    d_ff: int,
    num_layers: int,
    dropout: float,
) -> nn.TransformerDecoder:
    layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_ff,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerDecoder(
        layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
    )


class _EnvWorldCore(nn.Module):
    """Direct latent-trajectory world model used by the framework adapter."""

    def __init__(
        self,
        context_length: int,
        horizon: int,
        target_dim: int,
        environment_dim: int,
        d_model: int,
        n_heads: int,
        history_layers: int,
        world_layers: int,
        target_layers: int,
        d_ff: int,
        dropout: float,
        patch_length: int,
        patch_stride: int,
        revin_eps: float,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.horizon = horizon
        self.target_dim = target_dim
        self.environment_dim = environment_dim

        self.target_normalizer = _InstanceNormalizer(revin_eps)
        self.environment_normalizer = _InstanceNormalizer(revin_eps)
        encoder_args = dict(
            context_length=context_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=history_layers,
            dropout=dropout,
        )
        self.environment_history_encoder = _PatchSequenceEncoder(
            input_dim=environment_dim, **encoder_args
        )
        self.target_history_encoder = _PatchSequenceEncoder(
            input_dim=target_dim, **encoder_args
        )

        self.horizon_tokens = nn.Parameter(torch.empty(1, horizon, d_model))
        nn.init.trunc_normal_(self.horizon_tokens, std=0.02)
        self.world_decoder = _decoder_stack(
            d_model, n_heads, d_ff, world_layers, dropout
        )
        self.environment_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, environment_dim),
        )
        self.environment_value_embedding = nn.Sequential(
            nn.Linear(environment_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.environment_condition_norm = nn.LayerNorm(d_model)
        self.environment_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self.memory_type = nn.Parameter(torch.empty(3, d_model))
        nn.init.trunc_normal_(self.memory_type, std=0.02)
        self.target_decoder = _decoder_stack(
            d_model, n_heads, d_ff, target_layers, dropout
        )
        self.target_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, target_dim),
        )

    def forward(
        self,
        past_target: Tensor,
        past_environment: Tensor,
        future_environment_override: Optional[Tensor] = None,
        environment_mix: float = 0.0,
        use_environment: bool = True,
    ) -> ForecastOutput:
        batch = past_target.shape[0]
        expected_target = (batch, self.context_length, self.target_dim)
        expected_environment = (batch, self.context_length, self.environment_dim)
        if tuple(past_target.shape) != expected_target:
            raise ValueError(
                f"past_target: expected {expected_target}, got {tuple(past_target.shape)}"
            )
        if tuple(past_environment.shape) != expected_environment:
            raise ValueError(
                "past_environment: expected "
                f"{expected_environment}, got {tuple(past_environment.shape)}"
            )
        if not 0.0 <= environment_mix <= 1.0:
            raise ValueError("environment_mix must be in [0, 1]")
        if future_environment_override is None and environment_mix != 0.0:
            raise ValueError("environment_mix requires future environment values")
        if future_environment_override is not None:
            expected_future = (batch, self.horizon, self.environment_dim)
            if tuple(future_environment_override.shape) != expected_future:
                raise ValueError(
                    "future_environment_override: expected "
                    f"{expected_future}, got {tuple(future_environment_override.shape)}"
                )

        target_norm, target_mean, target_scale = self.target_normalizer.normalize(
            past_target
        )
        env_norm, env_mean, env_scale = self.environment_normalizer.normalize(
            past_environment
        )
        env_memory = self.environment_history_encoder(env_norm)
        target_memory = self.target_history_encoder(target_norm)

        horizon_query = self.horizon_tokens.expand(batch, -1, -1)
        env_states = self.world_decoder(tgt=horizon_query, memory=env_memory)
        predicted_env_norm = self.environment_head(env_states)
        predicted_environment = self.environment_normalizer.denormalize(
            predicted_env_norm, env_mean, env_scale
        )

        if future_environment_override is None:
            env_used_norm = predicted_env_norm
        else:
            override_norm = (future_environment_override - env_mean) / env_scale
            env_used_norm = (
                (1.0 - environment_mix) * predicted_env_norm
                + environment_mix * override_norm
            )
        environment_used = self.environment_normalizer.denormalize(
            env_used_norm, env_mean, env_scale
        )

        env_condition = self.environment_condition_norm(
            env_states + self.environment_value_embedding(env_used_norm)
        )
        gate = self.environment_gate(torch.cat([horizon_query, env_condition], dim=-1))
        if not use_environment:
            gate = torch.zeros_like(gate)
        target_query = horizon_query + gate * env_condition

        typed_target_memory = target_memory + self.memory_type[0][None, None, :]
        typed_env_memory = env_memory + self.memory_type[1][None, None, :]
        typed_future_env = env_states + self.memory_type[2][None, None, :]
        joint_memory = torch.cat(
            [typed_target_memory, typed_env_memory, typed_future_env], dim=1
        )
        target_states = self.target_decoder(tgt=target_query, memory=joint_memory)

        # Persistence skip makes the nonlinear head learn changes around the latest value.
        target_pred_norm = self.target_head(target_states) + target_norm[:, -1:, :]
        target_prediction = self.target_normalizer.denormalize(
            target_pred_norm, target_mean, target_scale
        )
        return ForecastOutput(
            target=target_prediction,
            environment=predicted_environment,
            environment_used=environment_used,
            target_normalized=target_pred_norm,
            environment_normalized=predicted_env_norm,
            environment_gate=gate,
        )


def _config_value(configs, name: str, default):
    value = configs.get(name, default)
    return default if value is None else value


def _resolve_indices(indices: Sequence[int], channel_count: int) -> list[int]:
    resolved = []
    for raw_index in indices:
        index = int(raw_index)
        if index < 0:
            index += channel_count
        if not 0 <= index < channel_count:
            raise ValueError(
                f"target index {raw_index} is outside {channel_count} input channels"
            )
        resolved.append(index)
    if not resolved:
        raise ValueError("target_indices must contain at least one channel")
    if len(resolved) != len(set(resolved)):
        raise ValueError("target_indices contains duplicate channels")
    return resolved


class Model(nn.Module):
    """Repository-compatible adapter for the deterministic environment world model.

    Required configuration:
      * ``enc_in``: total number of numeric channels in ``x``;
      * ``target_indices``: target positions in the selected data columns.

    The remaining channels become environment variables.  ``forward`` returns all
    channels in their input order because the universal pipeline supervises ``x``
    and ``y`` with identical channel layouts.
    """

    supports_future_values = True

    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.channels = int(configs.enc_in)
        target_indices = _resolve_indices(
            _config_value(configs, "target_indices", [-1]), self.channels
        )
        environment_indices = [
            index for index in range(self.channels) if index not in target_indices
        ]
        if not environment_indices:
            raise ValueError(
                "EnvWorld needs at least one environment channel in addition to target_indices"
            )

        self.register_buffer(
            "target_indices", torch.tensor(target_indices, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "environment_indices",
            torch.tensor(environment_indices, dtype=torch.long),
            persistent=False,
        )
        self.target_weight = float(_config_value(configs, "target_weight", 1.0))
        self.environment_weight = float(
            _config_value(configs, "environment_weight", 0.3)
        )
        self.curriculum_max_mix = float(
            _config_value(configs, "curriculum_max_mix", 0.5)
        )
        self.curriculum_epochs = int(_config_value(configs, "curriculum_epochs", 0))
        if self.target_weight < 0.0 or self.environment_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if not 0.0 <= self.curriculum_max_mix <= 1.0:
            raise ValueError("curriculum_max_mix must be in [0, 1]")
        if self.curriculum_epochs < 0:
            raise ValueError("curriculum_epochs cannot be negative")
        self._environment_mix = 0.0

        d_model = int(_config_value(configs, "d_model", 128))
        n_heads = int(_config_value(configs, "n_heads", 4))
        patch_length = int(_config_value(configs, "patch_len", 16))
        patch_stride = int(_config_value(configs, "stride", 8))
        if patch_length > self.seq_len:
            raise ValueError("patch_len cannot exceed input_len")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.core = _EnvWorldCore(
            context_length=self.seq_len,
            horizon=self.pred_len,
            target_dim=len(target_indices),
            environment_dim=len(environment_indices),
            d_model=d_model,
            n_heads=n_heads,
            history_layers=int(_config_value(configs, "history_layers", 2)),
            world_layers=int(_config_value(configs, "world_layers", 2)),
            target_layers=int(_config_value(configs, "target_layers", 2)),
            d_ff=int(_config_value(configs, "d_ff", 256)),
            dropout=float(_config_value(configs, "dropout", 0.1)),
            patch_length=patch_length,
            patch_stride=patch_stride,
            revin_eps=float(_config_value(configs, "revin_eps", 1e-5)),
        )

    @property
    def environment_mix(self) -> float:
        return self._environment_mix

    def set_train_epoch(self, epoch: int) -> None:
        """Linearly remove future-environment conditioning during early epochs."""

        if self.curriculum_epochs <= 0 or epoch >= self.curriculum_epochs:
            self._environment_mix = 0.0
        else:
            remaining = 1.0 - float(epoch + 1) / float(self.curriculum_epochs)
            self._environment_mix = self.curriculum_max_mix * max(0.0, remaining)

    def forward(
        self,
        x: Tensor,
        future_values: Optional[Tensor] = None,
        future_environment_override: Optional[Tensor] = None,
        environment_mix: Optional[float] = None,
        use_environment: bool = True,
        return_components: bool = False,
        **kwargs,
    ):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.seq_len, self.channels):
            raise ValueError(
                f"x must have shape [B,{self.seq_len},{self.channels}], got {tuple(x.shape)}"
            )
        past_target = x.index_select(2, self.target_indices)
        past_environment = x.index_select(2, self.environment_indices)

        if future_values is not None and future_environment_override is not None:
            raise ValueError(
                "pass either future_values for training curriculum or "
                "future_environment_override for a scenario, not both"
            )

        future_environment = future_environment_override
        mix = 0.0
        if future_environment_override is not None:
            expected_environment = (
                x.shape[0],
                self.pred_len,
                self.environment_indices.numel(),
            )
            if tuple(future_environment_override.shape) != expected_environment:
                raise ValueError(
                    "future_environment_override must have shape "
                    f"{expected_environment}, got {tuple(future_environment_override.shape)}"
                )
            mix = 1.0 if environment_mix is None else float(environment_mix)
        elif future_values is not None and self.training and self._environment_mix > 0.0:
            expected = (x.shape[0], self.pred_len, self.channels)
            if tuple(future_values.shape) != expected:
                raise ValueError(
                    f"future_values must have shape {expected}, got {tuple(future_values.shape)}"
                )
            future_environment = future_values.index_select(
                2, self.environment_indices
            )
            mix = self._environment_mix
        elif environment_mix not in (None, 0.0):
            raise ValueError("environment_mix requires future_environment_override")

        details = self.core(
            past_target=past_target,
            past_environment=past_environment,
            future_environment_override=future_environment,
            environment_mix=mix,
            use_environment=use_environment,
        )
        prediction = x.new_zeros(x.shape[0], self.pred_len, self.channels)
        prediction = prediction.index_copy(2, self.target_indices, details.target)
        prediction = prediction.index_copy(
            2, self.environment_indices, details.environment
        )
        if return_components:
            return prediction, details
        return prediction

    def compute_loss(self, prediction: Tensor, truth: Tensor, criterion) -> Tensor:
        """Weighted joint objective used by the optional experiment hook."""
        target_loss = criterion(
            prediction.index_select(2, self.target_indices),
            truth.index_select(2, self.target_indices),
        )
        environment_loss = criterion(
            prediction.index_select(2, self.environment_indices),
            truth.index_select(2, self.environment_indices),
        )
        return (
            self.target_weight * target_loss
            + self.environment_weight * environment_loss
        )


__all__ = ["Model", "ForecastOutput"]
