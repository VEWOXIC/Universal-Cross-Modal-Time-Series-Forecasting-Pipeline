"""Parallel environment-target forecasting with joint latent interaction.

The model splits a multivariate history into environment and target channels,
encodes every channel as temporal patches, and directly decodes the complete
future of all channels. Environment and target future tokens interact in the
latent space, but neither prediction head consumes the numeric output of the
other head. This avoids the environment-to-target error cascade of a serial
environment-first model while retaining environment forecasting as an
auxiliary task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ParallelForecastOutput:
    """Detailed outputs returned when ``return_components=True``."""

    target: Tensor
    environment: Tensor
    target_normalized: Tensor
    environment_normalized: Tensor
    future_states: Tensor


class _InstanceNormalizer(nn.Module):
    """Per-window, per-channel reversible normalization."""

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


class _FactorizedSelfAttentionBlock(nn.Module):
    """Mix temporal and variable axes without full (time * variable)^2 cost."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.variable_norm = nn.LayerNorm(d_model)
        self.variable_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N_time, C, D]
        batch, steps, channels, width = x.shape

        temporal = x.permute(0, 2, 1, 3).reshape(batch * channels, steps, width)
        temporal_input = self.temporal_norm(temporal)
        temporal_update, _ = self.temporal_attention(
            temporal_input, temporal_input, temporal_input, need_weights=False
        )
        temporal = temporal + self.dropout(temporal_update)
        x = temporal.reshape(batch, channels, steps, width).permute(0, 2, 1, 3)

        variable = x.reshape(batch * steps, channels, width)
        variable_input = self.variable_norm(variable)
        variable_update, _ = self.variable_attention(
            variable_input, variable_input, variable_input, need_weights=False
        )
        variable = variable + self.dropout(variable_update)
        x = variable.reshape(batch, steps, channels, width)

        return x + self.dropout(self.ff(self.ff_norm(x)))


class _JointHistoryEncoder(nn.Module):
    """Patch and jointly encode all environment and target histories."""

    def __init__(
        self,
        context_length: int,
        channel_count: int,
        role_ids: Tensor,
        patch_length: int,
        patch_stride: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.channel_count = channel_count
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.num_patches = (
            max(0, context_length - patch_length) + patch_stride - 1
        ) // patch_stride + 1
        covered = (self.num_patches - 1) * patch_stride + patch_length
        self.pad_right = covered - context_length

        self.patch_projection = nn.Linear(patch_length, d_model)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.num_patches, 1, d_model)
        )
        self.variable_embedding = nn.Parameter(
            torch.empty(1, 1, channel_count, d_model)
        )
        self.role_embedding = nn.Embedding(2, d_model)
        self.register_buffer("role_ids", role_ids.clone(), persistent=False)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                _FactorizedSelfAttentionBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.variable_embedding, std=0.02)
        nn.init.trunc_normal_(self.role_embedding.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"history must have shape [B,L,C], got {tuple(x.shape)}")
        expected = (self.context_length, self.channel_count)
        if tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"expected history [B,{expected[0]},{expected[1]}], got {tuple(x.shape)}"
            )

        channel_first = x.transpose(1, 2)
        if self.pad_right:
            channel_first = F.pad(
                channel_first, (0, self.pad_right), mode="replicate"
            )
        patches = channel_first.unfold(
            dimension=-1, size=self.patch_length, step=self.patch_stride
        )
        patches = patches.permute(0, 2, 1, 3).contiguous()

        role = self.role_embedding(self.role_ids)[None, None, :, :]
        tokens = (
            self.patch_projection(patches)
            + self.position_embedding
            + self.variable_embedding
            + role
        )
        tokens = self.dropout(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        return self.final_norm(tokens)


class _JointFutureBlock(nn.Module):
    """Query history first, then mix future variables and horizon positions."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.variable_norm = nn.LayerNorm(d_model)
        self.variable_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.horizon_norm = nn.LayerNorm(d_model)
        self.horizon_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        # x: [B, N_future, C, D], memory: [B, N_history * C, D]
        batch, steps, channels, width = x.shape

        flat = x.reshape(batch, steps * channels, width)
        normalized_memory = self.memory_norm(memory)
        cross_update, _ = self.cross_attention(
            self.query_norm(flat),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        flat = flat + self.dropout(cross_update)
        x = flat.reshape(batch, steps, channels, width)

        variable = x.reshape(batch * steps, channels, width)
        variable_input = self.variable_norm(variable)
        variable_update, _ = self.variable_attention(
            variable_input, variable_input, variable_input, need_weights=False
        )
        variable = variable + self.dropout(variable_update)
        x = variable.reshape(batch, steps, channels, width)

        horizon = x.permute(0, 2, 1, 3).reshape(batch * channels, steps, width)
        horizon_input = self.horizon_norm(horizon)
        horizon_update, _ = self.horizon_attention(
            horizon_input, horizon_input, horizon_input, need_weights=False
        )
        horizon = horizon + self.dropout(horizon_update)
        x = horizon.reshape(batch, channels, steps, width).permute(0, 2, 1, 3)

        return x + self.dropout(self.ff(self.ff_norm(x)))


class _ParallelEnvTargetCore(nn.Module):
    """Direct multi-horizon decoder with parallel environment/target heads."""

    def __init__(
        self,
        context_length: int,
        horizon: int,
        channel_count: int,
        target_indices: Tensor,
        environment_indices: Tensor,
        d_model: int,
        n_heads: int,
        d_ff: int,
        history_layers: int,
        decoder_layers: int,
        dropout: float,
        patch_length: int,
        patch_stride: int,
        revin_eps: float,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.horizon = horizon
        self.channel_count = channel_count
        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.num_future_patches = (horizon + patch_stride - 1) // patch_stride
        self.register_buffer(
            "target_indices", target_indices.clone(), persistent=False
        )
        self.register_buffer(
            "environment_indices", environment_indices.clone(), persistent=False
        )

        role_ids = torch.zeros(channel_count, dtype=torch.long)
        role_ids[target_indices] = 1
        self.normalizer = _InstanceNormalizer(revin_eps)
        self.history_encoder = _JointHistoryEncoder(
            context_length=context_length,
            channel_count=channel_count,
            role_ids=role_ids,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_layers=history_layers,
            dropout=dropout,
        )

        self.future_position = nn.Parameter(
            torch.empty(1, self.num_future_patches, 1, d_model)
        )
        self.recent_patch_projection = nn.Linear(patch_length, d_model)
        self.query_dropout = nn.Dropout(dropout)
        self.future_blocks = nn.ModuleList(
            [
                _JointFutureBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(decoder_layers)
            ]
        )
        self.future_norm = nn.LayerNorm(d_model)
        self.environment_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, patch_length),
        )
        self.target_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, patch_length),
        )
        nn.init.trunc_normal_(self.future_position, std=0.02)

    def _future_queries(self, normalized_history: Tensor) -> Tensor:
        batch = normalized_history.shape[0]
        recent_patch = normalized_history[:, -self.patch_length :, :]
        recent = self.recent_patch_projection(recent_patch.permute(0, 2, 1))
        recent = recent[:, None, :, :]
        variable = self.history_encoder.variable_embedding
        role = self.history_encoder.role_embedding(
            self.history_encoder.role_ids
        )[None, None, :, :]
        queries = self.future_position + variable + role + recent
        return self.query_dropout(queries.expand(batch, -1, -1, -1))

    def _overlap_add(self, patches: Tensor) -> Tensor:
        # patches: [B, N_future, C, patch_length]
        batch, patch_count, channels, patch_length = patches.shape
        total_length = (patch_count - 1) * self.patch_stride + patch_length
        fold_input = patches.permute(0, 2, 3, 1).reshape(
            batch * channels, patch_length, patch_count
        )
        output = F.fold(
            fold_input,
            output_size=(1, total_length),
            kernel_size=(1, patch_length),
            stride=(1, self.patch_stride),
        )
        count = F.fold(
            torch.ones_like(fold_input),
            output_size=(1, total_length),
            kernel_size=(1, patch_length),
            stride=(1, self.patch_stride),
        )
        output = (output / count).reshape(batch, channels, total_length)
        return output[:, :, : self.horizon].transpose(1, 2)

    def forward(self, history: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        normalized, mean, scale = self.normalizer.normalize(history)
        history_states = self.history_encoder(normalized)
        memory = history_states.reshape(history.shape[0], -1, history_states.shape[-1])

        future_states = self._future_queries(normalized)
        for block in self.future_blocks:
            future_states = block(future_states, memory)
        future_states = self.future_norm(future_states)

        environment_states = future_states.index_select(2, self.environment_indices)
        target_states = future_states.index_select(2, self.target_indices)
        environment_patches = self.environment_head(environment_states)
        target_patches = self.target_head(target_states)

        all_patches = future_states.new_zeros(
            future_states.shape[0],
            self.num_future_patches,
            self.channel_count,
            self.patch_length,
        )
        all_patches = all_patches.index_copy(
            2, self.environment_indices, environment_patches
        )
        all_patches = all_patches.index_copy(2, self.target_indices, target_patches)

        # Both heads learn changes around a persistence baseline.
        prediction_normalized = self._overlap_add(all_patches)
        prediction_normalized = prediction_normalized + normalized[:, -1:, :]
        prediction = self.normalizer.denormalize(prediction_normalized, mean, scale)
        return prediction, prediction_normalized, future_states


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
    """Repository adapter for parallel environment-target forecasting."""

    supports_future_values = False

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
                "ParallelEnvTarget needs at least one environment channel"
            )
        self.register_buffer(
            "target_indices",
            torch.tensor(target_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "environment_indices",
            torch.tensor(environment_indices, dtype=torch.long),
            persistent=False,
        )

        self.target_weight = float(_config_value(configs, "target_weight", 1.0))
        self.environment_weight = float(
            _config_value(configs, "environment_weight", 0.1)
        )
        if self.target_weight < 0.0 or self.environment_weight < 0.0:
            raise ValueError("loss weights must be non-negative")

        d_model = int(_config_value(configs, "d_model", 128))
        n_heads = int(_config_value(configs, "n_heads", 4))
        d_ff = int(_config_value(configs, "d_ff", 4 * d_model))
        history_layers = int(_config_value(configs, "history_layers", 2))
        decoder_layers = int(_config_value(configs, "decoder_layers", 3))
        dropout = float(_config_value(configs, "dropout", 0.1))
        patch_length = int(_config_value(configs, "patch_len", 12))
        patch_stride = int(_config_value(configs, "stride", 6))
        if not 0 < patch_length <= self.seq_len:
            raise ValueError("patch_len must be positive and cannot exceed input_len")
        if not 0 < patch_stride <= patch_length:
            raise ValueError("stride must be positive and cannot exceed patch_len")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if history_layers <= 0 or decoder_layers <= 0:
            raise ValueError("history_layers and decoder_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.core = _ParallelEnvTargetCore(
            context_length=self.seq_len,
            horizon=self.pred_len,
            channel_count=self.channels,
            target_indices=self.target_indices,
            environment_indices=self.environment_indices,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            history_layers=history_layers,
            decoder_layers=decoder_layers,
            dropout=dropout,
            patch_length=patch_length,
            patch_stride=patch_stride,
            revin_eps=float(_config_value(configs, "revin_eps", 1e-5)),
        )

    def forward(self, x: Tensor, return_components: bool = False, **kwargs):
        expected = (self.seq_len, self.channels)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"x must have shape [B,{expected[0]},{expected[1]}], got {tuple(x.shape)}"
            )
        prediction, normalized, future_states = self.core(x)
        if not return_components:
            return prediction

        details = ParallelForecastOutput(
            target=prediction.index_select(2, self.target_indices),
            environment=prediction.index_select(2, self.environment_indices),
            target_normalized=normalized.index_select(2, self.target_indices),
            environment_normalized=normalized.index_select(
                2, self.environment_indices
            ),
            future_states=future_states,
        )
        return prediction, details

    def compute_loss(self, prediction: Tensor, truth: Tensor, criterion) -> Tensor:
        """Combine separately averaged target and environment objectives."""

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


__all__ = ["Model", "ParallelForecastOutput"]
