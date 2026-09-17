"""Dataset-level perturbations for environment-variable ablations."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def shuffle_environment_batch(
    history: torch.Tensor,
    future: torch.Tensor,
    environment_indices: Sequence[int],
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shuffle complete environment trajectories between samples in a batch.

    A single donor permutation is used for both the historical environment in
    ``history`` and the future environment labels in ``future``.  This destroys
    the target/environment pairing while keeping the auxiliary environment
    forecasting task internally consistent.  Target channels are never changed.

    Returns the shuffled history, shuffled future labels, and donor indices.
    """

    if not torch.is_tensor(history) or not torch.is_tensor(future):
        raise TypeError("history and future must be torch tensors")
    if history.ndim != 3 or future.ndim != 3:
        raise ValueError("history and future must have shape [batch, time, channel]")
    if history.shape[0] != future.shape[0]:
        raise ValueError("history and future batch sizes must match")
    if history.shape[2] != future.shape[2]:
        raise ValueError("history and future channel counts must match")

    batch_size = history.shape[0]
    if batch_size < 2:
        raise ValueError("environment shuffling needs a batch size of at least two")

    channel_indices = torch.as_tensor(
        tuple(int(i) for i in environment_indices),
        dtype=torch.long,
        device=history.device,
    )
    if channel_indices.numel() == 0:
        raise ValueError("environment_indices cannot be empty")
    if torch.any(channel_indices < 0) or torch.any(
        channel_indices >= history.shape[2]
    ):
        raise IndexError("environment index is outside the channel dimension")

    # A non-zero cyclic shift is a derangement: no sample keeps its own env.
    shift = int(
        torch.randint(1, batch_size, (1,), generator=generator).item()
    )
    donor_indices_cpu = torch.roll(torch.arange(batch_size), shifts=shift)
    donor_indices = donor_indices_cpu.to(history.device)

    shuffled_history = history.clone()
    shuffled_future = future.clone()
    history_donors = history.index_select(0, donor_indices)
    future_donors = future.index_select(0, donor_indices.to(future.device))
    future_channel_indices = channel_indices.to(future.device)
    shuffled_history[:, :, channel_indices] = history_donors[:, :, channel_indices]
    shuffled_future[:, :, future_channel_indices] = future_donors[
        :, :, future_channel_indices
    ]

    return shuffled_history, shuffled_future, donor_indices_cpu


def zero_environment_batch(
    history: torch.Tensor,
    environment_indices: Sequence[int],
) -> torch.Tensor:
    """Return a batch whose historical environment channels are all zero.

    Only model inputs are changed.  Future labels are intentionally left to the
    caller so target supervision remains attached to the original sample.
    """

    if not torch.is_tensor(history):
        raise TypeError("history must be a torch tensor")
    if history.ndim != 3:
        raise ValueError("history must have shape [batch, time, channel]")

    channel_indices = torch.as_tensor(
        tuple(int(i) for i in environment_indices),
        dtype=torch.long,
        device=history.device,
    )
    if channel_indices.numel() == 0:
        raise ValueError("environment_indices cannot be empty")
    if torch.any(channel_indices < 0) or torch.any(
        channel_indices >= history.shape[2]
    ):
        raise IndexError("environment index is outside the channel dimension")

    zeroed_history = history.clone()
    zeroed_history[:, :, channel_indices] = 0
    return zeroed_history


def make_derangement(length: int, seed: int) -> tuple[int, ...]:
    """Return a reproducible permutation in which no sample keeps its own index.

    Sattolo's algorithm creates one cycle, so every target sample receives its
    environment history from a different sample.  Unlike a one-step shift, the
    random cycle does not systematically pair highly overlapping neighbours.
    """

    if length < 2:
        raise ValueError(
            "Shuffled-environment evaluation needs at least two samples"
        )

    generator = torch.Generator().manual_seed(int(seed))
    permutation = list(range(length))
    for index in range(length - 1, 0, -1):
        donor = int(
            torch.randint(0, index, (1,), generator=generator).item()
        )
        permutation[index], permutation[donor] = (
            permutation[donor],
            permutation[index],
        )

    return tuple(permutation)


class ShuffledEnvironmentDataset(Dataset):
    """Keep each target sample intact while replacing its environment history.

    The replacement is performed across complete samples.  All environment
    channels therefore come from the same donor, and the donor's temporal order
    and cross-variable relationships are preserved.  Labels and all non-history
    fields always remain attached to the original target sample.
    """

    def __init__(
        self,
        dataset: Dataset,
        environment_indices: Sequence[int],
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.environment_indices = tuple(int(i) for i in environment_indices)
        if not self.environment_indices:
            raise ValueError("environment_indices cannot be empty")
        self.permutation = make_derangement(len(dataset), seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        original = self.dataset[index]
        donor = self.dataset[self.permutation[index]]
        if not isinstance(original, (tuple, list)) or not isinstance(
            donor, (tuple, list)
        ):
            raise TypeError("The wrapped dataset must return a tuple or list")

        original_fields = list(original)
        original_history = original_fields[0]
        donor_history = donor[0]

        if torch.is_tensor(original_history):
            shuffled_history = original_history.clone()
            channel_indices = torch.as_tensor(
                self.environment_indices,
                dtype=torch.long,
                device=shuffled_history.device,
            )
            donor_history = torch.as_tensor(
                donor_history,
                dtype=shuffled_history.dtype,
                device=shuffled_history.device,
            )
            shuffled_history[:, channel_indices] = donor_history.index_select(
                1, channel_indices
            )
        else:
            shuffled_history = np.array(original_history, copy=True)
            donor_history = np.asarray(donor_history)
            shuffled_history[:, list(self.environment_indices)] = donor_history[
                :, list(self.environment_indices)
            ]

        original_fields[0] = shuffled_history
        return tuple(original_fields)


class ZeroEnvironmentDataset(Dataset):
    """Replace environment history with zero and preserve target data/labels."""

    def __init__(
        self,
        dataset: Dataset,
        environment_indices: Sequence[int],
    ) -> None:
        self.dataset = dataset
        self.environment_indices = tuple(int(i) for i in environment_indices)
        if not self.environment_indices:
            raise ValueError("environment_indices cannot be empty")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        original = self.dataset[index]
        if not isinstance(original, (tuple, list)):
            raise TypeError("The wrapped dataset must return a tuple or list")

        original_fields = list(original)
        original_history = original_fields[0]
        if torch.is_tensor(original_history):
            zeroed_history = original_history.clone()
            channel_indices = torch.as_tensor(
                self.environment_indices,
                dtype=torch.long,
                device=zeroed_history.device,
            )
            zeroed_history[:, channel_indices] = 0
        else:
            zeroed_history = np.array(original_history, copy=True)
            zeroed_history[:, list(self.environment_indices)] = 0

        original_fields[0] = zeroed_history
        return tuple(original_fields)


__all__ = [
    "ShuffledEnvironmentDataset",
    "ZeroEnvironmentDataset",
    "make_derangement",
    "shuffle_environment_batch",
    "zero_environment_batch",
]
