"""Distributed training setup — FSDP2 sharding, device detection, process group."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from omegaconf import DictConfig


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_distributed(cfg: DictConfig) -> tuple[int, int]:
    """Init process group. Returns (rank, world_size). No-op for single GPU / MPS."""
    if "RANK" not in os.environ:
        return 0, 1

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"torchrun launched {world_size} workers but torch.cuda.is_available() "
            "is False. This usually means your PyTorch was compiled for a newer CUDA "
            "version than your driver supports. Check `python -c 'import torch; "
            "print(torch.version.cuda)'` and install a matching version."
        )

    dist.init_process_group(backend=cfg.distributed.backend)
    torch.cuda.set_device(local_rank)

    return rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def apply_fsdp2(
    model: torch.nn.Module,
    cfg: DictConfig,
) -> None:
    """Apply FSDP2 sharding in-place: shard each TransformerBlock, then the root."""
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from minimaker.model import TransformerBlock

    # --- Mixed precision policy ---
    mp_cfg = cfg.training.mixed_precision
    mp_policy = None
    if mp_cfg == "bf16":
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        )
    elif mp_cfg == "fp16":
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
        )

    # --- Sharding strategy ---
    # FSDP2: reshard_after_forward=True  → FULL_SHARD (default)
    #         reshard_after_forward=False → SHARD_GRAD_OP
    strategy = cfg.distributed.fsdp.sharding_strategy
    reshard_after_forward = strategy != "SHARD_GRAD_OP"

    fsdp_kwargs: dict = {"reshard_after_forward": reshard_after_forward}
    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy

    # --- Shard each transformer block, then the root model ---
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    # --- Activation checkpointing ---
    if cfg.distributed.fsdp.activation_checkpointing:
        model.enable_activation_checkpointing()
