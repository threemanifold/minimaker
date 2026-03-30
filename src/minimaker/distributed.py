"""Distributed training setup — FSDP wrapping, device detection, process group."""

from __future__ import annotations

import functools
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
    if not torch.cuda.is_available() or "RANK" not in os.environ:
        return 0, 1

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend=cfg.distributed.backend)
    torch.cuda.set_device(local_rank)

    return rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_with_fsdp(
    model: torch.nn.Module,
    cfg: DictConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Wrap model with FSDP + optional mixed precision and activation checkpointing."""
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from minimaker.model import TransformerBlock

    # --- Mixed precision policy ---
    mp_cfg = cfg.training.mixed_precision
    mp_policy = None
    if mp_cfg == "bf16":
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif mp_cfg == "fp16":
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    # --- Wrap policy: one FSDP unit per TransformerBlock ---
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # --- Sharding strategy ---
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    strategy = strategy_map.get(
        cfg.distributed.fsdp.sharding_strategy, ShardingStrategy.FULL_SHARD
    )

    model = FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        auto_wrap_policy=wrap_policy,
        device_id=device,
        use_orig_params=True,  # required for torch.compile compatibility
    )

    # --- Activation checkpointing ---
    if cfg.distributed.fsdp.activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
        )

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=lambda m: isinstance(m, TransformerBlock),
        )

    return model
