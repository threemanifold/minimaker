"""Checkpoint save / load / cleanup — supports both single-GPU and FSDP."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: str,
    rank: int,
    is_distributed: bool,
) -> None:
    ckpt_dir = Path(output_dir) / "checkpoints" / f"step_{step}"

    if is_distributed:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            FullStateDictConfig,
        )

        # Gather full state dicts to rank 0
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state = model.state_dict()
        optim_state = FSDP.full_optim_state_dict(model, optimizer)

        if rank == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model_state, "optimizer": optim_state, "step": step},
                ckpt_dir / "checkpoint.pt",
            )

        dist.barrier()
    else:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            },
            ckpt_dir / "checkpoint.pt",
        )


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: str,
    rank: int,
    is_distributed: bool,
) -> int:
    """Load latest checkpoint if it exists. Returns the step to resume from."""
    ckpt_root = Path(output_dir) / "checkpoints"
    if not ckpt_root.exists():
        return 0

    step_dirs = [
        d for d in ckpt_root.iterdir()
        if d.is_dir() and d.name.startswith("step_")
    ]
    if not step_dirs:
        return 0

    latest_step = max(int(d.name.split("_")[1]) for d in step_dirs)
    ckpt_path = ckpt_root / f"step_{latest_step}" / "checkpoint.pt"
    if not ckpt_path.exists():
        return 0

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if is_distributed:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            model.load_state_dict(checkpoint["model"])

        sharded_osd = FSDP.shard_full_optim_state_dict(
            checkpoint["optimizer"], model
        )
        optimizer.load_state_dict(sharded_osd)
    else:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    if rank == 0:
        print(f"Resumed from checkpoint at step {latest_step}")
    return latest_step


def cleanup_checkpoints(output_dir: str, keep: int) -> None:
    """Delete oldest checkpoints, keeping the most recent `keep`."""
    ckpt_root = Path(output_dir) / "checkpoints"
    if not ckpt_root.exists():
        return

    step_dirs = sorted(
        [d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    for old_dir in step_dirs[:-keep]:
        for f in old_dir.iterdir():
            f.unlink()
        old_dir.rmdir()
