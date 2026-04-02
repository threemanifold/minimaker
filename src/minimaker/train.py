"""Main training loop — Hydra entry point."""

from __future__ import annotations

import math
from contextlib import nullcontext

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from minimaker.model import GPT
from minimaker.data import build_dataloader
from minimaker.distributed import (
    get_device,
    setup_distributed,
    cleanup_distributed,
    wrap_with_fsdp,
)
from minimaker.metrics import StepTimer, MetricsTracker, get_cuda_memory_stats
from minimaker.checkpoint import save_checkpoint, load_checkpoint, cleanup_checkpoints


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# GPU peak FLOPS (bf16) for MFU calculation
# ---------------------------------------------------------------------------

_GPU_PEAK_FLOPS_BF16: dict[str, float] = {
    "H100": 990e12,
    "H200": 990e12,
    "A100": 312e12,
    "L40S": 733e12,
    "L40": 181e12,
    "A10G": 70e12,
    "V100": 125e12,
}


def get_gpu_peak_flops(device: torch.device) -> float | None:
    """Return peak bf16 FLOPS for the GPU, or None if unknown/not CUDA."""
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device)
    for gpu, flops in _GPU_PEAK_FLOPS_BF16.items():
        if gpu in name:
            return flops
    return None


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, steps: int, amp_ctx) -> float:
    model.eval()
    total_loss, count = 0.0, 0
    data_iter = iter(loader)
    for _ in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with amp_ctx:
            _, loss = model(ids, labels)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    rank, world_size = setup_distributed(cfg)
    device = get_device()
    is_distributed = world_size > 1

    torch.manual_seed(cfg.seed + rank)

    if rank == 0:
        print(OmegaConf.to_yaml(cfg))

    # ---- Model ----
    model = GPT(cfg.model).to(device)
    if rank == 0:
        print(f"Model: {cfg.model.name} | {model.param_count():,} parameters")

    # Activation checkpointing (single-GPU path)
    if not is_distributed and cfg.distributed.fsdp.activation_checkpointing:
        model.enable_activation_checkpointing()

    # FSDP wrapping
    if is_distributed:
        model = wrap_with_fsdp(model, cfg, device)

    # torch.compile (CUDA only)
    if cfg.compile and device.type == "cuda":
        model = torch.compile(model)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )

    # ---- Data ----
    train_loader = build_dataloader(cfg, rank, world_size, split="train")
    eval_loader = build_dataloader(cfg, rank, world_size, split="val")

    # ---- Metrics ----
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    tracker = MetricsTracker(cfg, output_dir, rank)
    timer = StepTimer(device)

    # ---- Resume ----
    start_step = load_checkpoint(model, optimizer, output_dir, rank, is_distributed)

    # ---- Mixed precision ----
    mp = cfg.training.mixed_precision
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mp)

    # autocast: CUDA and CPU support bf16; MPS has limited support
    use_amp = amp_dtype is not None and device.type in ("cuda", "cpu")
    amp_ctx = torch.amp.autocast(device.type, dtype=amp_dtype) if use_amp else nullcontext()

    # GradScaler only for fp16 on single GPU (FSDP handles its own scaling)
    scaler = None
    if mp == "fp16" and not is_distributed and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    # ---- Training loop ----
    tcfg = cfg.training
    grad_accum = tcfg.gradient_accumulation_steps
    tokens_per_step = tcfg.batch_size * cfg.data.seq_len * grad_accum * world_size
    epoch = 0
    data_iter = iter(train_loader)
    flops_per_token = model.module.flops_per_token() if is_distributed else model.flops_per_token()
    peak_flops = get_gpu_peak_flops(device)

    for step in range(start_step, tcfg.max_steps):
        timer.reset()
        model.train()

        # Set LR
        lr = get_lr(step, tcfg.warmup_steps, tcfg.max_steps, tcfg.lr, tcfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(grad_accum):
            # Skip gradient sync on intermediate micro-steps
            no_sync = (
                is_distributed
                and micro < grad_accum - 1
                and hasattr(model, "no_sync")
            )
            sync_ctx = model.no_sync() if no_sync else nullcontext()

            with timer.track("data_load"):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    if hasattr(train_loader.sampler, "set_epoch"):
                        train_loader.sampler.set_epoch(epoch)
                    data_iter = iter(train_loader)
                    batch = next(data_iter)
                ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

            with sync_ctx:
                with timer.track("forward"):
                    with amp_ctx:
                        _, loss = model(ids, labels)
                        loss = loss / grad_accum

                with timer.track("backward"):
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

            accum_loss += loss.item()

        with timer.track("optimizer"):
            if tcfg.grad_clip > 0:
                if scaler:
                    scaler.unscale_(optimizer)
                if is_distributed:
                    model.clip_grad_norm_(tcfg.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)

            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        # ---- Log ----
        metrics = {
            "train/loss": accum_loss,
            "train/lr": lr,
            "train/step": step,
            "throughput/tokens_per_sec": tokens_per_step / (timer.total_ms / 1000)
            if timer.total_ms > 0
            else 0,
        }
        # MFU — fraction of peak hardware FLOPS
        if timer.total_ms > 0 and peak_flops is not None:
            achieved_flops = flops_per_token * tokens_per_step / (timer.total_ms / 1000)
            metrics["throughput/mfu"] = achieved_flops / peak_flops

        for phase, ms in timer.timings.items():
            metrics[f"timing/{phase}_ms"] = ms
            if ms > 0:
                metrics[f"throughput/{phase}_tok_per_sec"] = tokens_per_step / (ms / 1000)
        metrics["timing/total_ms"] = timer.total_ms

        metrics.update(get_cuda_memory_stats(device))
        tracker.log(metrics, step)

        # ---- Eval ----
        if tcfg.eval.every > 0 and step > 0 and step % tcfg.eval.every == 0:
            eval_loss = evaluate(model, eval_loader, device, tcfg.eval.steps, amp_ctx)
            tracker.log({"eval/loss": eval_loss}, step)

        # ---- Checkpoint ----
        if tcfg.checkpoint.every > 0 and step > 0 and step % tcfg.checkpoint.every == 0:
            save_checkpoint(model, optimizer, step, output_dir, rank, is_distributed)
            cleanup_checkpoints(output_dir, tcfg.checkpoint.keep)

    # Final checkpoint
    save_checkpoint(model, optimizer, tcfg.max_steps, output_dir, rank, is_distributed)
    tracker.finish()
    cleanup_distributed()

    if rank == 0:
        print("Training complete.")


if __name__ == "__main__":
    main()
