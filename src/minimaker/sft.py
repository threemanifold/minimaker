"""Supervised fine-tuning — train on instruction/response pairs."""

from __future__ import annotations

import math
from contextlib import nullcontext

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from minimaker.model import GPT
from minimaker.sft_data import build_sft_dataloader
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
# Training
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../configs", config_name="sft")
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

    # Load pretrained weights if specified
    if cfg.get("pretrained_checkpoint"):
        ckpt = torch.load(cfg.pretrained_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if rank == 0:
            print(f"Loaded pretrained weights from {cfg.pretrained_checkpoint}")

    if not is_distributed and cfg.distributed.fsdp.activation_checkpointing:
        model.enable_activation_checkpointing()

    if is_distributed:
        model = wrap_with_fsdp(model, cfg, device)

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
    train_loader = build_sft_dataloader(cfg, rank, world_size)

    # ---- Metrics ----
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    tracker = MetricsTracker(cfg, output_dir, rank)
    timer = StepTimer(device)

    # ---- Resume ----
    start_step = load_checkpoint(model, optimizer, output_dir, rank, is_distributed)

    # ---- Mixed precision ----
    mp = cfg.training.mixed_precision
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mp)
    use_amp = amp_dtype is not None and device.type in ("cuda", "cpu")
    amp_ctx = torch.amp.autocast(device.type, dtype=amp_dtype) if use_amp else nullcontext()

    scaler = None
    if mp == "fp16" and not is_distributed and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    # ---- Training loop ----
    tcfg = cfg.training
    grad_accum = tcfg.gradient_accumulation_steps
    tokens_per_step = tcfg.batch_size * cfg.data.seq_len * grad_accum * world_size
    data_iter = iter(train_loader)

    for step in range(start_step, tcfg.max_steps):
        timer.reset()
        model.train()

        lr = get_lr(step, tcfg.warmup_steps, tcfg.max_steps, tcfg.lr, tcfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(grad_accum):
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
            "sft/loss": accum_loss,
            "sft/lr": lr,
            "sft/step": step,
            "throughput/tokens_per_sec": tokens_per_step / (timer.total_ms / 1000)
            if timer.total_ms > 0
            else 0,
        }
        for phase, ms in timer.timings.items():
            metrics[f"timing/{phase}_ms"] = ms
        metrics["timing/total_ms"] = timer.total_ms
        metrics.update(get_cuda_memory_stats(device))
        tracker.log(metrics, step)

        # ---- Checkpoint ----
        if tcfg.checkpoint.every > 0 and step > 0 and step % tcfg.checkpoint.every == 0:
            save_checkpoint(model, optimizer, step, output_dir, rank, is_distributed)
            cleanup_checkpoints(output_dir, tcfg.checkpoint.keep)

    save_checkpoint(model, optimizer, tcfg.max_steps, output_dir, rank, is_distributed)
    tracker.finish()
    cleanup_distributed()

    if rank == 0:
        print("SFT complete.")


if __name__ == "__main__":
    main()
