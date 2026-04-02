"""DPO — Direct Preference Optimization.

Simpler alternative to PPO-based RLHF. Trains directly on preference pairs
without a separate reward model.

Loss: -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
"""

from __future__ import annotations

import copy
import math
from contextlib import nullcontext

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from minimaker.model import GPT
from minimaker.rl_data import build_preference_dataloader
from minimaker.distributed import (
    get_device,
    setup_distributed,
    cleanup_distributed,
    wrap_with_fsdp,
)
from minimaker.metrics import StepTimer, MetricsTracker, get_cuda_memory_stats
from minimaker.checkpoint import save_checkpoint, load_checkpoint, cleanup_checkpoints


def get_lr(step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def sequence_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    amp_ctx,
) -> torch.Tensor:
    """Sum of log-probs over response tokens (after prompt_len) for each sequence.

    Args:
        input_ids: (B, T)
        prompt_len: (B,) number of prompt tokens per sequence
    Returns:
        (B,) sum of log-probs over response tokens
    """
    B, T = input_ids.shape
    with amp_ctx:
        logits, _ = model(input_ids[:, :-1])
    log_p = F.log_softmax(logits, dim=-1)
    token_lp = log_p.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    # Mask: 1 for response tokens, 0 for prompt tokens
    positions = torch.arange(T - 1, device=input_ids.device).unsqueeze(0)  # (1, T-1)
    mask = (positions >= prompt_len.unsqueeze(1)).float()  # (B, T-1)

    return (token_lp * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_lp: torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_chosen_lp: torch.Tensor,
    ref_rejected_lp: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPO loss.

    Returns: (loss, chosen_reward, rejected_reward)
    """
    chosen_logratios = policy_chosen_lp - ref_chosen_lp
    rejected_logratios = policy_rejected_lp - ref_rejected_lp
    logits = beta * (chosen_logratios - rejected_logratios)

    loss = -F.logsigmoid(logits).mean()

    # Implicit rewards (for logging)
    chosen_reward = beta * chosen_logratios.detach()
    rejected_reward = beta * rejected_logratios.detach()

    return loss, chosen_reward.mean(), rejected_reward.mean()


@hydra.main(version_base=None, config_path="../../configs", config_name="dpo")
def main(cfg: DictConfig) -> None:
    rank, world_size = setup_distributed(cfg)
    device = get_device()
    is_distributed = world_size > 1

    torch.manual_seed(cfg.seed + rank)

    if rank == 0:
        print(OmegaConf.to_yaml(cfg))

    # ---- Policy model ----
    model = GPT(cfg.model).to(device)
    if rank == 0:
        print(f"Model: {cfg.model.name} | {model.param_count():,} parameters")

    if cfg.get("pretrained_checkpoint"):
        ckpt = torch.load(cfg.pretrained_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if rank == 0:
            print(f"Loaded pretrained weights from {cfg.pretrained_checkpoint}")

    # ---- Reference model (frozen copy) ----
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad = False

    if not is_distributed and cfg.distributed.fsdp.activation_checkpointing:
        model.enable_activation_checkpointing()

    if is_distributed:
        model = wrap_with_fsdp(model, cfg, device)
        ref_model = wrap_with_fsdp(ref_model, cfg, device)

    if cfg.compile and device.type == "cuda":
        model = torch.compile(model)

    ref_model.eval()

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.95),
    )

    # ---- Data ----
    train_loader = build_preference_dataloader(cfg, rank, world_size)

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

    # ---- DPO config ----
    beta = cfg.training.dpo.beta

    # ---- Training loop ----
    tcfg = cfg.training
    grad_accum = tcfg.gradient_accumulation_steps
    data_iter = iter(train_loader)

    for step in range(start_step, tcfg.max_steps):
        timer.reset()
        model.train()

        lr = get_lr(step, tcfg.warmup_steps, tcfg.max_steps, tcfg.lr, tcfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_chosen_r = 0.0
        accum_rejected_r = 0.0

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
                chosen_ids = batch["chosen_ids"].to(device, non_blocking=True)
                rejected_ids = batch["rejected_ids"].to(device, non_blocking=True)
                prompt_len = batch["prompt_len"].to(device, non_blocking=True)

            with sync_ctx:
                with timer.track("forward"):
                    # Policy log-probs
                    policy_chosen_lp = sequence_log_probs(model, chosen_ids, prompt_len, amp_ctx)
                    policy_rejected_lp = sequence_log_probs(
                        model, rejected_ids, prompt_len, amp_ctx
                    )

                    # Reference log-probs (no grad)
                    with torch.no_grad():
                        ref_chosen_lp = sequence_log_probs(
                            ref_model, chosen_ids, prompt_len, amp_ctx
                        )
                        ref_rejected_lp = sequence_log_probs(
                            ref_model, rejected_ids, prompt_len, amp_ctx
                        )

                    loss, chosen_r, rejected_r = dpo_loss(
                        policy_chosen_lp, policy_rejected_lp,
                        ref_chosen_lp, ref_rejected_lp,
                        beta,
                    )
                    loss = loss / grad_accum

                with timer.track("backward"):
                    loss.backward()

            accum_loss += loss.item()
            accum_chosen_r += chosen_r.item() / grad_accum
            accum_rejected_r += rejected_r.item() / grad_accum

        with timer.track("optimizer"):
            if tcfg.grad_clip > 0:
                if is_distributed:
                    model.clip_grad_norm_(tcfg.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            optimizer.step()

        # ---- Log ----
        reward_margin = accum_chosen_r - accum_rejected_r
        accuracy = 1.0 if reward_margin > 0 else 0.0

        metrics = {
            "dpo/loss": accum_loss,
            "dpo/chosen_reward": accum_chosen_r,
            "dpo/rejected_reward": accum_rejected_r,
            "dpo/reward_margin": reward_margin,
            "dpo/accuracy": accuracy,
            "dpo/lr": lr,
            "dpo/step": step,
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
        print("DPO training complete.")


if __name__ == "__main__":
    main()
