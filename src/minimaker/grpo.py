"""GRPO — Group Relative Policy Optimization for reasoning.

Algorithm:
  1. Sample prompts, generate G completions per prompt from policy
  2. Score completions with reward function
  3. Compute group-relative advantages (normalize within each prompt's group)
  4. Policy gradient loss weighted by advantages + KL penalty vs reference
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
from minimaker.rl_data import build_prompt_dataloader
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


def compute_log_probs(
    model: torch.nn.Module, input_ids: torch.Tensor, response_mask: torch.Tensor, amp_ctx
) -> torch.Tensor:
    """Compute per-token log probs for response tokens only.

    Args:
        input_ids: (B, T) full sequences (prompt + response)
        response_mask: (B, T) binary mask, 1 for response tokens
    Returns:
        (B,) sum of log-probs over response tokens per sequence
    """
    with amp_ctx:
        logits, _ = model(input_ids[:, :-1])
    log_p = F.log_softmax(logits, dim=-1)
    token_log_probs = log_p.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    # Mask to only response tokens (shift mask to align with token_log_probs)
    mask = response_mask[:, 1:]
    return (token_log_probs * mask).sum(dim=-1)


@hydra.main(version_base=None, config_path="../../configs", config_name="grpo")
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
    prompt_loader = build_prompt_dataloader(cfg, rank, world_size)

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

    # ---- GRPO config ----
    grpo = cfg.training.grpo
    group_size = grpo.group_size        # G completions per prompt
    max_gen_len = grpo.max_gen_len      # max tokens to generate
    kl_coeff = grpo.kl_coeff            # KL penalty coefficient
    temperature = grpo.temperature      # sampling temperature

    # ---- Training loop ----
    tcfg = cfg.training
    data_iter = iter(prompt_loader)
    policy_model = model.module if is_distributed else model

    for step in range(start_step, tcfg.max_steps):
        timer.reset()
        model.train()

        lr = get_lr(step, tcfg.warmup_steps, tcfg.max_steps, tcfg.lr, tcfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- 1. Sample prompts ----
        with timer.track("data_load"):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(prompt_loader)
                batch = next(data_iter)
            prompt_ids = batch["prompt_ids"].to(device, non_blocking=True)  # (B, P)

        B, P = prompt_ids.shape

        # ---- 2. Generate G completions per prompt ----
        with timer.track("generate"):
            # Repeat each prompt G times: (B*G, P)
            prompts_expanded = prompt_ids.repeat_interleave(group_size, dim=0)

            policy_model_raw = model.module if is_distributed else model
            policy_model_raw.eval()
            with torch.no_grad():
                full_seqs = policy_model_raw.generate(
                    prompts_expanded,
                    max_new_tokens=max_gen_len,
                    temperature=temperature,
                    top_k=50,
                )  # (B*G, P + max_gen_len)
            policy_model_raw.train()

        # Build response mask: 1 for generated tokens, 0 for prompt
        seq_len = full_seqs.size(1)
        response_mask = torch.zeros(B * group_size, seq_len, device=device)
        response_mask[:, P:] = 1.0

        # ---- 3. Compute rewards ----
        with timer.track("reward"):
            # Simple reward: negative perplexity under reference model as proxy
            # (In practice, plug in task-specific rewards from rewards.py)
            with torch.no_grad():
                ref_log_probs = compute_log_probs(ref_model, full_seqs, response_mask, amp_ctx)
            # Sequence-level reward: mean log-prob (higher = more fluent)
            response_lens = response_mask[:, 1:].sum(dim=-1).clamp(min=1)
            rewards = ref_log_probs / response_lens  # (B*G,)

        # ---- 4. Group-relative advantages ----
        rewards_grouped = rewards.view(B, group_size)  # (B, G)
        mean_r = rewards_grouped.mean(dim=1, keepdim=True)
        std_r = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantages = ((rewards_grouped - mean_r) / std_r).view(B * group_size)  # (B*G,)

        # ---- 5. Policy gradient loss ----
        with timer.track("forward"):
            policy_log_probs = compute_log_probs(model, full_seqs, response_mask, amp_ctx)
            # Normalize by response length
            policy_log_probs_normed = policy_log_probs / response_lens

        with timer.track("kl"):
            # KL penalty: KL(π_ref || π_θ) per sequence
            with torch.no_grad():
                ref_lp = compute_log_probs(ref_model, full_seqs, response_mask, amp_ctx)
            kl = (ref_lp - policy_log_probs).mean()

        # GRPO objective: maximize advantage-weighted log-probs minus KL
        pg_loss = -(advantages.detach() * policy_log_probs_normed).mean()
        loss = pg_loss + kl_coeff * kl

        # ---- 6. Update ----
        optimizer.zero_grad(set_to_none=True)
        with timer.track("backward"):
            loss.backward()

        with timer.track("optimizer"):
            if tcfg.grad_clip > 0:
                if is_distributed:
                    model.clip_grad_norm_(tcfg.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            optimizer.step()

        # ---- Log ----
        metrics = {
            "grpo/loss": loss.item(),
            "grpo/pg_loss": pg_loss.item(),
            "grpo/kl": kl.item(),
            "grpo/reward_mean": rewards.mean().item(),
            "grpo/reward_std": rewards.std().item(),
            "grpo/advantage_mean": advantages.mean().item(),
            "grpo/lr": lr,
            "grpo/step": step,
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
        print("GRPO training complete.")


if __name__ == "__main__":
    main()
