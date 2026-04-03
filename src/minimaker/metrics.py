"""Metrics tracking — timing context manager, W&B / TensorBoard / console backends."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import torch
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class StepTimer:
    """Context-manager timer for per-phase throughput breakdown.

    Usage::

        timer = StepTimer(device)
        with timer.track("data_load"):
            batch = next(loader)
        with timer.track("forward"):
            loss = model(batch)
        # timings accumulate across gradient-accumulation micro-steps
        print(timer.timings)   # {"data_load": 12.3, "forward": 45.6, ...}
        timer.reset()
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.timings: dict[str, float] = {}
        self._use_cuda = device.type == "cuda"

    @contextmanager
    def track(self, name: str):
        if self._use_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            yield
            elapsed = (time.perf_counter() - t0) * 1000.0

        self.timings[name] = self.timings.get(name, 0.0) + elapsed

    def reset(self):
        self.timings.clear()

    @property
    def total_ms(self) -> float:
        return sum(self.timings.values())


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def get_cuda_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "memory/allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
        "memory/reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "memory/peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "memory/peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
        "memory/available_gb": free / 1e9,
        "memory/total_gb": total / 1e9,
    }


# ---------------------------------------------------------------------------
# Logging backends
# ---------------------------------------------------------------------------

class ConsoleBackend:
    """Compact one-line-per-step console output."""

    def log(self, metrics: dict[str, Any], step: int):
        parts = [f"step {step:>6d}"]
        for k, v in metrics.items():
            short = k.rsplit("/", 1)[-1]
            if isinstance(v, float):
                parts.append(f"{short}={v:.4g}")
            else:
                parts.append(f"{short}={v}")
        print(" | ".join(parts))

    def finish(self):
        pass


class TensorBoardBackend:
    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, metrics: dict[str, Any], step: int):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, step)

    def finish(self):
        self.writer.close()


class WandbBackend:
    def __init__(self, cfg: DictConfig):
        import os
        import wandb
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            wandb.login(key=api_key)

        wandb.init(
            project=cfg.logging.project,
            name=cfg.logging.get("run_name"),
            config=dict(cfg),
        )

    def log(self, metrics: dict[str, Any], step: int):
        import wandb

        wandb.log(metrics, step=step)

    def finish(self):
        import wandb

        wandb.finish()


# ---------------------------------------------------------------------------
# Unified tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Routes metrics to all configured backends (rank-0 only)."""

    def __init__(self, cfg: DictConfig, log_dir: str, rank: int = 0):
        self.rank = rank
        self.log_every = cfg.logging.log_every
        self.backends: list = []

        if rank == 0:
            self.backends.append(ConsoleBackend())
            for name in cfg.logging.backends:
                if name == "tensorboard":
                    self.backends.append(TensorBoardBackend(log_dir))
                elif name == "wandb":
                    self.backends.append(WandbBackend(cfg))

    def log(self, metrics: dict[str, Any], step: int):
        if self.rank != 0 or step % self.log_every != 0:
            return
        for backend in self.backends:
            backend.log(metrics, step)

    def finish(self):
        for backend in self.backends:
            backend.finish()
