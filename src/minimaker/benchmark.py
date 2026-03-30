"""Throughput & memory benchmarking — sweep batch sizes, report a table."""

from __future__ import annotations

from contextlib import nullcontext

import hydra
import torch
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

from minimaker.model import GPT
from minimaker.distributed import get_device
from minimaker.metrics import StepTimer


def bench_config(
    cfg: DictConfig,
    device: torch.device,
    batch_size: int,
    warmup: int = 3,
    steps: int = 10,
) -> dict[str, float] | None:
    """Benchmark a single (model, batch_size) combo. Returns None on OOM."""
    seq_len = cfg.data.seq_len
    vocab_size = cfg.model.vocab_size

    try:
        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
        model = GPT(cfg.model).to(device)

        # Mixed precision context
        mp = cfg.training.mixed_precision
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mp)
        use_amp = amp_dtype is not None and device.type in ("cuda", "cpu")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def run_step():
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            ctx = torch.amp.autocast(device.type, dtype=amp_dtype) if use_amp else nullcontext()
            with ctx:
                _, loss = model(x, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Warmup
        for _ in range(warmup):
            run_step()

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        # Measured steps
        timer = StepTimer(device)
        for _ in range(steps):
            timer.reset()
            with timer.track("forward"):
                x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                with torch.amp.autocast(device.type, dtype=amp_dtype) if use_amp else nullcontext():
                    _, loss = model(x, x)

            with timer.track("backward"):
                loss.backward()

            with timer.track("optimizer"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        tokens_total = batch_size * seq_len * steps
        total_sec = timer.total_ms / 1000.0

        result = {
            "batch_size": batch_size,
            "tokens_per_sec": tokens_total / total_sec if total_sec > 0 else 0,
            "step_ms": timer.total_ms / steps,
            "forward_ms": sum(
                v for k, v in timer.timings.items() if k == "forward"
            ) / steps,
            "backward_ms": sum(
                v for k, v in timer.timings.items() if k == "backward"
            ) / steps,
            "optimizer_ms": sum(
                v for k, v in timer.timings.items() if k == "optimizer"
            ) / steps,
        }

        if device.type == "cuda":
            result["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        else:
            result["peak_mem_gb"] = 0.0

        return result

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    finally:
        # Free GPU memory between configs
        del model, optimizer  # noqa: F821
        if device.type == "cuda":
            torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = get_device()
    console = Console()

    console.print(f"\n[bold]Benchmarking model={cfg.model.name} on {device}[/bold]\n")

    batch_sizes = [1, 2, 4, 8, 16, 32]

    table = Table(title=f"Throughput — {cfg.model.name} ({cfg.training.mixed_precision})")
    table.add_column("Batch", justify="right")
    table.add_column("Tokens/s", justify="right")
    table.add_column("Step (ms)", justify="right")
    table.add_column("Fwd (ms)", justify="right")
    table.add_column("Bwd (ms)", justify="right")
    table.add_column("Opt (ms)", justify="right")
    table.add_column("Peak Mem (GB)", justify="right")

    for bs in batch_sizes:
        result = bench_config(cfg, device, bs)
        if result is None:
            table.add_row(str(bs), "[red]OOM[/red]", *["—"] * 5)
            break
        table.add_row(
            str(bs),
            f"{result['tokens_per_sec']:,.0f}",
            f"{result['step_ms']:.1f}",
            f"{result['forward_ms']:.1f}",
            f"{result['backward_ms']:.1f}",
            f"{result['optimizer_ms']:.1f}",
            f"{result['peak_mem_gb']:.2f}",
        )

    console.print(table)


if __name__ == "__main__":
    main()
