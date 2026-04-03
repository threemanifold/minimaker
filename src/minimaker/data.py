"""Data loading — synthetic for testing, HuggingFace for real training."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

def _cache_paths(cfg: DictConfig) -> tuple[Path, Path]:
    """Return (train_cache, val_cache) paths for a dataset config."""
    cache_dir = Path(cfg.cache_dir)
    return cache_dir / f"{cfg.name}_train.pt", cache_dir / f"{cfg.name}_val.pt"


# ---------------------------------------------------------------------------
# Prepare (download + tokenize + chunk + cache)
# ---------------------------------------------------------------------------

def prepare_dataset(cfg: DictConfig) -> None:
    """Download, tokenize, chunk, and cache a HuggingFace dataset.

    Safe to call repeatedly — skips if cache already exists.
    Run this *before* training: ``python -m minimaker.data data=openwebtext``
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    train_cache, val_cache = _cache_paths(cfg)

    if train_cache.exists() and val_cache.exists():
        print(f"Cache already exists at {train_cache.parent}, skipping.")
        return

    print(f"Downloading {cfg.name} dataset...")
    raw = load_dataset(cfg.path, split="train", trust_remote_code=True)

    print(f"Tokenizing with {cfg.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    tokenized = raw.map(
        lambda ex: {"tokens": tokenizer(ex["text"], return_attention_mask=False)["input_ids"]},
        batched=True,
        remove_columns=raw.column_names,
        num_proc=getattr(cfg, "num_workers", 4),
        desc="Tokenizing",
    )

    # Stream through tokenized data and build fixed-length chunks.
    # Periodically flush to tensors to avoid Python-int memory overhead.
    seq_len = cfg.seq_len
    chunk_size = seq_len + 1
    buffer: list[int] = []
    pending: list[list[int]] = []
    tensor_parts: list[torch.Tensor] = []
    FLUSH_EVERY = 10_000

    for example in tokenized:
        buffer.extend(example["tokens"])
        while len(buffer) >= chunk_size:
            pending.append(buffer[:chunk_size])
            buffer = buffer[chunk_size:]

        if len(pending) >= FLUSH_EVERY:
            tensor_parts.append(torch.tensor(pending, dtype=torch.long))
            pending = []

    if pending:
        tensor_parts.append(torch.tensor(pending, dtype=torch.long))

    all_data = torch.cat(tensor_parts, dim=0)

    # Hold out last 0.5% for validation
    n = len(all_data)
    val_size = max(1, int(n * 0.005))
    train_data = all_data[:-val_size]
    val_data = all_data[-val_size:]

    train_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(train_data, train_cache)
    torch.save(val_data, val_cache)

    print(
        f"Cached {len(train_data):,} train and {len(val_data):,} val chunks "
        f"(seq_len={seq_len}) to {train_cache.parent}"
    )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """Random token sequences — zero download, instant startup."""

    def __init__(self, cfg: DictConfig):
        self.seq_len = cfg.seq_len
        self.data = torch.randint(0, cfg.vocab_size, (cfg.num_samples, cfg.seq_len + 1))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.data[idx]
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


class HuggingFaceDataset(Dataset):
    """Loads pre-tokenized chunks from cache. Run ``prepare_dataset`` first."""

    def __init__(
        self, cfg: DictConfig, split: str = "train", rank: int = 0, world_size: int = 1
    ):
        self.seq_len = cfg.seq_len
        train_cache, val_cache = _cache_paths(cfg)
        cache_path = train_cache if split == "train" else val_cache

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Tokenized cache not found at {cache_path}. "
                f"Run data preparation first:\n"
                f"  python -m minimaker.data data={cfg.name}"
            )

        self.data = torch.load(cache_path, weights_only=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.data[idx]
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


def build_dataloader(
    cfg: DictConfig,
    rank: int = 0,
    world_size: int = 1,
    split: str = "train",
) -> DataLoader:
    if cfg.data.type == "synthetic":
        dataset = SyntheticDataset(cfg.data)
    elif cfg.data.type == "huggingface":
        dataset = HuggingFaceDataset(cfg.data, split=split, rank=rank, world_size=world_size)
    else:
        raise ValueError(f"Unknown dataset type: {cfg.data.type}")

    sampler = None
    shuffle = split == "train"
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=getattr(cfg.data, "num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# CLI: python -m minimaker.data data=openwebtext
# ---------------------------------------------------------------------------

import hydra

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.data.type != "huggingface":
        print(f"Nothing to prepare for data.type={cfg.data.type}")
        return
    prepare_dataset(cfg.data)


if __name__ == "__main__":
    main()
