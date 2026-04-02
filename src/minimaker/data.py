"""Data loading — synthetic for testing, HuggingFace for real training."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from omegaconf import DictConfig


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
    """Tokenized + chunked HuggingFace text dataset with disk caching.

    On first use, rank 0 tokenizes the full dataset, splits into train/val,
    and caches both to disk. Other ranks wait for the cache via a barrier.
    """

    def __init__(
        self, cfg: DictConfig, split: str = "train", rank: int = 0, world_size: int = 1
    ):
        self.seq_len = cfg.seq_len
        cache_dir = Path(cfg.cache_dir)
        train_cache = cache_dir / f"{cfg.name}_train.pt"
        val_cache = cache_dir / f"{cfg.name}_val.pt"
        cache_path = train_cache if split == "train" else val_cache

        # Fast path: both caches already exist
        if train_cache.exists() and val_cache.exists():
            self.data = torch.load(cache_path, weights_only=True)
            return

        # Only rank 0 prepares the cache; other ranks wait
        if world_size > 1 and rank != 0:
            dist.barrier()
            self.data = torch.load(cache_path, weights_only=True)
            return

        self._prepare_cache(cfg, train_cache, val_cache)
        self.data = torch.load(cache_path, weights_only=True)

        if world_size > 1:
            dist.barrier()

    def _prepare_cache(
        self, cfg: DictConfig, train_cache: Path, val_cache: Path
    ) -> None:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        print(f"Preparing {cfg.name} dataset cache (this is a one-time cost)...")

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        raw = load_dataset(cfg.path, split="train", trust_remote_code=True)

        # Batch-tokenize via datasets .map() — Arrow-backed, memory efficient
        tokenized = raw.map(
            lambda ex: {"tokens": tokenizer(ex["text"], return_attention_mask=False)["input_ids"]},
            batched=True,
            remove_columns=raw.column_names,
            num_proc=getattr(cfg, "num_workers", 4),
            desc="Tokenizing",
        )

        # Stream through tokenized data and build fixed-length chunks.
        # Periodically flush to tensors to avoid Python-int memory overhead.
        chunk_size = self.seq_len + 1
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
            f"(seq_len={self.seq_len}) to {train_cache.parent}"
        )

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
