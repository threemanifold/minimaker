"""Data loading — synthetic for testing, HuggingFace for real training."""

from __future__ import annotations

from pathlib import Path

import torch
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
    """Tokenized + chunked HuggingFace text dataset with disk caching."""

    def __init__(self, cfg: DictConfig, split: str = "train"):
        self.seq_len = cfg.seq_len
        cache_path = Path(cfg.cache_dir) / f"{cfg.name}_{split}.pt"

        if cache_path.exists():
            self.data = torch.load(cache_path, weights_only=True)
            return

        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        raw = load_dataset(cfg.path, split=split, trust_remote_code=True)

        # Tokenize all text, concatenate, chunk into fixed-length sequences
        all_tokens: list[int] = []
        for example in raw:
            all_tokens.extend(tokenizer.encode(example["text"]))

        n_chunks = len(all_tokens) // (self.seq_len + 1)
        all_tokens = all_tokens[: n_chunks * (self.seq_len + 1)]
        self.data = torch.tensor(all_tokens, dtype=torch.long).reshape(n_chunks, self.seq_len + 1)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.data, cache_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.data[idx]
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


def build_dataloader(
    cfg: DictConfig,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    if cfg.data.type == "synthetic":
        dataset = SyntheticDataset(cfg.data)
    elif cfg.data.type == "huggingface":
        dataset = HuggingFaceDataset(cfg.data)
    else:
        raise ValueError(f"Unknown dataset type: {cfg.data.type}")

    sampler = None
    shuffle = True
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
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
