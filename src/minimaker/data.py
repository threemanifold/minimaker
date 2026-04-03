"""Data loading — synthetic for testing, HuggingFace for real training."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

def _cache_paths(cfg: DictConfig) -> tuple[Path, Path]:
    """Return (train_cache, val_cache) .npy paths for a dataset config."""
    cache_dir = Path(cfg.cache_dir)
    return cache_dir / f"{cfg.name}_train.npy", cache_dir / f"{cfg.name}_val.npy"


def _gcs_paths(cfg: DictConfig) -> tuple[str, str] | None:
    """Return (train_gcs, val_gcs) URIs, or None if GCS is not configured."""
    bucket = getattr(cfg, "gcs_bucket", None)
    if not bucket:
        return None
    bucket = bucket.rstrip("/")
    return f"{bucket}/{cfg.name}_train.npy", f"{bucket}/{cfg.name}_val.npy"


# ---------------------------------------------------------------------------
# GCS helpers (uses gcloud CLI — pre-installed on GCP VMs)
# ---------------------------------------------------------------------------

def _gcs_exists(gcs_path: str) -> bool:
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path],
        capture_output=True,
    )
    return result.returncode == 0


def _gcs_upload(local_path: Path, gcs_path: str) -> None:
    print(f"Uploading {local_path.name} → {gcs_path}")
    subprocess.run(
        ["gcloud", "storage", "cp", str(local_path), gcs_path],
        check=True,
    )


def _gcs_download(gcs_path: str, local_path: Path) -> None:
    print(f"Downloading {gcs_path} → {local_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gcloud", "storage", "cp", gcs_path, str(local_path)],
        check=True,
    )


# ---------------------------------------------------------------------------
# Prepare (download + tokenize + save flat token file)
# ---------------------------------------------------------------------------

def prepare_dataset(cfg: DictConfig) -> None:
    """Download, tokenize, and save a flat token array to disk + optional GCS.

    Safe to call repeatedly — skips if cache already exists.
    Run this *before* training: ``python -m minimaker.data data=openwebtext``
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    train_cache, val_cache = _cache_paths(cfg)
    gcs = _gcs_paths(cfg)

    # Check if already done (local or GCS)
    if train_cache.exists() and val_cache.exists():
        print(f"Cache already exists at {train_cache.parent}, skipping.")
        if gcs and not _gcs_exists(gcs[0]):
            _gcs_upload(train_cache, gcs[0])
            _gcs_upload(val_cache, gcs[1])
        return

    if gcs and _gcs_exists(gcs[0]) and _gcs_exists(gcs[1]):
        print("Cache found in GCS, skipping tokenization.")
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

    # Flatten all tokens into one contiguous array.
    # Process in shards to avoid Arrow 32-bit offset overflow (>2B elements).
    print("Flattening tokens...")
    SHARD_SIZE = 500_000
    parts: list[np.ndarray] = []

    for start in range(0, len(tokenized), SHARD_SIZE):
        end = min(start + SHARD_SIZE, len(tokenized))
        shard = tokenized.select(range(start, end))
        flat = np.array(
            shard.with_format("arrow")["tokens"].combine_chunks().values, copy=False
        )
        parts.append(flat.astype(np.uint16))

    all_tokens = np.concatenate(parts)

    # Hold out last 0.5% of tokens for validation
    val_size = max(1025, int(len(all_tokens) * 0.005))
    train_tokens = all_tokens[:-val_size]
    val_tokens = all_tokens[-val_size:]

    train_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(train_cache, train_tokens)
    np.save(val_cache, val_tokens)

    print(
        f"Cached {len(train_tokens):,} train and {len(val_tokens):,} val tokens "
        f"to {train_cache.parent}"
    )

    # Upload to GCS if configured
    if gcs:
        _gcs_upload(train_cache, gcs[0])
        _gcs_upload(val_cache, gcs[1])


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
    """Memory-mapped flat token file, sliced into chunks on the fly.

    The .npy file is one flat 1-D array of token IDs. Each index maps to a
    contiguous window of ``seq_len + 1`` tokens. Data is read lazily via
    numpy mmap — only accessed pages are loaded into RAM.

    If ``gcs_bucket`` is set and local cache is missing, downloads from GCS first.
    """

    def __init__(
        self, cfg: DictConfig, split: str = "train", rank: int = 0, world_size: int = 1
    ):
        self.seq_len = cfg.seq_len
        train_cache, val_cache = _cache_paths(cfg)
        cache_path = train_cache if split == "train" else val_cache

        # Pull from GCS if local cache is missing
        if not cache_path.exists():
            gcs = _gcs_paths(cfg)
            if gcs:
                gcs_path = gcs[0] if split == "train" else gcs[1]
                # In distributed: only rank 0 downloads, others wait
                if rank == 0:
                    _gcs_download(gcs_path, cache_path)
                if world_size > 1:
                    dist.barrier()

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Tokenized cache not found at {cache_path}. "
                f"Run data preparation first:\n"
                f"  python -m minimaker.data data={cfg.name}"
            )

        self.data = np.load(cache_path, mmap_mode="r")
        self.n_chunks = len(self.data) // (self.seq_len + 1)

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * (self.seq_len + 1)
        tokens = torch.from_numpy(self.data[start : start + self.seq_len + 1].copy()).long()
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
