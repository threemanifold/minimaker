"""SFT data loading — instruction/response pairs with prompt masking."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from omegaconf import DictConfig


class SyntheticSFTDataset(Dataset):
    """Random instruction/response pairs — zero download, instant startup.

    Labels are -100 for prompt tokens (masked from loss), real token ids for response.
    """

    def __init__(self, cfg: DictConfig):
        self.seq_len = cfg.seq_len
        prompt_len = cfg.seq_len // 3  # roughly 1/3 prompt, 2/3 response
        resp_len = cfg.seq_len - prompt_len

        prompts = torch.randint(0, cfg.vocab_size, (cfg.num_samples, prompt_len))
        responses = torch.randint(0, cfg.vocab_size, (cfg.num_samples, resp_len))
        self.input_ids = torch.cat([prompts, responses[:, :-1]], dim=1)

        # Labels: -100 for prompt tokens (no loss), real ids for response tokens
        labels = torch.cat([
            torch.full((cfg.num_samples, prompt_len), -100, dtype=torch.long),
            responses[:, 1:],
        ], dim=1)
        # Trim to seq_len - 1 to match input_ids length
        self.input_ids = self.input_ids[:, : self.seq_len]
        self.labels = labels[:, : self.seq_len]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}


class HuggingFaceSFTDataset(Dataset):
    """Instruction-following dataset from HuggingFace with prompt masking.

    Expects dataset with 'instruction' and 'output' fields (e.g. alpaca format).
    """

    def __init__(self, cfg: DictConfig, split: str = "train"):
        self.seq_len = cfg.seq_len
        cache_path = Path(cfg.cache_dir) / f"{cfg.name}_sft_{split}.pt"

        if cache_path.exists():
            cached = torch.load(cache_path, weights_only=True)
            self.input_ids = cached["input_ids"]
            self.labels = cached["labels"]
            return

        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        raw = load_dataset(cfg.path, split=split, trust_remote_code=True)

        all_input_ids, all_labels = [], []
        instruction_field = getattr(cfg, "instruction_field", "instruction")
        response_field = getattr(cfg, "response_field", "output")

        for example in raw:
            prompt = example[instruction_field]
            response = example[response_field]

            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response, add_special_tokens=False)

            # Concatenate and truncate
            ids = (prompt_ids + response_ids)[: self.seq_len + 1]
            if len(ids) < 4:  # skip very short examples
                continue

            # Build labels: -100 for prompt, real ids for response
            prompt_len = min(len(prompt_ids), self.seq_len)
            labels = [-100] * prompt_len + response_ids[: self.seq_len + 1 - prompt_len]

            # Pad to seq_len + 1
            pad_len = self.seq_len + 1 - len(ids)
            ids = ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

            all_input_ids.append(ids[:-1])
            all_labels.append(labels[1:])

        self.input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        self.labels = torch.tensor(all_labels, dtype=torch.long)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"input_ids": self.input_ids, "labels": self.labels}, cache_path)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}


def build_sft_dataloader(
    cfg: DictConfig,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    if cfg.data.type == "synthetic":
        dataset = SyntheticSFTDataset(cfg.data)
    elif cfg.data.type == "huggingface":
        dataset = HuggingFaceSFTDataset(cfg.data)
    else:
        raise ValueError(f"Unknown SFT dataset type: {cfg.data.type}")

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
