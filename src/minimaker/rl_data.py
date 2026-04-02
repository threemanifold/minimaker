"""RL data loading — prompts for GRPO, preference pairs for DPO."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Prompt datasets (for GRPO — just prompts, model generates completions)
# ---------------------------------------------------------------------------

class SyntheticPromptDataset(Dataset):
    """Random prompt tokens for GRPO testing."""

    def __init__(self, cfg: DictConfig):
        self.prompt_len = cfg.prompt_len
        self.data = torch.randint(0, cfg.vocab_size, (cfg.num_samples, cfg.prompt_len))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"prompt_ids": self.data[idx]}


class HuggingFacePromptDataset(Dataset):
    """Prompts from a HuggingFace dataset for GRPO."""

    def __init__(self, cfg: DictConfig, split: str = "train"):
        self.prompt_len = cfg.prompt_len
        cache_path = Path(cfg.cache_dir) / f"{cfg.name}_prompts_{split}.pt"

        if cache_path.exists():
            self.data = torch.load(cache_path, weights_only=True)
            return

        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        raw = load_dataset(cfg.path, split=split, trust_remote_code=True)

        prompt_field = getattr(cfg, "prompt_field", "prompt")
        all_prompts = []
        for example in raw:
            ids = tokenizer.encode(example[prompt_field])[: self.prompt_len]
            pad_len = self.prompt_len - len(ids)
            ids = ids + [tokenizer.pad_token_id] * pad_len
            all_prompts.append(ids)

        self.data = torch.tensor(all_prompts, dtype=torch.long)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.data, cache_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"prompt_ids": self.data[idx]}


# ---------------------------------------------------------------------------
# Preference datasets (for DPO — prompt + chosen + rejected)
# ---------------------------------------------------------------------------

class SyntheticPreferenceDataset(Dataset):
    """Random preference pairs for DPO testing."""

    def __init__(self, cfg: DictConfig):
        seq_len = cfg.seq_len
        prompt_len = seq_len // 3
        resp_len = seq_len - prompt_len

        prompts = torch.randint(0, cfg.vocab_size, (cfg.num_samples, prompt_len))
        chosen = torch.randint(0, cfg.vocab_size, (cfg.num_samples, resp_len))
        rejected = torch.randint(0, cfg.vocab_size, (cfg.num_samples, resp_len))

        # Full sequences: prompt + response
        self.chosen_ids = torch.cat([prompts, chosen], dim=1)
        self.rejected_ids = torch.cat([prompts, rejected], dim=1)
        self.prompt_lens = torch.full((cfg.num_samples,), prompt_len, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.chosen_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "chosen_ids": self.chosen_ids[idx],
            "rejected_ids": self.rejected_ids[idx],
            "prompt_len": self.prompt_lens[idx],
        }


class HuggingFacePreferenceDataset(Dataset):
    """Preference pairs from HuggingFace (e.g. Anthropic hh-rlhf format)."""

    def __init__(self, cfg: DictConfig, split: str = "train"):
        self.seq_len = cfg.seq_len
        cache_path = Path(cfg.cache_dir) / f"{cfg.name}_pref_{split}.pt"

        if cache_path.exists():
            cached = torch.load(cache_path, weights_only=True)
            self.chosen_ids = cached["chosen_ids"]
            self.rejected_ids = cached["rejected_ids"]
            self.prompt_lens = cached["prompt_lens"]
            return

        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        raw = load_dataset(cfg.path, split=split, trust_remote_code=True)

        chosen_field = getattr(cfg, "chosen_field", "chosen")
        rejected_field = getattr(cfg, "rejected_field", "rejected")

        all_chosen, all_rejected, all_prompt_lens = [], [], []
        for example in raw:
            chosen_ids = tokenizer.encode(example[chosen_field])[: self.seq_len]
            rejected_ids = tokenizer.encode(example[rejected_field])[: self.seq_len]

            # Find common prefix length (= prompt)
            prompt_len = 0
            for c, r in zip(chosen_ids, rejected_ids):
                if c == r:
                    prompt_len += 1
                else:
                    break
            prompt_len = max(prompt_len, 1)

            # Pad both to seq_len
            pad_c = self.seq_len - len(chosen_ids)
            pad_r = self.seq_len - len(rejected_ids)
            chosen_ids = chosen_ids + [tokenizer.pad_token_id] * pad_c
            rejected_ids = rejected_ids + [tokenizer.pad_token_id] * pad_r

            all_chosen.append(chosen_ids)
            all_rejected.append(rejected_ids)
            all_prompt_lens.append(prompt_len)

        self.chosen_ids = torch.tensor(all_chosen, dtype=torch.long)
        self.rejected_ids = torch.tensor(all_rejected, dtype=torch.long)
        self.prompt_lens = torch.tensor(all_prompt_lens, dtype=torch.long)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "chosen_ids": self.chosen_ids,
            "rejected_ids": self.rejected_ids,
            "prompt_lens": self.prompt_lens,
        }, cache_path)

    def __len__(self) -> int:
        return len(self.chosen_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "chosen_ids": self.chosen_ids[idx],
            "rejected_ids": self.rejected_ids[idx],
            "prompt_len": self.prompt_lens[idx],
        }


# ---------------------------------------------------------------------------
# Dataloader builders
# ---------------------------------------------------------------------------

def build_prompt_dataloader(
    cfg: DictConfig,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    if cfg.data.type == "synthetic":
        dataset = SyntheticPromptDataset(cfg.data)
    elif cfg.data.type == "huggingface":
        dataset = HuggingFacePromptDataset(cfg.data)
    else:
        raise ValueError(f"Unknown prompt dataset type: {cfg.data.type}")

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


def build_preference_dataloader(
    cfg: DictConfig,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    if cfg.data.type == "synthetic":
        dataset = SyntheticPreferenceDataset(cfg.data)
    elif cfg.data.type == "huggingface":
        dataset = HuggingFacePreferenceDataset(cfg.data)
    else:
        raise ValueError(f"Unknown preference dataset type: {cfg.data.type}")

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
